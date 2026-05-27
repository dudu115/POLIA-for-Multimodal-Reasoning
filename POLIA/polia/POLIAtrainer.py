import os
import json
import time
import random
import base64
import io
import re
import warnings
from copy import deepcopy
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union

import numpy as np
from PIL import Image

import torch
import torch.utils.data
from torch import nn
from torch.utils.data import Sampler

import transformers
from packaging import version
from datasets import Dataset, IterableDataset

from accelerate.utils import (
    broadcast_object_list,
    gather,
    gather_object,
)
from accelerate.utils.other import is_compiled_module

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoProcessor,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    load_tool,
)

from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.import_utils import is_vllm_available
from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    selective_log_softmax,
)

from vision_process import process_vision_info

# Optional dependencies
if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams


RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset N times.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`):
            Number of times to repeat each index.

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d"], repeat_count=2)
    >>> list(sampler)
    [2, 2, 0, 0, 3, 3, 1, 1]
    ```
    """

    def __init__(self, data_source: Sized, repeat_count: int):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)

    def __iter__(self):
        indexes = [idx for idx in torch.randperm(self.num_samples).tolist() for _ in range(self.repeat_count)]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count


class POLIAtrainer(Trainer):
    # Anonymous-safe: drop explicit project tag
    _tag_names = ["trl"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        tool_names: list[str],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 256 * 28 * 28,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        # Anonymous-safe: do not hardcode a public model id as default
        tool_model_id: Optional[str] = os.environ.get("POLIA_TOOL_MODEL_ID", ""),
        max_eval_samples_per_dataset: Optional[int] = None,
    ):
        self.max_eval_samples_per_dataset = max_eval_samples_per_dataset
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if attn_implementation == "flash_attention_2" and "torch_dtype" not in model_init_kwargs:
            model_init_kwargs["torch_dtype"] = torch.bfloat16

        def _load_optimizer_and_scheduler(self, checkpoint_path):
            optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
            scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")

            if os.path.exists(optimizer_path):
                self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.accelerator.device))
                print("Optimizer state loaded from checkpoint.")

            if os.path.exists(scheduler_path):
                self.lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.accelerator.device))
                print("Scheduler state loaded from checkpoint.")

        assert isinstance(model, str), f"model must be a string, but got {type(model)}"
        model_id = model
        self.model_id = model_id
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass
        elif isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)
            model_init_kwargs["torch_dtype"] = torch_dtype
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )

        if "qwen" in model_id.lower():
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
        else:
            raise ValueError("Only qwen models are supported.")

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        if is_deepspeed_zero3_enabled():
            if "qwen" in model_id.lower():
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                raise ValueError("Only qwen models are supported for reference model.")
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        if processing_class is None:
            if "qwen" in model_id.lower():
                try:
                    processing_class = AutoProcessor.from_pretrained(model_id)
                except Exception:
                    processing_class = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

                processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                processing_class.image_processor.max_pixels = max_pixels
                processing_class.image_processor.min_pixels = min_pixels
            else:
                raise ValueError("Only qwen models are supported for processing class.")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):
            return features

        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.use_vllm = args.use_vllm
        self.beta = args.beta

        # IOU and L1 weight parameters
        self.iou_weight = getattr(args, "iou_weight", 0.7)
        self.l1_weight = getattr(args, "l1_weight", 0.3)

        # Subgroup advantage weight parameter
        self.subgroup_advantage_weight = getattr(args, "subgroup_advantage_weight", 1.0)

        # Reward function weights
        self.gpt_reward_weight = getattr(args, "gpt_reward_weight", 1.5)
        self.default_reward_weight = getattr(args, "default_reward_weight", 1.0)

        self.gpt_binary_reward = getattr(args, "gpt_binary_reward", True)

        model.warnings_issued["estimate_tokens"] = True

        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        assert self.use_vllm is False

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            pad_token_id=processing_class.pad_token_id,
        )
        self.model_accepts_loss_kwargs = False

        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        self.max_turns = self.args.max_turns
        self.max_tool_response = 30
        self.tool_model_id = tool_model_id

        self.process_uinque_id = int(self.accelerator.process_index) + self.args.tool_port_starting_num
        assert len(tool_names) == 1, "Only one tool is supported for now"
        self.tool_func = tool_names[0].split(":")[0]
        self.tools = {}
        for tool_name in tool_names:
            assert len(tool_name.split(":")) == 2
            tool_func = tool_name.split(":")[0]
            tool_path = tool_name.split(":")[1]
            self.tools[tool_func] = load_tool(
                f"./custom_tools/{tool_path}",
                port=self.process_uinque_id,
                project_root_path=self.args.project_root_path,
                python_path_for_dino=self.args.python_path_for_dino,
            )

        to_phrase = "Wait, I need to think again. "
        from_phrase = "<rethink>\n"

        if "qwen" in model_id.lower():
            self.eos_token_id = processing_class.tokenizer.eos_token_id
            self.pad_token_id = processing_class.tokenizer.pad_token_id
        else:
            raise ValueError("Only qwen models are supported.")

        self.is_train = True

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["message"]

    def _get_train_sampler(self) -> Sampler:
        return RepeatRandomSampler(self.train_dataset, self.num_generations)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        return RepeatRandomSampler(eval_dataset, 2)

    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw, logits_to_keep):
        if "qwen" in self.model_id.lower():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        else:
            raise ValueError("Only qwen models are supported.")
        logits = outputs.logits
        logits = logits[:, -logits_to_keep - 1 : -1, :]

        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)

    def parse_grounding(self, text):
        """
        Parse grounding string, supporting coordinate formats:
        1. Simple format: 100, 200, 300, 400
        2. JSON format: {"bbox_2d": [100, 200, 300, 400]}
        3. Bracketed format: [100, 200, 300, 400]
        4. Sentence format: "The coordinates are 100, 200, 300, 400."
        """
        coords = []

        json_match = re.search(r'\{[^}]*"bbox_2d"[^}]*\}', text)
        if json_match:
            json_str = json_match.group()
            json_data = json.loads(json_str)
            if "bbox_2d" in json_data:
                bboxes = json_data["bbox_2d"]
                if isinstance(bboxes, list):
                    if bboxes and isinstance(bboxes[0], (list, tuple)) and len(bboxes[0]) == 4:
                        coords.extend(bboxes)
                    elif isinstance(bboxes, (list, tuple)) and len(bboxes) == 4:
                        coords.append(bboxes)

        coord_pattern = r"\b(?:\[?)\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:\]?)\b"
        coord_matches = re.findall(coord_pattern, text)

        for match in coord_matches:
            x1, y1, x2, y2 = map(int, match)
            new_box = [x1, y1, x2, y2]
            if new_box not in coords:
                coords.append(new_box)

        return coords

    def calculate_iou(self, box1, box2):
        """
        Calculate IOU between two bounding boxes
        """
        if not isinstance(box1, (list, tuple)) or len(box1) != 4:
            return 0
        if not isinstance(box2, (list, tuple)) or len(box2) != 4:
            return 0

        try:
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
        except (TypeError, ValueError):
            return 0

        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def calculate_l1(self, box1, box2, image_size):
        """
        Calculate L1 similarity between two bounding boxes
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        center1_x = (x1_1 + x2_1) / 2
        center1_y = (y1_1 + y2_1) / 2
        center2_x = (x1_2 + x2_2) / 2
        center2_y = (y1_2 + y2_2) / 2

        l1 = abs(center1_x - center2_x) + abs(center1_y - center2_y)

        diagonal = (image_size[0] ** 2 + image_size[1] ** 2) ** 0.5
        return 1.0 - (l1 / diagonal)

    def flatten_rewards(self, rewards):
        """
        Flatten reward list, handling nested lists
        """
        flattened = []
        stack = [rewards]

        while stack:
            current = stack.pop()
            if isinstance(current, list):
                stack.extend(reversed(current))
            elif isinstance(current, (int, float)):
                flattened.append(current)

        return flattened

    def match_bboxes_to_groundtruth(self, pred_boxes, groundtruth_boxes):
        """
        Match predicted bounding boxes to groundtruth boxes
        """
        matched_ids = []

        if not pred_boxes or not isinstance(pred_boxes, (list, tuple)):
            return matched_ids

        if not groundtruth_boxes or not isinstance(groundtruth_boxes, dict):
            return ["nomatch"] * len(pred_boxes)

        for pred_box in pred_boxes:
            best_iou = 0
            best_id = "nomatch"

            for gt_id, gt_box in groundtruth_boxes.items():
                iou = self.calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_id = gt_id

            iou_threshold = 0.5
            if best_iou < iou_threshold:
                best_id = "nomatch"

            matched_ids.append(best_id)

        return matched_ids

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]], is_train=True) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        self.is_train = is_train

        total_completion_mask = torch.zeros(len(inputs), self.max_completion_length, device=device)
        total_completion_ids = torch.full(
            (len(inputs), self.max_completion_length), self.processing_class.pad_token_id, device=device
        )
        total_completion_pointers = np.zeros((len(inputs),), dtype=int)
        finish_flags = np.zeros((len(inputs),), dtype=bool)

        j = -1
        while True:
            j += 1

            if j == 0:
                original_prompts = [x["message"] for x in inputs]
                inputs_tobe_updated_each_turn = inputs.copy()
            if "qwen" in self.model_id.lower():
                # Process inputs
                text = [
                    self.processing_class.apply_chat_template(
                        inp["message"],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for inp in inputs_tobe_updated_each_turn
                ]
                image_inputs, video_inputs = [], []

                image_inputs, video_inputs = [], []
                for inp in inputs_tobe_updated_each_turn:
                    image_input, video_input = process_vision_info(inp["message"])
                    if image_input:
                        image_inputs.append(image_input)
                    if video_input:
                        video_inputs.append(video_input)

                for i in range(len(text)):
                    if len(text[i].split("<|im_end|>\n<|im_start|>assistant\n")) > 2:
                        t_list = text[i].split("<|im_end|>\n<|im_start|>assistant\n")[:-1]
                        text[i] = "<|im_end|>\n<|im_start|>assistant\n".join(t_list[:2]) + "".join(t_list[2:])

                video_list = [vi for vid_inps in video_inputs for vi in vid_inps]
                if len(video_list) == 0:
                    video_list = None
                image_list = [ii for img_inps in image_inputs for ii in img_inps]
                if len(image_list) == 0:
                    image_list = None

                prompt_inputs = self.processing_class(
                    text=text,
                    images=image_list,
                    videos=video_list,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                )
            else:
                raise ValueError("Only qwen models are supported.")

            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            latest_prompt_ids, latest_prompt_mask, latest_pixel_values = (
                prompt_inputs["input_ids"],
                prompt_inputs["attention_mask"],
                prompt_inputs.get("pixel_values"),
            )

            if latest_pixel_values is not None:
                model_dtype = next(self.model.parameters()).dtype
                if latest_pixel_values.dtype != model_dtype:
                    prompt_inputs["pixel_values"] = latest_pixel_values.to(model_dtype)
                    latest_pixel_values = prompt_inputs["pixel_values"]

            latest_image_grid_thw = prompt_inputs.get("image_grid_thw")

            if self.max_turns == j:
                break

            if j == 0:
                if latest_prompt_ids.size(1) > self.max_prompt_length:
                    print(
                        f"Warning: prompt length {latest_prompt_ids.size(1)} exceeds the maximum prompt length {self.max_prompt_length}. Truncating."
                    )
                original_prompt_ids = latest_prompt_ids.clone()
                original_prompt_mask = latest_prompt_mask.clone()

                if self.generation_config.stop_strings is None:
                    self.generation_config.stop_strings = ["</answer>"]
                elif "</answer>" not in self.generation_config.stop_strings:
                    self.generation_config.stop_strings += ["</answer>"]
                grounded_images = []

            p_mask = latest_prompt_mask

            start_time = time.time()
            if "qwen" in self.model_id.lower():
                with torch.no_grad():
                    if self.is_train:
                        prompt_completion_ids = self.model.generate(
                            **prompt_inputs,
                            generation_config=self.generation_config,
                            tokenizer=self.processing_class.tokenizer,
                        )
                    else:
                        eval_generation_config = deepcopy(self.generation_config)
                        eval_generation_config.temperature = 0.001
                        eval_generation_config.do_sample = True
                        eval_generation_config.top_k = 1
                        eval_generation_config.top_p = 0.0
                        prompt_completion_ids = self.model.generate(
                            **prompt_inputs,
                            tokenizer=self.processing_class.tokenizer,
                            generation_config=eval_generation_config,
                        )

                prompt_length = latest_prompt_ids.size(1)
                if not (latest_prompt_ids == prompt_completion_ids[:, :prompt_length]).all():
                    print("Prompt mismatch, unexpected.")
                completion_ids = prompt_completion_ids[:, prompt_length:]
            else:
                raise ValueError("Only qwen models are supported.")

            time_taken = time.time() - start_time
            print(f"\n[Timer]: \nTime taken for generation: {time_taken:.2f} seconds.\n")

            start_time = time.time()

            is_eos = completion_ids == self.eos_token_id
            is_pad = completion_ids == self.pad_token_id
            is_eos = is_eos | is_pad
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

            for i in range(len(completion_ids)):
                if total_completion_pointers[i] < self.max_completion_length and not finish_flags[i]:
                    left_tk_space = (
                        min(self.max_completion_length, total_completion_pointers[i] + eos_idx[i])
                        - total_completion_pointers[i]
                    )
                    total_completion_ids[i, total_completion_pointers[i] : (total_completion_pointers[i] + left_tk_space)] = completion_ids[i, :left_tk_space]
                    total_completion_mask[i, total_completion_pointers[i] : (total_completion_pointers[i] + left_tk_space)] = completion_mask[i, :left_tk_space]
                    total_completion_pointers[i] += eos_idx[i]
                    completion = self.processing_class.decode(completion_ids[i][: eos_idx[i]], skip_special_tokens=True)
                    if "assistant" != inputs_tobe_updated_each_turn[i]["message"][-1]["role"]:
                        inputs_tobe_updated_each_turn[i]["message"].append({"role": "assistant", "content": []})
                    inputs_tobe_updated_each_turn[i]["message"][-1]["content"] += [{"type": "text", "text": completion}]
                    print(i, "_", j, completion)

                    tool_call_query = self.parse_grounding(completion)

                    image_with_bbox = None
                    response = ""
                    if not tool_call_query:
                        finish_flags[i] = True
                    else:
                        if self.args.log_completions and "eval" in inputs[i] and inputs[i]["eval"]:
                            buffered = io.BytesIO()
                            image_inputs[i][-1].save(buffered, format="PNG")
                            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                            try:
                                image_base64, response = self.tools[self.tool_func](
                                    img_str=img_str, tool_call_query=tool_call_query, normalized_bboxs=False
                                )

                                image_with_bbox = base64.b64decode(image_base64)
                                image_with_bbox = Image.open(io.BytesIO(image_with_bbox))
                                if "crop" in self.args.setting:
                                    bbox = re.search(r"\d+, \d+, \d+, \d+", response)[0]
                                    bbox = [int(x) for x in bbox.split(", ")]
                                    image_with_bbox = image_with_bbox.crop(bbox)

                            except Exception as error:
                                if response == "":
                                    response = f"Failure: {str(error)}"

                    if image_with_bbox:
                        grounded_images.append(image_with_bbox)
                    else:
                        grounded_images.append(None)

                    print("tool response: ", response)

        for i in range(len(completion_ids)):
            if total_completion_pointers[i] < self.max_completion_length:
                total_completion_ids[i, total_completion_pointers[i]] = self.eos_token_id
                total_completion_mask[i, total_completion_pointers[i]] = 1
                total_completion_pointers[i] += 1

        completion_ids = total_completion_ids
        completion_mask = total_completion_mask

        is_eos = completion_ids == self.eos_token_id
        is_pad = completion_ids == self.pad_token_id
        is_eos = is_eos | is_pad
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask_no_tool = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        attention_mask = torch.cat([original_prompt_mask, completion_mask_no_tool], dim=1)
        prompt_completion_ids = torch.cat([original_prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)

        if is_train:
            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        latest_pixel_values,
                        latest_image_grid_thw,
                        logits_to_keep,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            latest_pixel_values,
                            latest_image_grid_thw,
                            logits_to_keep,
                        )
        else:
            ref_per_token_logps = None

        time_taken = time.time() - start_time
        print(f"\n[Timer]: \nTime taken for tool call and logit probability computation: {time_taken:.2f} seconds.\n")

        start_time = time.time()
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions_text]
        else:
            completions = completions_text

        processed_completions = []
        for completion in completions_text:
            pred_bboxes = self.parse_grounding(completion)
            processed_completions.append({"text": completion, "pred_bboxes": pred_bboxes})

        rewards_per_func = torch.zeros(len(original_prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            reward_start_time = time.time()
            if isinstance(reward_func, nn.Module):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(original_prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(original_prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                keys = [key for key in inputs[0] if key not in ["message", "completion"]]
                reward_kwargs = {key: [example.get(key) for example in inputs] for key in keys}

                reward_kwargs["is_train"] = is_train
                output_reward_func = reward_func(
                    prompts=original_prompts,
                    completions=processed_completions,
                    completion_ids=completion_ids,
                    **reward_kwargs,
                )
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

            reward_time = time.time() - reward_start_time
            # reward_func may be nn.Module without __name__
            func_name = getattr(reward_func, "__name__", reward_func.__class__.__name__)
            print(f"\n[Timer]: \nTime taken for reward function {func_name}: {reward_time:.2f} seconds.\n")

        current_batch_rewards = rewards_per_func.clone()
        rewards_per_func = gather(rewards_per_func)

        idx_gpt_reward = None
        idx_grounded_region_specific_thinking_format_reward = None
        idx_grounded_region_bbox_IOU_loss = None
        idx_grounded_region_bbox_repetitive_loss = None
        idx_answer_format_reward = None

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            if "grounded_region_specific_thinking_format_reward" in reward_func_name:
                idx_grounded_region_specific_thinking_format_reward = i
            elif "gpt" in reward_func_name:
                idx_gpt_reward = i
            elif "grounded_region_bbox_IOU_loss" in reward_func_name:
                idx_grounded_region_bbox_IOU_loss = i
            elif "grounded_region_bbox_repetitive_loss" in reward_func_name:
                idx_grounded_region_bbox_repetitive_loss = i
            elif "answer_format" in reward_func_name:
                idx_answer_format_reward = i

        if self.gpt_binary_reward and idx_gpt_reward is not None and is_train:
            rewards_per_func[:, idx_gpt_reward] = torch.where(rewards_per_func[:, idx_gpt_reward] > 0, 1.0, 0.0)

        if idx_grounded_region_bbox_IOU_loss is not None:
            rewards_per_func[:, idx_grounded_region_bbox_IOU_loss] = 0 * rewards_per_func[:, idx_grounded_region_bbox_IOU_loss]

        original_rewards = rewards_per_func.sum(dim=1)

        weighted_rewards_per_func = rewards_per_func.clone()
        for i, reward_func in enumerate(self.reward_funcs):
            if hasattr(reward_func, "__name__") and "gpt_score_reward" in reward_func.__name__:
                weighted_rewards_per_func[:, i] *= self.gpt_reward_weight
            else:
                weighted_rewards_per_func[:, i] *= self.default_reward_weight

        rewards = weighted_rewards_per_func.sum(dim=1)

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

       
        local_bs = len(original_prompts)  
        start = self.accelerator.process_index * local_bs
        end = (self.accelerator.process_index + 1) * local_bs

        
        g_start = start * self.num_generations
        g_end = end * self.num_generations

        rewards = rewards[g_start:g_end]
        advantages = advantages[g_start:g_end]

        has_gt_boxes = any(("groundtruth_boxes" in x) for x in inputs)
        has_bboxs = any(("bboxs" in x) for x in inputs)

        subgroup_groups = []
        if has_gt_boxes or has_bboxs:
            completions_per_prompt = []
            for i in range(0, len(completions_text), self.num_generations):
                completions_per_prompt.append(completions_text[i : i + self.num_generations])

            pred_bboxes_per_prompt = []
            for i in range(0, len(processed_completions), self.num_generations):
                prompt_pred_bboxes = [completion["pred_bboxes"] for completion in processed_completions[i : i + self.num_generations]]
                pred_bboxes_per_prompt.append(prompt_pred_bboxes)

            for prompt_idx in range(len(completions_per_prompt)):
               
                sample = inputs[prompt_idx]
            
                groundtruth_boxes = sample.get("groundtruth_boxes", {})
             
                if not groundtruth_boxes and has_bboxs:
                    bboxs_list = sample.get("bboxs", [])
                    groundtruth_boxes = {str(i): bbox for i, bbox in enumerate(bboxs_list)}

                prompt_completions = completions_per_prompt[prompt_idx]
                prompt_pred_bboxes = pred_bboxes_per_prompt[prompt_idx]

                completion_groups = {}
                for comp_idx, (completion, pred_bboxes) in enumerate(zip(prompt_completions, prompt_pred_bboxes)):
                 
                    image_size = sample.get("image_size", (224, 224)) if isinstance(sample, dict) else (224, 224)
                    pred_boxes = pred_bboxes
                    matched_ids = self.match_bboxes_to_groundtruth(pred_boxes, groundtruth_boxes)
                    object_set = tuple(sorted([id for id in matched_ids if id != "nomatch"]))

                    if object_set not in completion_groups:
                        completion_groups[object_set] = []
                    completion_groups[object_set].append((comp_idx, pred_boxes, matched_ids, pred_bboxes))

                subgroup_groups.append(completion_groups)

        subgroup_advantages = advantages.clone()
        all_matched_ids = []
        per_object_subgroup_advantages = []

        for prompt_idx, completion_groups in enumerate(subgroup_groups):
        
            sample = inputs[prompt_idx]
         
            groundtruth_boxes = sample.get("groundtruth_boxes", {})
          
            if not groundtruth_boxes and has_bboxs:
                bboxs_list = sample.get("bboxs", [])
                groundtruth_boxes = {str(i): bbox for i, bbox in enumerate(bboxs_list)}
          
            image_size = sample.get("image_size", (224, 224)) if isinstance(sample, dict) else (224, 224)
            
            for object_set, group_items in completion_groups.items():
                if len(group_items) > 1:
                    comp_indices = [item[0] for item in group_items]
                    group_rewards = rewards.view(-1, self.num_generations)[prompt_idx, comp_indices]

                per_object_rewards = {}
                completion_per_object_rewards = [{} for _ in range(len(group_items))]

                for item_idx, (comp_idx, pred_boxes, matched_ids, pred_bboxes) in enumerate(group_items):
                    global_idx = prompt_idx * self.num_generations + comp_idx
                    
                    if len(group_items) > 1:
                        response_reward = group_rewards[item_idx]
                    else:
                        response_reward = rewards[global_idx]

                    for obj_idx, (pred_box, gt_id, pred_bbox) in enumerate(zip(pred_boxes, matched_ids, pred_bboxes)):
                        if gt_id != "nomatch":
                            gt_box = groundtruth_boxes[gt_id]
                            iou = self.calculate_iou(pred_box, gt_box)
                            l1_similarity = self.calculate_l1(pred_box, gt_box, image_size)
                            match_score = (self.iou_weight * iou + self.l1_weight * l1_similarity)
                            match_score = max(0.0, min(1.0, match_score))
                            object_reward = response_reward * match_score

                            if gt_id not in per_object_rewards:
                                per_object_rewards[gt_id] = []
                            per_object_rewards[gt_id].append(object_reward)

                            completion_per_object_rewards[item_idx][gt_id] = object_reward

                    if len(all_matched_ids) <= global_idx:
                        all_matched_ids.extend([[]] * (global_idx + 1 - len(all_matched_ids)))
                    all_matched_ids[global_idx] = matched_ids

                    if len(per_object_subgroup_advantages) <= global_idx:
                        per_object_subgroup_advantages.extend([{} for _ in range(global_idx + 1 - len(per_object_subgroup_advantages))])
                    per_object_subgroup_advantages[global_idx] = completion_per_object_rewards[item_idx]

               
                if len(group_items) > 1:
                    for gt_id, obj_rewards in per_object_rewards.items():
                        obj_rewards = torch.tensor(obj_rewards, device=rewards.device)
                        mean_obj_reward = obj_rewards.mean()
                        std_obj_reward = obj_rewards.std()
                        obj_advantages = (obj_rewards - mean_obj_reward) / (std_obj_reward + 1e-4)
                        for item_idx, comp_idx in enumerate(comp_indices):
                            global_idx = prompt_idx * self.num_generations + comp_idx
                            
                            if gt_id in per_object_subgroup_advantages[global_idx]:
                                per_object_subgroup_advantages[global_idx][gt_id] = float(obj_advantages[item_idx])

                for item_idx, (comp_idx, pred_boxes, matched_ids, pred_bboxes) in enumerate(group_items):
                    nomatch_count = sum(1 for gt_id in matched_ids if gt_id == "nomatch")
                    if nomatch_count > 0:
                        global_idx = prompt_idx * self.num_generations + comp_idx
                        subgroup_advantages[global_idx] += (-1.0 * nomatch_count)

            advantages = subgroup_advantages

        reward_per_func = current_batch_rewards.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"{inputs[0]['dataset']}/rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics[f"{inputs[0]['dataset']}/reward"].append(original_rewards.mean().item())
        self._metrics[f"{inputs[0]['dataset']}/reward_std"].append(std_grouped_rewards.mean().item())

        time_taken = time.time() - start_time
        print(f"\n[Timer]: \nTime taken for reward computation: {time_taken:.2f} seconds.\n")

        matched_ids = all_matched_ids if "all_matched_ids" in locals() else []

        if not matched_ids and any(("groundtruth_boxes" in x) for x in inputs):
            for global_idx, completion in enumerate(processed_completions):
            
                prompt_idx = global_idx // self.num_generations
                if prompt_idx < len(inputs):
                    sample = inputs[prompt_idx]
                    groundtruth_boxes = sample.get("groundtruth_boxes", {})
                    if not groundtruth_boxes and has_bboxs:
                        bboxs_list = sample.get("bboxs", [])
                        groundtruth_boxes = {str(i): bbox for i, bbox in enumerate(bboxs_list)}
                    image_size = sample.get("image_size", (224, 224)) if isinstance(sample, dict) else (224, 224)
                    pred_bboxes = completion["pred_bboxes"]
                    pred_boxes = pred_bboxes
                    matched_ids.append(self.match_bboxes_to_groundtruth(pred_boxes, groundtruth_boxes))
        reward_names = []
        for reward_func in self.reward_funcs:
            if isinstance(reward_func, nn.Module):
                name = reward_func.config._name_or_path.split("/")[-1]
            else:
                name = reward_func.__name__ if hasattr(reward_func, "__name__") else str(reward_func)
            reward_names.append(name)

        coordinate_token_positions = []
        for i, completion in enumerate(completions_text):
            json_match = re.search(r'\{[^}]*"bbox_2d"[^}]*\}', completion)

            if json_match:
                json_str = json_match.group()
                coord_pattern = r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]"
                coord_matches = re.findall(coord_pattern, json_str)

                if coord_matches:
                    coord_positions = []
                    pred_bboxes = []
                    if i < len(processed_completions):
                        pred_bboxes = processed_completions[i].get("pred_bboxes", [])
                    matched_ids_for_completion = []
                    if i < len(all_matched_ids):
                        matched_ids_for_completion = all_matched_ids[i]
                    
                    search_from = 0
                    for idx, coord_match in enumerate(coord_matches):
                        coord_start = completion.find(coord_match, search_from)
                        if coord_start != -1:
                            coord_end = coord_start + len(coord_match)
                            gt_id = "nomatch"
                            if idx < len(matched_ids_for_completion):
                                gt_id = matched_ids_for_completion[idx]
                            coord_positions.append({"span": (coord_start, coord_end), "gt_id": gt_id})

                            search_from = coord_end

                    if coord_positions:
                        coordinate_token_positions.append(coord_positions)
                    else:
                        coordinate_token_positions.append(None)
                else:
                    coordinate_token_positions.append(None)
            else:
                coordinate_token_positions.append(None)

        return_content = {
            "step": str(self.state.global_step),
            "input_output_text": inputs_tobe_updated_each_turn,
            "reward_name": reward_names,
            "reward_list": current_batch_rewards.tolist(),
            "raw_inputs": inputs,
            "image_inputs": image_inputs,
            "grounded_images": grounded_images,
            "prompt_ids": original_prompt_ids,
            "prompt_mask": original_prompt_mask,
            "pixel_values": latest_pixel_values,
            "image_grid_thw": latest_image_grid_thw,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "matched_ids": matched_ids,
            "coordinate_token_positions": coordinate_token_positions,
            "per_object_subgroup_advantages": per_object_subgroup_advantages,
            "completions_text": completions_text,
        }

        return return_content

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the loss for POLIA training.

        Args:
            model: The model to train
            inputs: Input dictionary containing prompt_ids, completion_ids, etc.
            return_outputs: Whether to return outputs (not supported)
            num_items_in_batch: Number of items in the batch

        Returns:
            The computed loss
        """
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        start_time = time.time()

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        pixel_values, image_grid_thw = inputs["pixel_values"], inputs["image_grid_thw"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        is_eos = completion_ids == self.processing_class.eos_token_id
        is_pad = completion_ids == self.processing_class.pad_token_id
        is_eos = is_eos | is_pad
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=self.accelerator.device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=self.accelerator.device).expand(is_eos.size(0), -1)
        completion_mask_no_tool = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        attention_mask = torch.cat([prompt_mask, completion_mask_no_tool], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, pixel_values, image_grid_thw, logits_to_keep
        )

        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        response_advantages = inputs["advantages"]
        per_token_advantages = response_advantages.unsqueeze(1).repeat(1, per_token_logps.size(1))

        coordinate_token_positions = inputs.get("coordinate_token_positions", [])
        subgroup_advantage_weight = getattr(self, "subgroup_advantage_weight", 1.0)
        per_object_subgroup_advantages = inputs.get("per_object_subgroup_advantages", [])
        completions_text = inputs.get("completions_text", [])

        for i, coord_positions in enumerate(coordinate_token_positions):
            if coord_positions is not None and i < len(per_token_advantages):
                if i < len(completions_text):
                    completion = completions_text[i]
                    completion_length = len(completion)

                    if completion_length > 0:
                        if i < len(per_object_subgroup_advantages):
                            obj_advantages = per_object_subgroup_advantages[i]

                            for item in coord_positions:
                                start_pos, end_pos = item["span"]
                                gt_id = item["gt_id"]
                                coord_length = end_pos - start_pos
                                coord_ratio = coord_length / completion_length

                                num_tokens = per_token_advantages[i].size(0)
                                coord_token_count = max(1, int(num_tokens * coord_ratio))
                                coord_token_count = min(coord_token_count, num_tokens)

                                if coord_token_count > 0:
                                    if gt_id in obj_advantages:
                                        subgroup_advantage = obj_advantages.get(gt_id, 0.0) * subgroup_advantage_weight
                                    else:
                                        if obj_advantages:
                                            avg_obj_advantage = sum(obj_advantages.values()) / len(obj_advantages)
                                            subgroup_advantage = avg_obj_advantage * subgroup_advantage_weight
                                        else:
                                            subgroup_advantage = response_advantages[i] * subgroup_advantage_weight

                                    text_position_ratio = start_pos / completion_length
                                    start_token = max(0, int(num_tokens * text_position_ratio))
                                    end_token = min(num_tokens, start_token + coord_token_count)

                                    per_token_advantages[i, start_token:end_token] += subgroup_advantage
                        else:
                            for item in coord_positions:
                                start_pos, end_pos = item["span"]
                                coord_length = end_pos - start_pos
                                coord_ratio = coord_length / completion_length

                                num_tokens = per_token_advantages[i].size(0)
                                coord_token_count = max(1, int(num_tokens * coord_ratio))
                                coord_token_count = min(coord_token_count, num_tokens)

                                if coord_token_count > 0:
                                    subgroup_advantage = response_advantages[i] * subgroup_advantage_weight

                                    text_position_ratio = start_pos / completion_length
                                    start_token = max(0, int(num_tokens * text_position_ratio))
                                    end_token = min(num_tokens, start_token + coord_token_count)

                                    per_token_advantages[i, start_token:end_token] += subgroup_advantage

        # Calculate importance ratio and clipped objective for PPO
        importance_ratio = torch.exp(per_token_logps - ref_per_token_logps)
        epsilon = getattr(self.args, "clip_epsilon", 0.2)
        clipped_ratio = torch.clamp(importance_ratio, 1 - epsilon, 1 + epsilon)
        surrogate_objective = torch.min(importance_ratio * per_token_advantages, clipped_ratio * per_token_advantages)
        per_token_loss = surrogate_objective

        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        time_taken = time.time() - start_time
        print(f"\n[Timer]: \nTime taken for loss computation: {time_taken:.2f} seconds.\n")
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        if not hasattr(self, "_eval_progress") and not self.is_train:
            self._eval_progress = {
                "total_samples": 0,
                "start_time": time.time(),
                "current_step": 0,
            }

        new_inputs = []
        existed_inputs_image_answer_question = []
        for i in range(len(inputs)):
            inputs_image_answer_question = (
                inputs[i]["image"] + inputs[i]["gt_answer"] + inputs[i]["message"][0]["content"][-1]["text"]
            )
            if inputs_image_answer_question not in existed_inputs_image_answer_question:
                existed_inputs_image_answer_question.append(inputs_image_answer_question)
                new_inputs.append(inputs[i])
        inputs = new_inputs

        if (
            (not self.is_train)
            and self.max_eval_samples_per_dataset is not None
            and self.max_eval_samples_per_dataset > 0
        ):
            if len(inputs) > self.max_eval_samples_per_dataset:
                original_len = len(inputs)
                random.seed(42)
                inputs = random.sample(inputs, self.max_eval_samples_per_dataset)
                if self.accelerator.is_main_process:
                    print(f"[Eval Data Sampling] Sampled {len(inputs)} samples from {original_len} total samples")

        for i in range(len(inputs)):
            inputs[i]["eval"] = True

        inputs = self._prepare_inputs(inputs, is_train=False)

        for j in range(len(inputs["input_output_text"])):
            for i in range(len(inputs["input_output_text"][j]["message"][1]["content"])):
                if (
                    "type" in inputs["input_output_text"][j]["message"][1]["content"][i]
                    and inputs["input_output_text"][j]["message"][1]["content"][i]["type"] == "image"
                ):
                    inputs["input_output_text"][j]["message"][1]["content"][i]["image"] = "New image with bounding box."

        _ = [
            inputs["image_inputs"][i] + ([inputs["grounded_images"][i]] if inputs["grounded_images"][i] else [])
            for i in range(len(inputs["image_inputs"]))
        ]

        # Anonymous review: no external logging (e.g., wandb). Keep local progress prints only.
        if self.accelerator.is_main_process and (not self.is_train) and hasattr(self, "_eval_progress"):
            flattened_rewards = self.flatten_rewards(inputs["reward_list"]) if inputs.get("reward_list") else []
            self._eval_progress["total_samples"] += len(flattened_rewards)
            self._eval_progress["current_step"] = inputs["step"]

            elapsed_time = time.time() - self._eval_progress["start_time"]
            samples_per_sec = self._eval_progress["total_samples"] / elapsed_time if elapsed_time > 0 else 0
            batch_reward = (sum(flattened_rewards) / len(flattened_rewards)) if flattened_rewards else 0.0

            print(
                f"Eval Progress - Step: {inputs['step']}, "
                f"Samples Processed: {self._eval_progress['total_samples']}, "
                f"Batch Reward: {batch_reward:.4f}, "
                f"Processing Speed: {samples_per_sec:.2f} samples/sec"
            )

        if self.args.output_dir:
            os.makedirs(self.args.output_dir, exist_ok=True)

            eval_summary_path = os.path.join(self.args.output_dir, f"eval_summary_step_{inputs['step']}.json")
            flattened_rewards = self.flatten_rewards(inputs["reward_list"]) if inputs.get("reward_list") else []

            eval_summary = {
                "step": inputs["step"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mean_reward": (sum(flattened_rewards) / len(flattened_rewards)) if flattened_rewards else 0,
                "num_samples": len(flattened_rewards),
                "reward_name": inputs.get("reward_name", "unknown"),
                "process_index": self.accelerator.process_index,
            }

            with open(eval_summary_path, "w") as f:
                json.dump(eval_summary, f, indent=2)

            eval_detail_path = os.path.join(self.args.output_dir, f"eval_detail_step_{inputs['step']}.json")

            eval_details = []
            for i in range(len(inputs["input_output_text"])):
                raw_i = inputs["raw_inputs"][i] if i < len(inputs.get("raw_inputs", [])) else None

                # Anonymous-safe: do not leak full local paths.
                image_path = None
                if raw_i and isinstance(raw_i, dict) and ("image" in raw_i) and raw_i["image"] is not None:
                    try:
                        image_path = os.path.basename(str(raw_i["image"]))
                    except Exception:
                        image_path = None

                sample_data = {
                    "sample_id": i,
                    "input_output_text": inputs["input_output_text"][i] if i < len(inputs["input_output_text"]) else None,
                    "reward": inputs["reward_list"][i] if i < len(inputs.get("reward_list", [])) else None,
                    "raw_input": raw_i,
                    "image_path": image_path,
                    "question": (raw_i.get("question") if raw_i else None),
                    "gt_answer": (raw_i.get("gt_answer") if raw_i else None),
                }
                eval_details.append(sample_data)

            with open(eval_detail_path, "w") as f:
                json.dump(eval_details, f, indent=2)

        if self.is_train:
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs)
                loss = loss.mean().detach()
            return loss, None, None
        else:
            return torch.tensor(0.0, device=self.accelerator.device), None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}

        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

