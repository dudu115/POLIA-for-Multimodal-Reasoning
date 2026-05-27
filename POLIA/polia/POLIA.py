import os, sys
from typing import Optional

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "trl"))

from rewards import (
    gpt_score_reward,
    bleu_score_reward,
    answer_format_reward,
    repetitive_reward,
    grounded_region_specific_thinking_format_reward_think_rethink,
    think_and_rethink_format_reward,
)
from POLIAtrainer import POLIAtrainer

from accelerate import Accelerator
from relationReasoningDataset import RelationReasoningDataset
accelerator = Accelerator()

import numpy as np
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from trl import (
    ModelConfig,
    GRPOConfig,
    ScriptArguments,
    get_peft_config,
)

# ----------------------------
# Config
# ----------------------------
@dataclass
class VLToolGRPOConfig(GRPOConfig):
    eval_only: bool = field(default=False)
    setting: str = field(default="rr_use_external_grounding_tool")

    project_root_path: str = field(default="")
    python_path_for_dino: str = field(default="")  # 匿名：不暴露本机路径

    train_data_path: str = field(default="")
    train_image_folder_path: str = field(default="")
    eval_data_path: str = field(default="")
    eval_image_folder_path: str = field(default="")

    max_turns: int = field(default=2)
    tool_port_starting_num: int = field(default=8020)

    force_image_size: int = field(default=448)
    use_backbone_lora: int = field(default=0)
    use_llm_lora: int = field(default=0)
    freeze_backbone: bool = field(default=True)
    freeze_llm: bool = field(default=False)
    unfreeze_vit_layers: int = field(default=0)
    freeze_mlp: bool = field(default=False)
    unfreeze_lm_head: bool = field(default=False)

    conv_style: str = field(default="internvl2_5")
    down_sample_ratio: float = field(default=0.5)
    max_eval_samples_per_dataset: Optional[int] = field(default=None)

    beta: float = field(default=0.01)

    iou_weight: float = field(default=0.5)
    l1_weight: float = field(default=0.5)
    subgroup_advantage_weight: float = field(default=1.0)

    gpt_reward_weight: float = field(default=1.5)
    default_reward_weight: float = field(default=1.0)
    gpt_binary_reward: bool = field(default=True)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, VLToolGRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)

    peft_config = get_peft_config(model_args)

    prompt = ""
    prompt_suffix = ""

    if script_args.dataset_name == "rr":
        assert training_args.max_turns == 1

        if "polia" in training_args.setting:
            prompt_suffix = (
                "First, think between <think> and </think> while providing all potentially useful "
                "2D bounding boxes in JSON format under the key 'bbox_2d' as [x_min,y_min,x_max,y_max]. "
                "Then, based on the thinking contents and coordinates, rethink between <rethink> </rethink> "
                "and answer the question using only one word or a single numeric value after <answer>.\n"
            )
        else:
            raise ValueError("Only settings containing 'polia' are supported.")

        tools = ["BoundingboxBrushTool:bbox-brush"]

        # -------- Train dataset --------
        if "," in training_args.train_data_path:
            train_dataset = RelationReasoningDataset(
                training_args.train_data_path.split(","),
                training_args.train_image_folder_path.split(","),
                prompt=prompt,
                prompt_suffix=prompt_suffix,
            )
        else:
            train_dataset = RelationReasoningDataset(
                training_args.train_data_path,
                training_args.train_image_folder_path,
                prompt=prompt,
                prompt_suffix=prompt_suffix,
            )

        
        if "," in training_args.eval_data_path:
            eval_dataset = {}
            for i, (dp, ip) in enumerate(
                zip(
                    training_args.eval_data_path.split(","),
                    training_args.eval_image_folder_path.split(","),
                )
            ):
                eval_dataset[f"eval_set_{i}"] = RelationReasoningDataset(
                    dp,
                    ip,
                    prompt=prompt,
                    prompt_suffix=prompt_suffix,
                    limits=training_args.max_eval_samples_per_dataset,
                )
        else:
            eval_dataset = RelationReasoningDataset(
                training_args.eval_data_path,
                training_args.eval_image_folder_path,
                prompt=prompt,
                prompt_suffix=prompt_suffix,
                limits=training_args.max_eval_samples_per_dataset,
            )

        REWARD_FUNCS_REGISTRY = {
            "answer_gpt_accuracy": gpt_score_reward,
            "answer_blue_score": bleu_score_reward,
            "answer_format_reward": answer_format_reward,
            "repetitive_reward": repetitive_reward,
        }

        if "polia" in training_args.setting:
            REWARD_FUNCS_REGISTRY["JSON_format_reward"] = (
                grounded_region_specific_thinking_format_reward_think_rethink
            )
            REWARD_FUNCS_REGISTRY["think_format_reward"] = think_and_rethink_format_reward

        reward_funcs = list(REWARD_FUNCS_REGISTRY.values())

    else:
        raise ValueError("Unsupported dataset")

    # -------- Resume guard --------
    if os.path.exists(os.path.join(training_args.output_dir, "end_of_training.txt")):
        print("Training already finished. Remove output dir to retrain.")
        exit(0)

    trainer = accelerator.prepare(
        POLIAtrainer(
            args=training_args,
            model=model_args.model_name_or_path,
            tool_names=tools,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            max_eval_samples_per_dataset=training_args.max_eval_samples_per_dataset,
        )
    )

    if training_args.eval_only:
        trainer.evaluate()
    else:
        trainer.train()

    with open(os.path.join(training_args.output_dir, "end_of_training.txt"), "w") as f:
        f.write("Training finished.\n")
