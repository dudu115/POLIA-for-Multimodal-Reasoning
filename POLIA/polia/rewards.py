import re
import time

import numpy as np
import json
from collections import defaultdict

import nltk
from nltk.translate.bleu_score import sentence_bleu

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from accelerate import Accelerator
accelerator = Accelerator()
process_index = accelerator.process_index

from callGPT import call_gpt


def gpt_score_reward(prompts, completions, gt_answer, **kwargs):
    rewards = []
    force_api_call = True
    
    for i in range(len(gt_answer)):
        if 'dataset' in kwargs and 'ovd' in kwargs['dataset'][i]:
            rewards.append(0.0)
            continue
        
        response, answer = completions[i], gt_answer[i]
        response_str = str(response) if not isinstance(response, str) else response
        
        question = prompts[i][0]['content'][-1]['text'].split('Please answer question: ')[-1]

        reward = 0.0
        if '<answer>' in response_str or force_api_call:
            match_pattern = response_str.split('<answer>', 1)
            predicted_content = match_pattern[-1].split('</answer>')[0]
            
            prompt = f"""You are responsible for proofreading the answers, you need to give a score to the model's answer by referring to the standard answer, based on the given question. The full score is 1 point and the minimum score is 0 points. Please output the score in the json form "{{score: <score>}}". The evaluation criteria require that the closer the model's answer is to the standard answer, the higher the score.
Question: {question} 
Standard answer: {answer} 
Model's answer: {predicted_content}"""

            output = 0.0
            prompt_with_process = f"[Process {process_index}] {prompt}"
            output = re.sub(r'[^\w\s\.]', '', call_gpt(prompt_with_process).strip().lower())
            output_num = output.split("score")[-1][:5]
            output = re.sub(r'[^\d.]', '', output_num)
            output = float(output)
            reward += output
        rewards.append(reward)
    
    return rewards


def bleu_score_reward(prompts, completions, gt_answer, **kwargs):
    rewards = []
    max_reward = 0.5
    
    for i in range(len(gt_answer)):
        response, answer = completions[i], gt_answer[i]
        
        response_text = response
        if isinstance(response, dict):
            response_text = response.get('text', '')
        elif not isinstance(response, str):
            response_text = str(response)
        
        reward = 0.0
        if '<answer>' in response_text:
            match_pattern = response_text.split('<answer>', 1)
            predicted_content = match_pattern[-1]
            
            predicted_content = re.sub(r'[^a-zA-Z0-9\s]', ' ', predicted_content)
            answer = re.sub(r'[^a-zA-Z0-9\s]', ' ', answer)
            reward += sentence_bleu([answer.lower().split()], predicted_content.lower().split(), weights=(1, 0, 0, 0))
        
        rewards.append(reward * max_reward)
    
    return rewards

def answer_format_reward(prompts, completions, gt_answer, **kwargs):
    """Reward if generated response contains correct answer."""
    rewards = []
    max_reward = 0.5
    keys = ["<answer>"]  # 'Result:', '<submit>' '<request><SimpleCalculatorTool>', '<call>', '<response>'
    
    for response, answer in zip(completions, gt_answer):
        reward = 0.0
        # Extract text from response if it's a dictionary
        response_text = response
        if isinstance(response, dict):
            response_text = response.get('text', '')
        elif not isinstance(response, str):
            response_text = str(response)
        
        for key in keys:
            if key in response_text:
                reward += 1.0
                if len(response_text.split(key)) == 2:
                    reward += 1.0
        
        rewards.append(reward / len(keys) / 2 * max_reward)
    
    return rewards


def think_and_rethink_format_reward(prompts, completions, gt_answer, **kwargs):
    rewards = []
    max_reward = 0.5
    keys = ["<think>", "</think>", "<rethink>", "</rethink>"]
    
    for response, answer in zip(completions, gt_answer):
        response_str = response
        if isinstance(response, dict):
            response_str = response.get('text', '')
        elif not isinstance(response, str):
            response_str = str(response)
        
        reward = 0.0
        response_original = response_str
        temp_response = response_str
        for key in keys:
            if key in temp_response:
                reward += 1.0
                temp_response = temp_response.split(key)[-1]
        
        if reward == len(keys):
            response_list = response_original.split("<think>")[-1].split("</think>")[0].strip()
            if isinstance(response_list, str) and len(response_list) > 1:
                response_list = re.sub(r'[^a-zA-Z0-9\s]', ' ', response_list)
                response_list = response_list.split(' ')
                if len(response_list) > 1:
                    reward += 1.0
        
        rewards.append(reward/(len(keys)+1)*max_reward)
    
    return rewards



def grounded_region_specific_thinking_format_reward_think_rethink(prompts, completions, gt_answer, **kwargs):
    rewards = []
    
    pattern = r'\b(?:\[?)\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:\]?)\b'
    
    for response, answer in zip(completions, gt_answer):
        response_str = response
        if isinstance(response, dict):
            response_str = response.get('text', '')
        elif not isinstance(response, str):
            response_str = str(response)
        
        reward = 0.0
        
        if not '<rethink>' in response_str:
            rewards.append(reward)
            continue
        
        content_before_rethink = response_str.split('<rethink>')[0]
        bbox_matches = re.findall(pattern, content_before_rethink)
        
        if len(bbox_matches) > 0:
            reward += 0.5
        
        if 'dataset' in kwargs and kwargs['dataset'] and len(kwargs['dataset']) > 0:
            if kwargs['dataset'][0] == 'tallyqa' and len(bbox_matches) == int(answer):
                reward += 1

        rewards.append(reward)
    
    return rewards

def zipngram(text: str, ngram_size: int):
    words = text.lower().split()
    return zip(*[words[i:] for i in range(ngram_size)])

def repetitive_reward(prompts, completions, completion_ids, gt_answer, **kwargs):
    ngram_size = 8
    max_reward = 0.5
    rewards = []
    
    pad_token_id = 151643
    
    for i, (ids, completion) in enumerate(zip(completion_ids, completions)):
        completion_text = completion
        if isinstance(completion, dict):
            completion_text = completion.get('text', '')
        elif not isinstance(completion, str):
            completion_text = str(completion)
        
        word_repetition_score = 1.0
        
        if completion_text != "" and len(completion_text.split()) >= ngram_size:
            repeat_count = 0
            total = 0
            tokens = completion_text.split()

            for j in range(len(tokens) - ngram_size):
                ng1 = tuple(tokens[j:j+ngram_size])
                ng2 = tuple(tokens[j+ngram_size:j+ngram_size+ngram_size])
                total += 1
                if ng1 == ng2:
                    repeat_count += 1

            if total > 0:
                word_repetition_score = 1.0 - (repeat_count / total)
        
        ids_list = ids.tolist()
        token_repetition_score = 1.0
        
        if pad_token_id is not None and pad_token_id in ids_list:
            ids_list = ids_list[: ids_list.index(pad_token_id)]
        
        if len(ids_list) >= 2 * ngram_size:
            repeat_count = 0
            total_pairs = 0
            
            for j in range(len(ids_list) - 2*ngram_size + 1):
                ng1 = tuple(ids_list[j : j + ngram_size])
                ng2 = tuple(ids_list[j + ngram_size : j + 2*ngram_size])
                total_pairs += 1
                if ng1 == ng2:
                    repeat_count += 1

            if total_pairs > 0:
                token_repetition_score = 1.0 - (repeat_count / total_pairs)
        
        final_reward = (token_repetition_score - (1-word_repetition_score)) * max_reward
        rewards.append(final_reward)
    
    return rewards
