import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt_credentials import api_base, api_key

try:
    from accelerate import Accelerator
    accelerator = Accelerator()
    process_index = accelerator.process_index
except Exception as e:
    process_index = 0
    print(f"Not in distributed training environment: {e}")

API_CALL_LOGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'api_call_logs.txt')
API_ERROR_LOGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'api_error_logs.txt')

from openai import OpenAI

import base64, json
from mimetypes import guess_type


def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    return f"data:{mime_type};base64,{base64_encoded_data}"

import time
import uuid

def call_gpt(prompt, images=None, model="gpt-4o", max_tokens=1000, temperature=0.0):
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    messages = [{"role": "user", "content": []}]
    
    messages[0]["content"].append({"type": "text", "text": prompt})
    
    image_count = len(images) if images else 0
    if images:
        for img_data in images:
            messages[0]["content"].append({"type": "image_url", "image_url": {"url": img_data}})
    
    cred_status = "Credentials available" if api_key else "No credentials found"
    print(f"[Process {process_index}] Starting API call (Request ID: {request_id}), {cred_status}, Images: {image_count}")
    
    client = OpenAI(
        api_key=api_key,
        base_url=api_base
    )
    
    print(f"[Process {process_index}] Total images added: {image_count}")
    print(f"[Process {process_index}] API config: Base URL={api_base}, API Key prefix={api_key[:5]}...")

    try:
        print(f"[Process {process_index}] Sending API request...")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        elapsed_time = time.time() - start_time
        print(f"[Process {process_index}] API call successful (Request ID: {request_id}), Time: {elapsed_time:.2f}s")
        
        response_data = response.model_dump()
        completion_id = response_data.get('id', 'unknown')
        model_used = response_data.get('model', 'unknown')
        usage = response_data.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
        
        print(f"[Process {process_index}] Response details: Completion ID={completion_id}")
        print(f"[Process {process_index}] Model: {model_used}, Tokens: {total_tokens}")
        print(f"[Process {process_index}] Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")
        
        content = response_data['choices'][0]['message']['content']
        print(f"[callGPT-{request_id}] Response length: {len(content)} characters")
        print(f"[callGPT-{request_id}] First 30 characters: {content[:30]}...")
        
        try:
            with open(API_CALL_LOGS_PATH, 'a') as log_file:
                log_file.write(f"[Process {process_index}] [{time.strftime('%Y-%m-%d %H:%M:%S')}] Request ID: {request_id}, Completion ID: {completion_id}, ")
                log_file.write(f"Model: {model_used}, Tokens: {total_tokens}, Time: {elapsed_time:.2f}s\n")
                log_file.flush()
                os.fsync(log_file.fileno())
        except Exception as log_error:
            print(f"[Process {process_index}] Failed to write to log file: {log_error}")
        
        return content
    except Exception as e:
        elapsed_time = time.time() - start_time
        
        error_message = str(e)
        try:
            with open(API_ERROR_LOGS_PATH, 'a') as error_log:
                error_log.write(f"[Process {process_index}] [{time.strftime('%Y-%m-%d %H:%M:%S')}] Request ID: {request_id}, Error: {error_message}\n")
                error_log.flush()
                os.fsync(error_log.fileno())
        except Exception as log_error:
            print(f"[Process {process_index}] Failed to write to error log: {log_error}")
        
        print(f"[Process {process_index}] API call failed (Request ID: {request_id}), Time: {elapsed_time:.2f}s")
        print(f"[Process {process_index}] Error details: {error_message}")
        
        return f"Error: {error_message}"
