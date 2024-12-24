import subprocess
import os
import json
import requests
import time
from datetime import datetime
import shutil
import re
from utilities.remove_metadata import remove_metadata_in_place, show_metadata, has_metadata

# -------------------------
# GLOBAL VARIABLES SECTION
# -------------------------

WORKFLOW_PATH = 'test_wf.json'
OUTPUT_FOLDER = 'ComfyUI\\output'
SERVER_ADDRESS = 'http://127.0.0.1:8188'
LOG_FILE = 'comfyui_test_log.txt'
API_OUTPUT_FOLDER = 'api_outputs'
CHECK_INTERVAL = 10
MAX_WAIT_TIME = 1200
MODEL_DIR = 'ComfyUI\\models\\checkpoints'
START_SCRIPT = 'cmd.exe /c start cmd.exe /k cd /d ComfyUI && call comfy_env\\Scripts\\activate && python main.py --lowvram --listen 0.0.0.0'
DELAY_BEFORE_MOVE = 5
PROMPT_TEXT = "a cat in a field of flowers"
INFERENCE_STEPS = 20
REMOVE_METADATA_AFTER = True

# List of all samplers and schedulers to test
SAMPLERS = [
    "euler", "euler_ancestral", "euler_cfg", "euler_ancestral_cfg", "heun", "heun2", 
    "heun2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", 
    "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", 
    "dpm_2", "dpm_2_ancestral", "dpm_2m", "lcm", "i_pndm", "uni_pndm", 
    "uni_pc_bh2", "uni_pc_bh2_ancestral", "uni_pc_bh2m"
]
SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta"]

# Best combinations identified (Green cells in subjective test)
# These combinations produce high-quality images
BEST_COMBINATIONS = [
    ("euler_ancestral", "normal"), ("euler_ancestral", "karras"), ("euler_ancestral", "exponential"), 
    ("euler_ancestral", "sgm_uniform"), ("euler_ancestral", "simple"), ("euler_ancestral", "ddim_uniform"), 
    ("euler_ancestral", "beta"),
    ("heun", "normal"), ("heun", "karras"), ("heun", "exponential"), ("heun", "sgm_uniform"), 
    ("heun", "simple"), ("heun", "ddim_uniform"), ("heun", "beta"),
    ("heun2", "normal"), ("heun2", "karras"), ("heun2", "exponential"), ("heun2", "sgm_uniform"), 
    ("heun2", "simple"), ("heun2", "ddim_uniform"), ("heun2", "beta"),
    ("lms", "normal"), ("lms", "karras"), ("lms", "exponential"), ("lms", "sgm_uniform"), 
    ("lms", "simple"), ("lms", "ddim_uniform"), ("lms", "beta"),
    ("dpmpp_2s_ancestral", "normal"), ("dpmpp_2s_ancestral", "karras"), 
    ("dpmpp_2s_ancestral", "exponential"), ("dpmpp_2s_ancestral", "sgm_uniform"), 
    ("dpmpp_2s_ancestral", "simple"), ("dpmpp_2s_ancestral", "ddim_uniform"), ("dpmpp_2s_ancestral", "beta"),
    ("dpmpp_sde", "normal"), ("dpmpp_sde", "karras"), ("dpmpp_sde", "exponential"), 
    ("dpmpp_sde", "sgm_uniform"), ("dpmpp_sde", "simple"), 
    ("dpmpp_sde", "beta"),
    ("i_pndm", "normal"), ("i_pndm", "karras"), ("i_pndm", "exponential"), 
    ("i_pndm", "sgm_uniform"), ("i_pndm", "simple"), ("i_pndm", "ddim_uniform"), 
    ("i_pndm", "beta"),
    ("uni_pc_bh2", "normal"), ("uni_pc_bh2", "simple"), ("uni_pc_bh2", "beta")
]

# Use this list if not empty; otherwise, fall back to the full combinations of samplers and schedulers
SAMPLER_SCHEDULER_COMBINATIONS = BEST_COMBINATIONS

# -------------------------
# HELPER FUNCTIONS SECTION
# -------------------------

def log(message):
    """ Log a message both to the console and to a file. """
    with open(LOG_FILE, 'a') as f:
        f.write(f"{message}\n")
    print(message)

def start_comfyui():
    """ Start the ComfyUI server in a new terminal window. """
    subprocess.Popen(START_SCRIPT, shell=True)
    log("Started ComfyUI server in a new terminal window.")

def is_comfyui_ready():
    """ Check if the ComfyUI server is ready to receive requests. """
    url = f"{SERVER_ADDRESS}/prompt"
    max_attempts = 10
    attempt_delay = 5
    
    for attempt in range(max_attempts):
        try:
            log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Attempt {attempt + 1}: Checking if ComfyUI is ready at {url}...")
            response = requests.get(url)
            if response.status_code == 200:
                log(f"ComfyUI is ready after {attempt + 1} attempts.")
                return True
            log(f"Status code received: {response.status_code}, Response: {response.text}")
        except requests.exceptions.ConnectionError as e:
            remaining_attempts = max_attempts - attempt - 1
            log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Attempt {attempt + 1}: ComfyUI not ready yet. {remaining_attempts} attempts left. Retrying in {attempt_delay} seconds...")
            log(f"ConnectionError: {e}")
            time.sleep(attempt_delay)
    
    log("ComfyUI is not ready.")
    return False

def queue_prompt(workflow_json):
    """ Queue a workflow prompt in the ComfyUI server. """
    url = f"{SERVER_ADDRESS}/prompt"
    headers = {'Content-Type': 'application/json'}
    data = {'prompt': workflow_json}
    
    log(f"Queueing prompt to {url} with data: {json.dumps(data, indent=4)}")
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        log("Prompt queued successfully.")
        return response.json()
    else:
        log(f"Error {response.status_code}: {response.text}")
        return None

def wait_for_image(output_path, wait_time, check_interval, prefix):
    """ Wait for an image file to appear in the output directory. """
    log(f"Waiting for an image with prefix '{prefix}' to appear in {output_path}...")
    total_wait_time = 0
    
    while total_wait_time < wait_time:
        pattern = re.compile(f"{prefix}.*.png")
        found_files = os.listdir(output_path)
        
        log(f"Current files in directory: {found_files}")
        for file in found_files:
            if pattern.match(file):
                log(f"New image '{file}' has been created in {output_path}.")
                return file
        
        log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] '{prefix}' not created yet. Rechecking in {check_interval} seconds...")
        time.sleep(check_interval)
        total_wait_time += check_interval
    
    log(f"Timeout: No image starting with '{prefix}' was created within {wait_time} seconds.")
    return None

def move_and_rename_image(src_path, dest_dir, new_filename):
    """ Move and rename the image file to the api_outputs directory. """
    os.makedirs(dest_dir, exist_ok=True)
    
    dest_path = os.path.join(dest_dir, new_filename)
    
    time.sleep(DELAY_BEFORE_MOVE)  # Wait before moving the file to ensure it is not being used
    
    shutil.move(src_path, dest_path)
    log(f"Image moved and renamed to {dest_path}")

def create_filename(prompt_text, sampler_name, scheduler):
    """ Create a descriptive filename based on the given criteria. """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filtered_prompt = ''.join(c if c.isalnum() else '_' for c in prompt_text.lower())
    prompt_segment = filtered_prompt[:10]
    filename = f"{timestamp}_{prompt_segment}_{sampler_name}_{scheduler}.png"
    return filename, filename.replace('.png', '')  # Return both filename and prefix

def remove_metadata_if_required(file_path):
    """Remove metadata from the image if REMOVE_METADATA_AFTER is set to True."""
    if REMOVE_METADATA_AFTER:
        if has_metadata(file_path):
            log(f"Removing metadata from {file_path}...")
            show_metadata(file_path)
            remove_metadata_in_place(file_path)
            show_metadata(file_path)

# -------------------------
# MAIN EXECUTION SECTION
# -------------------------

def main():
    """ Main function to execute the workflow. """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(API_OUTPUT_FOLDER, exist_ok=True)

    log("Checking if ComfyUI is already running...")
    if not is_comfyui_ready():
        log("ComfyUI is not running. Starting ComfyUI in a new terminal window.")
        start_comfyui()

        log("Waiting for ComfyUI to start...")
        if not is_comfyui_ready():
            log("ComfyUI did not start in time.")
            return
    else:
        log("ComfyUI is already running.")
    
    log(f"Loading workflow from {WORKFLOW_PATH}...")
    with open(WORKFLOW_PATH, 'r') as file:
        workflow_json = json.load(file)
    log("Workflow loaded.")

    workflow_json["6"]["inputs"]["text"] = PROMPT_TEXT
    log(f"Updated prompt text to '{PROMPT_TEXT}' in the workflow.")

    total_start_time = time.time()
    total_files = 0

    # Ensure there are combinations to process
    combinations_to_process = SAMPLER_SCHEDULER_COMBINATIONS if SAMPLER_SCHEDULER_COMBINATIONS else [(s, sc) for s in SAMPLERS for sc in SCHEDULERS]

    for sampler, scheduler in combinations_to_process:
        # Update the inference steps, sampler, and scheduler in the workflow
        workflow_json["3"]["inputs"]["steps"] = INFERENCE_STEPS
        workflow_json["3"]["inputs"]["sampler_name"] = sampler
        workflow_json["3"]["inputs"]["scheduler"] = scheduler
        log(f"Updated inference steps to '{INFERENCE_STEPS}', sampler to '{sampler}', and scheduler to '{scheduler}' in the workflow.")

        model_name = workflow_json["4"]["inputs"]["ckpt_name"]
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            log(f"Model file '{model_name}' not found in 'checkpoints' directory.")
            return
        log(f"Model file '{model_name}' found in 'checkpoints' directory.")

        filename, filename_prefix = create_filename(PROMPT_TEXT, sampler, scheduler)
        workflow_json["9"]["inputs"]["filename_prefix"] = filename_prefix
        log(f"Updated filename prefix to '{filename_prefix}' in the workflow.")
    
        log("Queueing the prompt...")
        start_time = time.time()
        response = queue_prompt(workflow_json)
        if not response:
            log("Failed to queue the prompt.")
            continue
        prompt_id = response.get('prompt_id')
        if not prompt_id:
            log("No prompt ID received.")
            continue
        log(f"Prompt queued successfully with ID: {prompt_id}")

        new_file = wait_for_image(OUTPUT_FOLDER, MAX_WAIT_TIME, CHECK_INTERVAL, filename_prefix)
        end_time = time.time()
        time_taken = end_time - start_time
        
        if new_file:
            log(f"Image created successfully: {os.path.join(OUTPUT_FOLDER, new_file)}")
            output_path = os.path.join(OUTPUT_FOLDER, new_file)
            dest_path = os.path.join(API_OUTPUT_FOLDER, filename)
            move_and_rename_image(output_path, API_OUTPUT_FOLDER, filename)
            log(f"Time taken for creation: {time_taken:.2f} seconds")
            total_files += 1

            # Remove metadata if required
            remove_metadata_if_required(dest_path)
        else:
            log("Image creation failed or timed out.")
        
    total_end_time = time.time()
    total_time_taken = total_end_time - total_start_time
    average_time_per_file = total_time_taken / total_files if total_files > 0 else 0

    log(f"Total time taken for all combinations: {total_time_taken:.2f} seconds")
    log(f"Average time per file creation: {average_time_per_file:.2f} seconds")

    log("Completed successfully!")

if __name__ == "__main__":
    log("Clearing the log file...")
    with open(LOG_FILE, 'w') as f:
        f.write("")
    main()