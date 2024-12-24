import os
import json
import requests
import time
from datetime import datetime, timedelta
import shutil
import random
import traceback
import re
import torch
import gc
from utilities.remove_metadata import remove_metadata_in_place, show_metadata, has_metadata
from utilities.comfy_starter import initialize_comfyui

# -------------------------
# GLOBAL VARIABLES SECTION
# -------------------------

WORKFLOW_PATH = 'workflow_json\\superhero_creator.json'
OUTPUT_FOLDER = 'ComfyUI\\output'
SERVER_ADDRESS = 'http://127.0.0.1:8188'
LOG_FILE = 'superhero_test_log.txt'
API_OUTPUT_FOLDER = 'api_outputs'
CHECK_INTERVAL = 60
MAX_WAIT_TIME = 3600
MODEL_DIRS = {
    'checkpoints': 'ComfyUI\\models\\checkpoints',
    'unet': 'ComfyUI\\models\\unet',
    'vae': 'ComfyUI\\models\\vae',
    'clip': 'ComfyUI\\models\\clip',
    'loras': 'ComfyUI\\models\\loras'
}
DELAY_BEFORE_MOVE = 5
PROMPT_TEXT = "(photorealistic:1.8), (ultra realistic:1.7) superhero man!"
INFERENCE_STEPS = 28
REMOVE_METADATA_AFTER = True

LORA1 = "flux_realism_lora.safetensors"
LORA2 = "andy.safetensors"
LORA3 = "Dever_Flux_Enhancer.safetensors"

LORA1_WEIGHT = 0.7
LORA2_WEIGHT = 1.0
LORA3_WEIGHT = 0.5

DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"

UNET_FILENAME = "flux1-dev-fp8.safetensors"
CLIP1_FILENAME = "t5xxl_fp8_e4m3fn.safetensors"
CLIP2_FILENAME = "clip_l.safetensors"
VAE_FILENAME = "ae.safetensors"

REPEAT_LATENT_BATCH_AMOUNT = 3
NUMBER_OF_LOOPS = 10

USE_ALL_CONFIGS = True

# List of samplers and schedulers to test
SAMPLERS = [
    "euler", "euler_ancestral", "euler_cfg", "euler_ancestral_cfg", "heun", "heun2", 
    "heun2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", 
    "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", 
    "dpm_2", "dpm_2_ancestral", "dpm_2m", "lcm", "i_pndm", "uni_pndm", 
    "uni_pc_bh2", "uni_pc_bh2_ancestral", "uni_pc_bh2m"
]
SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta"]

# -------------------------
# HELPER FUNCTIONS SECTION
# -------------------------

def log(message):
    """ Log a message both to the console and to a file. """
    with open(LOG_FILE, 'a') as f:
        f.write(f"{message}\n")
    print(message)

def log_error(message):
    """ Log an error message with traceback both to the console and to a file. """
    with open(LOG_FILE, 'a') as f:
        f.write(f"{message}\n")
        f.write(traceback.format_exc() + "\n")
    print(message)
    print(traceback.format_exc())

def queue_prompt(workflow_json):
    """ Queue a workflow prompt in the ComfyUI server with error handling. """
    url = f"{SERVER_ADDRESS}/prompt"
    headers = {'Content-Type': 'application/json'}
    data = {'prompt': workflow_json}

    try:
        log(f"Queueing prompt to {url} with data: {json.dumps(data, indent=4)}")
        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            log("Prompt queued successfully.")
            return response.json()
        else:
            log(f"Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        log_error(f"An exception occurred while queuing the prompt: {str(e)}")
        return None

def wait_for_images(output_path, wait_time, check_interval, prefix, expected_count, loop, start_time_total, start_time_loop, sampler_name, scheduler_name):
    """ Wait for multiple image files to appear in the output directory. """
    log(f"Waiting for {expected_count} images with prefix '{prefix}' to appear in {output_path}...")
    total_wait_time = 0
    found_files = []

    while total_wait_time < wait_time:
        pattern = re.compile(f"{prefix}.*.png")
        current_files = os.listdir(output_path)

        for file in current_files:
            if pattern.match(file) and file not in found_files:
                found_files.append(file)
                log(f"New image '{file}' has been created in {output_path}.")
        
        if len(found_files) >= expected_count:
            break

        current_time = datetime.now()
        total_elapsed_time = current_time - start_time_total
        loop_elapsed_time = current_time - start_time_loop

        estimated_time_remaining_str = "N/A"
        
        if USE_ALL_CONFIGS:
            total_combinations = len(SAMPLERS) * len(SCHEDULERS)
            completed_combinations = (loop - 1) * REPEAT_LATENT_BATCH_AMOUNT / REPEAT_LATENT_BATCH_AMOUNT + expected_count / REPEAT_LATENT_BATCH_AMOUNT
            estimated_time_remaining = (time.time() - start_time_total.timestamp()) / completed_combinations * (total_combinations - completed_combinations)
            estimated_time_remaining_str = str(timedelta(seconds=int(estimated_time_remaining)))

        log(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {len(found_files)}/{expected_count} images created in loop {loop} of {NUMBER_OF_LOOPS}. Still creating, rechecking in {check_interval} seconds...")
        log(f"Prompt we're creating: {PROMPT_TEXT}")
        log(f"Inference steps: {INFERENCE_STEPS}")
        log(f"Loras in use: {LORA1}, {LORA2}, {LORA3}")
        log(f"Sampler Name: {sampler_name}")
        log(f"Scheduler Name: {scheduler_name}")
        log(f"Running time since start of script: {str(total_elapsed_time)}")
        log(f"Running time for this image creation set: {str(loop_elapsed_time)}")
        log(f"Images remaining: {total_combinations - int(completed_combinations)}")
        log(f"Estimated time remaining: ~{estimated_time_remaining_str}")
        log(f"================")

        time.sleep(check_interval)
        total_wait_time += check_interval

    if len(found_files) >= expected_count:
        log(f"All {expected_count} images created successfully.")
    else:
        log(f"Timeout: Only {len(found_files)} out of {expected_count} images were created within {wait_time} seconds.")

    return found_files

def move_and_rename_images(src_path, dest_dir, new_filename_prefix, num_images):
    """ Move and rename the image files to the api_outputs directory with indexed filenames. """
    os.makedirs(dest_dir, exist_ok=True)
    moved_files = []

    for i in range(1, num_images + 1):
        original_filename = f"{new_filename_prefix}_{i:05d}_.png"
        dest_filename = f"{new_filename_prefix}_{i:05d}_.png"
        src_filepath = os.path.join(src_path, original_filename)
        dest_filepath = os.path.join(dest_dir, dest_filename)
        
        time.sleep(DELAY_BEFORE_MOVE)  # Wait before moving the file to ensure it is not being used

        try:
            shutil.move(src_filepath, dest_filepath)
            moved_files.append(dest_filepath)
            log(f"Image moved and renamed to {dest_filepath}")
        except FileNotFoundError:
            log(f"Failed to move file: {src_filepath} not found.")

    return moved_files

def create_filename_prefix(prompt_text, sampler_name, scheduler):
    """ Create a descriptive filename prefix based on the given criteria. """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filtered_prompt = ''.join(c if c.isalnum() else '_' for c in prompt_text.lower())
    prompt_segment = filtered_prompt[:10]
    return f"{timestamp}_{prompt_segment}_{sampler_name}_{scheduler}"

def remove_metadata_if_required(file_path):
    """Remove metadata from the image if REMOVE_METADATA_AFTER is set to True."""
    if REMOVE_METADATA_AFTER:
        if has_metadata(file_path):
            log(f"Removing metadata from {file_path}...")
            show_metadata(file_path)
            remove_metadata_in_place(file_path)
            show_metadata(file_path)

def check_model_file(model_filename, category):
    """Check if the model file exists in the given directory."""
    model_path = os.path.join(MODEL_DIRS[category], model_filename)
    if not os.path.exists(model_path):
        log(f"Model file '{model_filename}' not found in '{MODEL_DIRS[category]}' directory.")
        return False
    log(f"Model file '{model_filename}' found in '{MODEL_DIRS[category]}' directory.")
    return True

def clear_vram():
    """ Clear VRAM and collect garbage. """
    torch.cuda.empty_cache()
    gc.collect()

def execute_workflow_loop(loop, total_start_time, workflow_json, sampler_name, scheduler_name):
    loop_start_time = datetime.now()
    try:
        log(f"Starting loop {loop}/{NUMBER_OF_LOOPS}...")

        if not all([
            check_model_file(UNET_FILENAME, 'unet'),
            check_model_file(CLIP1_FILENAME, 'clip'),
            check_model_file(CLIP2_FILENAME, 'clip'),
            check_model_file(VAE_FILENAME, 'vae'),
            check_model_file(LORA1, 'loras'),
            check_model_file(LORA2, 'loras'),
            check_model_file(LORA3, 'loras')]):
            return False

        new_seed = random.randint(0, 2**32 - 1)
        workflow_json["25"]["inputs"]["noise_seed"] = new_seed
        log(f"Set new random seed to {new_seed}.")

        workflow_json["6"]["inputs"]["text"] = PROMPT_TEXT
        workflow_json["10"]["inputs"]["vae_name"] = VAE_FILENAME
        workflow_json["11"]["inputs"]["clip_name1"] = CLIP1_FILENAME
        workflow_json["11"]["inputs"]["clip_name2"] = CLIP2_FILENAME
        workflow_json["12"]["inputs"]["unet_name"] = UNET_FILENAME
        workflow_json["16"]["inputs"]["sampler_name"] = sampler_name
        workflow_json["17"]["inputs"]["scheduler"] = scheduler_name
        workflow_json["17"]["inputs"]["steps"] = INFERENCE_STEPS
        workflow_json["41"]["inputs"]["amount"] = REPEAT_LATENT_BATCH_AMOUNT
        workflow_json["38"]["inputs"]["lora_name"] = LORA1
        workflow_json["38"]["inputs"]["strength_model"] = LORA1_WEIGHT
        workflow_json["42"]["inputs"]["lora_name"] = LORA2
        workflow_json["42"]["inputs"]["strength_model"] = LORA2_WEIGHT
        workflow_json["43"]["inputs"]["lora_name"] = LORA3
        workflow_json["43"]["inputs"]["strength_model"] = LORA3_WEIGHT

        filename_prefix = create_filename_prefix(PROMPT_TEXT, sampler_name, scheduler_name)
        workflow_json["9"]["inputs"]["filename_prefix"] = filename_prefix
        log(f"Updated filename prefix to '{filename_prefix}' in the workflow.")

        log("Queueing the prompt...")
        start_time = datetime.now()
        response = queue_prompt(workflow_json)
        if not response:
            log("Failed to queue the prompt.")
            return False
        prompt_id = response.get('prompt_id')
        if not prompt_id:
            log("No prompt ID received.")
            return False

        log(f"Prompt queued successfully with ID: {prompt_id}")

        new_files = wait_for_images(OUTPUT_FOLDER, MAX_WAIT_TIME, CHECK_INTERVAL, filename_prefix, REPEAT_LATENT_BATCH_AMOUNT, loop, total_start_time, loop_start_time, sampler_name, scheduler_name)
        end_time = datetime.now()
        time_taken = end_time - start_time

        moved_files = move_and_rename_images(OUTPUT_FOLDER, API_OUTPUT_FOLDER, filename_prefix, REPEAT_LATENT_BATCH_AMOUNT)
        for file in moved_files:
            remove_metadata_if_required(file)

        log(f"Time taken for creation: {time_taken} for {len(moved_files)} files")
        return True

    except Exception as e:
        log_error(f"Exception occurred during loop {loop}: {str(e)}")
        return False

# -------------------------
# MAIN EXECUTION SECTION
# -------------------------

def main():
    """ Main function to execute the workflow with error handling. """
    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(API_OUTPUT_FOLDER, exist_ok=True)

        log("Initializing ComfyUI...")
        if not initialize_comfyui():
            log("Failed to initialize ComfyUI.")
            return

        log(f"Loading workflow from {WORKFLOW_PATH}...")
        with open(WORKFLOW_PATH, 'r') as file:
            workflow_json = json.load(file)
        log("Workflow loaded.")

        total_start_time = datetime.now()
        total_files = 0

        if USE_ALL_CONFIGS:
            # Loop through all sampler and scheduler combinations
            all_configs = [(sampler, scheduler) for sampler in SAMPLERS for scheduler in SCHEDULERS]
            total_combinations = len(all_configs)
            
            for idx, (sampler, scheduler) in enumerate(all_configs):
                log(f"Testing with sampler: {sampler} and scheduler: {scheduler}")
                start_time = time.time()
                success = execute_workflow_loop(idx + 1, total_start_time, workflow_json, sampler, scheduler)
                time_taken_this_set = time.time() - start_time
                
                images_remaining = total_combinations * REPEAT_LATENT_BATCH_AMOUNT - total_files
                estimated_time_remaining = (time_taken_this_set * (images_remaining / REPEAT_LATENT_BATCH_AMOUNT))
                estimated_time_remaining_str = str(timedelta(seconds=int(estimated_time_remaining)))

                if not success:
                    log(f"Failed to create image with sampler: {sampler} and scheduler: {scheduler}")
                    continue

                total_files += REPEAT_LATENT_BATCH_AMOUNT
                clear_vram()

        else:
            # Default behavior with single sampler and scheduler
            for loop in range(1, NUMBER_OF_LOOPS + 1):
                success = execute_workflow_loop(loop, total_start_time, workflow_json, DEFAULT_SAMPLER, DEFAULT_SCHEDULER)
                if not success:
                    log(f"Loop {loop} failed, stopping.")
                    break

                total_files += REPEAT_LATENT_BATCH_AMOUNT
                # Clear VRAM after each loop
                clear_vram()

        total_end_time = datetime.now()
        total_time_taken = total_end_time - total_start_time
        average_time_per_file = total_time_taken / total_files if total_files > 0 else timedelta(0)

        log(f"Total time taken for all loops: {total_time_taken}")
        log(f"Average time per file creation: {average_time_per_file}")
        log("Completed successfully!")

    except Exception as e:
        log_error(f"Exception occurred in main: {str(e)}")


if __name__ == "__main__":
    log("Clearing the log file...")
    with open(LOG_FILE, 'w') as f:
        f.write("")
    main()