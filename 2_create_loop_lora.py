import os
import json
import requests
import time
from datetime import datetime, timedelta
import shutil
import random
import torch
import gc
from utilities.lora_utils import update_lora_metadata, cleanse_prompt
from utilities.remove_metadata import remove_metadata_in_place, show_metadata, has_metadata
from utilities.comfy_starter import initialize_comfyui
from utilities.image_creation_utils import wait_for_images, move_and_rename_images, create_filename_prefix, remove_metadata_if_required
from utilities.logging_utils import log, log_error, log_iteration_details

# Load configurations from global_variables.json
def load_configurations():
    with open('global_variables.json', 'r', encoding='utf-8') as f:
        configs = json.load(f)
        return {key: value['value'] for key, value in configs.items()}

config = load_configurations()

WORKFLOW_PATH = config['WORKFLOW_PATH']
OUTPUT_FOLDER = config['OUTPUT_FOLDER']
SERVER_ADDRESS = config['SERVER_ADDRESS']
LOG_FILE = config['LOG_FILE']
API_OUTPUT_FOLDER = config['API_OUTPUT_FOLDER']
ITERATION_LOG_FILE = config['ITERATION_LOG_FILE']
CHECK_INTERVAL = config['CHECK_INTERVAL']
MAX_WAIT_TIME = config['MAX_WAIT_TIME']
MODEL_DIRS = config['MODEL_DIRS']
DELAY_BEFORE_MOVE = config['DELAY_BEFORE_MOVE']
LORA_DIRECTORY = config['LORA_DIRECTORY']

INFERENCE_STEPS = config['INFERENCE_STEPS']
REMOVE_METADATA_AFTER = config['REMOVE_METADATA_AFTER']

LORA1 = config['LORA1_NAME']
LORA1_WEIGHT = config['LORA1_WEIGHT']
LORA1_CLIP_STRENGTH = config['LORA1_CLIP_STRENGTH']

LORA2_WEIGHT = config['LORA2_WEIGHT']
LORA3_WEIGHT = config['LORA3_WEIGHT']

LORA2_CLIP_STRENGTH = config['LORA2_CLIP_STRENGTH']
LORA3_CLIP_STRENGTH = config['LORA3_CLIP_STRENGTH']

GUIDANCE_SCALE = config['GUIDANCE_SCALE']

UNET_FILENAME = config['UNET_FILENAME']
CLIP1_FILENAME = config['CLIP1_FILENAME']
CLIP2_FILENAME = config['CLIP2_FILENAME']
VAE_FILENAME = config['VAE_FILENAME']

DEFAULT_SAMPLER = config['DEFAULT_SAMPLER']
DEFAULT_SCHEDULER = config['DEFAULT_SCHEDULER']

REPEAT_LATENT_BATCH_AMOUNT = config['REPEAT_LATENT_BATCH_AMOUNT']
NUMBER_OF_LOOPS = config['NUMBER_OF_LOOPS']

BEST_SAMPLERS_SCHEDULERS = config['BEST_SAMPLERS_SCHEDULERS']

USE_ALL_CONFIGS = config['USE_ALL_CONFIGS']

# Load LoRA combinations (ensure metadata is current but do not create new combos)
update_lora_metadata()  # Ensure metadata is up-to-date
with open(config['LORA_COMBOS_PATH'], 'r', encoding='utf-8') as f:
    lora_combos = json.load(f)

# Shuffle the lora_combos list
random.shuffle(lora_combos)

# HELPER FUNCTIONS SECTION

def prepend_trigger_words_to_prompt(lora_choices, prompt_text, lora_metadata):
    """Prepend trigger words using loaded metadata."""
    trigger_words = []
    for lora in lora_choices:
        metadata = lora_metadata.get(lora)
        if metadata:
            trigger_word = metadata.get("trigger_word", "")
            description = metadata.get("description", "No description available.")
            log(f"Match found for LoRA '{lora}': Trigger Word = '{trigger_word}', Description = '{description}'")
            trigger_words.append(trigger_word)
        else:
            log(f"No match found for LoRA '{lora}'. Proceeding without addition.")
    trigger_words = ' '.join(filter(None, trigger_words))  # Remove empty strings
    full_prompt = f"{trigger_words} {prompt_text}" if trigger_words else prompt_text
    return full_prompt

def find_lora_set(lora1, lora2, lora3):
    """ Find the corresponding iteration set in lora_combos.json for the given LoRAs. """
    for iteration in lora_combos:
        if (iteration["LORA1"]["name"] == lora1 and
            iteration["LORA2"]["name"] == lora2 and
            iteration["LORA3"]["name"] == lora3):
            print(f"Match found: iteration {iteration['iteration']}")
            print(f"Using PROMPT_TEXT: {iteration['PROMPT_TEXT']}")
            return iteration["PROMPT_TEXT"]

    # Use the default PROMPT_TEXT if no match is found
    print("No match found.")
    return config['PROMPT_TEXT']

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



def execute_workflow_loop(loop, total_start_time, workflow_json, sampler_name, scheduler_name, lora_combos, available_loras):
    """ Execute one iteration of the workflow loop. """
    lora_combo = random.choice(lora_combos)  # Select a random lora_combo on each iteration
    LORA2, LORA3 = lora_combo['LORA2']['name'], lora_combo['LORA3']['name']

    try:
        # Use the function to get the appropriate PROMPT_TEXT
        final_prompt_text = find_lora_set(LORA1, LORA2, LORA3)
        
    except Exception as e:
        log_error(f"An error occurred: {str(e)}")
        final_prompt_text = config['PROMPT_TEXT']

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

        # Load LoRA metadata; assume a function or dictionary providing the needed data structure
        lora_metadata = {f: '' for f in available_loras}  # Simplified placeholder to mock metadata

        # Generate prompt with triggers and use the new final_prompt_text
        final_prompt = prepend_trigger_words_to_prompt([LORA1, LORA2, LORA3], final_prompt_text, lora_metadata)
        workflow_json["6"]["inputs"]["text"] = final_prompt

        # Log the prompt after it has been updated
        log(f"Prompt we're creating: {final_prompt}")

        workflow_json["10"]["inputs"]["vae_name"] = VAE_FILENAME
        workflow_json["11"]["inputs"]["clip_name1"] = CLIP1_FILENAME
        workflow_json["11"]["inputs"]["clip_name2"] = CLIP2_FILENAME
        workflow_json["12"]["inputs"]["unet_name"] = UNET_FILENAME
        workflow_json["16"]["inputs"]["sampler_name"] = sampler_name
        workflow_json["17"]["inputs"]["scheduler"] = scheduler_name
        workflow_json["17"]["inputs"]["steps"] = INFERENCE_STEPS
        workflow_json["26"]["inputs"]["guidance"] = GUIDANCE_SCALE
        workflow_json["41"]["inputs"]["amount"] = REPEAT_LATENT_BATCH_AMOUNT
        workflow_json["38"]["inputs"]["lora_name"] = LORA1
        workflow_json["38"]["inputs"]["strength_model"] = LORA1_WEIGHT
        workflow_json["38"]["inputs"]["strength_clip"] = LORA1_CLIP_STRENGTH
        workflow_json["42"]["inputs"]["lora_name"] = LORA2
        workflow_json["42"]["inputs"]["strength_model"] = LORA2_WEIGHT
        workflow_json["42"]["inputs"]["strength_clip"] = LORA2_CLIP_STRENGTH
        workflow_json["43"]["inputs"]["lora_name"] = LORA3
        workflow_json["43"]["inputs"]["strength_model"] = LORA3_WEIGHT
        workflow_json["43"]["inputs"]["strength_clip"] = LORA3_CLIP_STRENGTH

        filename_prefix = create_filename_prefix(final_prompt_text, sampler_name, scheduler_name)
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
        log(f"Final prompt queued: {final_prompt}")

        # Wait for images to be created
        new_files = wait_for_images(
            OUTPUT_FOLDER, MAX_WAIT_TIME, CHECK_INTERVAL,
            filename_prefix, REPEAT_LATENT_BATCH_AMOUNT, log, loop, 
            total_start_time, loop_start_time, final_prompt, LORA1, LORA2, LORA3
        )
        end_time = datetime.now()
        time_taken = end_time - start_time

        moved_files = move_and_rename_images(
            OUTPUT_FOLDER, API_OUTPUT_FOLDER, filename_prefix, 
            REPEAT_LATENT_BATCH_AMOUNT, DELAY_BEFORE_MOVE
        )

        for file in moved_files:
            remove_metadata_if_required(
                file, remove_metadata_in_place, show_metadata, 
                has_metadata, log, REMOVE_METADATA_AFTER
            )

        log(f"Time taken for creation: {time_taken} for {len(moved_files)} files")

        # Capture the time to respond in the combos file after the workflow executes
        for iteration in lora_combos:
            if iteration["LORA1"]["name"] == LORA1 and iteration["LORA2"]["name"] == LORA2 and iteration["LORA3"]["name"] == LORA3:
                iteration["time_to_respond"] = time_taken.total_seconds()
                iteration["PROMPT_TEXT"] = final_prompt
                break

        with open(config['LORA_COMBOS_PATH'], 'w', encoding='utf-8') as f:
            json.dump(lora_combos, f, indent=2)

        log_iteration_details(
            loop, start_time, end_time, INFERENCE_STEPS, 
            REPEAT_LATENT_BATCH_AMOUNT, scheduler_name, sampler_name, 
            moved_files, LORA2, LORA3, final_prompt
        )

        return True

    except Exception as e:
        log_error(f"Exception occurred during loop {loop}: {str(e)}")
        return False

# MAIN EXECUTION SECTION

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
        with open(WORKFLOW_PATH, 'r', encoding='utf-8') as file:
            workflow_json = json.load(file)
        log("Workflow loaded.")

        total_start_time = datetime.now()
        total_files = 0

        # Get all available LORA files except the primary LORA
        available_loras = [f for f in os.listdir(LORA_DIRECTORY) if f != LORA1 and f.endswith('.safetensors')]

        # Shuffle the LoRA combinations
        random.shuffle(lora_combos)

        total_lora_combinations = len(lora_combos)
        total_sampler_scheduler_combinations = len(BEST_SAMPLERS_SCHEDULERS)
        total_expected_images = total_lora_combinations * total_sampler_scheduler_combinations * REPEAT_LATENT_BATCH_AMOUNT * NUMBER_OF_LOOPS

        for loop_count in range(NUMBER_OF_LOOPS):
            for combo_index, lora_combo in enumerate(lora_combos):  # Iterate systematically over each randomized combo
                LORA2, LORA3 = lora_combo['LORA2']['name'], lora_combo['LORA3']['name']
                final_prompt_text = find_lora_set(LORA1, LORA2, LORA3)

                for idx, (sampler, scheduler) in enumerate(BEST_SAMPLERS_SCHEDULERS):
                    log(f"Loop {loop_count + 1}/{NUMBER_OF_LOOPS}, Combo {combo_index + 1}/{total_lora_combinations}, Sampler Name: {sampler}, Scheduler Name: {scheduler}")

                    # Calculate running and estimated times
                    current_time = datetime.now()
                    running_time = current_time - total_start_time

                    images_done = (combo_index * total_sampler_scheduler_combinations + idx + 1) * REPEAT_LATENT_BATCH_AMOUNT
                    total_remaining_images = total_expected_images - images_done
                    estimated_time_remaining = (running_time / total_files * total_remaining_images) if total_files > 0 else timedelta(0)

                    log(f"Running time since start of script: {running_time}")
                    log(f"Total images remaining: {total_remaining_images}")
                    log(f"Estimated time remaining: ~{timedelta(seconds=int(estimated_time_remaining.total_seconds()))}")
                    log("================")

                    start_time = time.time()
                    success = execute_workflow_loop(
                        combo_index * total_sampler_scheduler_combinations + idx + 1,
                        total_start_time,
                        workflow_json,
                        sampler,
                        scheduler,
                        lora_combos,
                        available_loras
                    )
                    time_taken_this_set = time.time() - start_time

                    if success:
                        total_files += REPEAT_LATENT_BATCH_AMOUNT
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
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("")

    main()