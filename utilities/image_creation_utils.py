import os
import re
import shutil
import time
import json
from datetime import datetime, timedelta

def load_lora_combos():
    """Load LoRA combinations from a JSON file."""
    # Calculate the base filepath relative to the script's directory 
    base_path = os.path.dirname(os.path.dirname(__file__))
    filepath = os.path.join(base_path, 'lora_combos.json')
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def wait_for_images(output_path, wait_time, check_interval, prefix, expected_count, log, loop, start_time_total, start_time_loop, prompt, lora1, lora2, lora3):
    """Wait for multiple image files to appear in the output directory."""
    # Calls the modified lora loader
    lora_combos = load_lora_combos()
    total_combinations = len(lora_combos)

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

        # Log estimates for the current loop and entire process
        log_estimates(log, loop, len(found_files), expected_count, total_elapsed_time, loop_elapsed_time, total_combinations, prompt, lora1, lora2, lora3)

        time.sleep(check_interval)
        total_wait_time += check_interval

    if len(found_files) >= expected_count:
        log(f"All {expected_count} images created successfully.")
    else:
        log(f"Timeout: Only {len(found_files)} out of {expected_count} images were created within {wait_time} seconds.")

    return found_files


def load_configurations():
    """Load configurations from global_variables.json."""
    base_path = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(base_path, 'global_variables.json')
    with open(config_path, 'r', encoding='utf-8') as file:
        configs = json.load(file)
        return {key: value['value'] for key, value in configs.items()}

def log_estimates(log, loop, images_created, expected_count, total_elapsed_time, loop_elapsed_time, total_combinations, prompt, lora1, lora2, lora3):
    """Log estimates for the current loop and entire process."""
    config = load_configurations()
    total_sampler_scheduler_combinations = len(config['BEST_SAMPLERS_SCHEDULERS'])

    completed_images = images_created + (expected_count * loop)
    number_of_loops = config['NUMBER_OF_LOOPS']
    repeat_latent_batch_amount = config['REPEAT_LATENT_BATCH_AMOUNT']

    total_combinations_per_loop = total_combinations * repeat_latent_batch_amount * total_sampler_scheduler_combinations
    total_expected_images = number_of_loops * total_combinations_per_loop

    total_remaining_images = total_expected_images - completed_images
    estimated_time_remaining = total_elapsed_time.total_seconds() / completed_images * total_remaining_images if completed_images > 0 else 0
    estimated_remaining_str = str(timedelta(seconds=int(estimated_time_remaining)))

    # Log for entire process
    log("\n===Estimates for entire process===")
    log(f"Total images (Expected): {total_expected_images}")
    log(f"Total iterations in lora_combos.json: {total_combinations}")
    log(f"Completed images: {completed_images}")
    log(f"Total remaining images: {total_remaining_images}")
    log(f"Total time elapsed: {str(total_elapsed_time)}")
    log(f"Estimated time remaining: ~{estimated_remaining_str}\n")

    # Log for current loop
    log("===Estimates for this loop===")
    log(f"Images: {images_created}/{expected_count}")
    log(f"Time elapsed: {str(loop_elapsed_time)}")
    log(f"Prompt: {prompt}")  # Log the prompt
    log(f"LORA1: {lora1}")
    log(f"LORA2: {lora2}")
    log(f"LORA3: {lora3}")

def move_and_rename_images(src_path, dest_dir, new_filename_prefix, num_images, delay=5):
    """Move and rename image files to directory with indexed filenames."""
    os.makedirs(dest_dir, exist_ok=True)
    moved_files = []

    for i in range(1, num_images + 1):
        original_filename = f"{new_filename_prefix}_{i:05d}_.png"
        dest_filename = f"{new_filename_prefix}_{i:05d}_.png"
        src_filepath = os.path.join(src_path, original_filename)
        dest_filepath = os.path.join(dest_dir, dest_filename)
        
        time.sleep(delay)  # Ensures file is not in use

        try:
            shutil.move(src_filepath, dest_filepath)
            moved_files.append(dest_filepath)
            print(f"Image moved and renamed to {dest_filepath}")
        except FileNotFoundError:
            print(f"Failed to move file: {src_filepath} not found.")

    return moved_files

def create_filename_prefix(prompt_text, sampler_name, scheduler):
    """Create a descriptive filename prefix based on the given criteria."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use just a placeholder like 'desc' to replace non-specific text segment
    return f"{timestamp}_desc_{sampler_name}_{scheduler}"

def remove_metadata_if_required(file_path, remove_func, show_func, has_func, log, remove_metadata_after):
    """Remove metadata from the image if REMOVE_METADATA_AFTER is True."""
    if remove_metadata_after:
        if has_func(file_path):
            log(f"Removing metadata from {file_path}...")
            show_func(file_path)
            remove_func(file_path)
            show_func(file_path)