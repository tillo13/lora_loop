import os
import json
import requests
import re
from datetime import datetime, timedelta
from time import time
import glob
import shutil
from collections import defaultdict
from utilities.ollama_utils import (
    install_and_setup_ollama,
    get_story_response_from_model,
    stop_ollama_service,
    kill_existing_ollama_service,
    clear_gpu_memory,
    start_ollama_service
)
from utilities.lora_utils import create_lora_combos_json, update_lora_metadata, cleanse_prompt

def load_configurations():
    with open('global_variables.json', 'r', encoding='utf-8') as file:
        configs = json.load(file)
        return {key: value['value'] for key, value in configs.items()}

def load_archived_lora_combos(archive_path):
    archived_combos = {}
    archived_match_counts = defaultdict(int)
    if os.path.exists(archive_path):
        for archive_file in glob.glob(os.path.join(archive_path, 'lora_combos_*.json')):
            with open(archive_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for combo in data:
                    key = frozenset((combo["LORA1"]["name"], combo["LORA2"]["name"], combo["LORA3"]["name"]))
                    archived_combos[key] = {
                        "combo": combo,
                        "file": archive_file
                    }
    return archived_combos, archived_match_counts

def check_and_move_lora_file(lora_name):
    base_dir = LORA_DIRECTORY  # Ensure this points to the correct base directory
    expected_path = os.path.join(base_dir, lora_name)
    people_dir = os.path.join(base_dir, 'people')
    people_path = os.path.join(people_dir, lora_name)

    print(f"[DEBUG] Checking for '{lora_name}' in '{expected_path}' and '{people_dir}'")

    if os.path.exists(expected_path):
        print(f"[INFO] LoRA file '{lora_name}' is already in the correct directory.")
        return True
    elif os.path.exists(people_path):
        print(f"[INFO] LoRA file '{lora_name}' found in /people/. Moving it to the correct directory.")
        shutil.move(people_path, expected_path)
        return True
    else:
        print(f"[ERROR] LoRA file '{lora_name}' could not be found. Please ensure it is placed in the /people/ directory and restart the script.")
        return False

def check_and_move_lora_metadata():
    base_dir = LORA_DIRECTORY  # Ensure this points to the correct base directory
    metadata_filename = 'lora_metadata.json'
    expected_path = os.path.join(base_dir, metadata_filename)
    data_dir = os.path.join(base_dir, 'data')
    backup_path = os.path.join(data_dir, metadata_filename)

    print(f"[DEBUG] Checking for '{metadata_filename}' in '{expected_path}' and '{data_dir}'")

    if os.path.exists(expected_path):
        print(f"[INFO] LoRA metadata file '{metadata_filename}' is already in the correct directory.")
        return True
    elif os.path.exists(backup_path):
        print(f"[INFO] LoRA metadata file '{metadata_filename}' found in /data/. Moving it to the correct directory.")
        shutil.copy(backup_path, expected_path)
        return True
    else:
        print(f"[ERROR] LoRA metadata file '{metadata_filename}' could not be found. Please ensure it is placed in the /data/ directory and restart the script.")
        return False

config = load_configurations()

MODEL_NAME = config["MODEL_NAME"]
LORA_COMBOS_PATH = config["LORA_COMBOS_PATH"]
ARCHIVE_PATH = config["ARCHIVE_PATH"]
LORA_DIRECTORY = config["LORA_DIRECTORY"]

def load_lora_combos():
    with open(LORA_COMBOS_PATH, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_lora_combos(lora_combos):
    with open(LORA_COMBOS_PATH, 'w', encoding='utf-8') as file:
        json.dump(lora_combos, file, indent=2)

def archive_lora_combos():
    if os.path.exists(LORA_COMBOS_PATH):
        if not os.path.exists(ARCHIVE_PATH):
            os.makedirs(ARCHIVE_PATH)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_filename = os.path.join(ARCHIVE_PATH, f"lora_combos_{timestamp}.json")
        os.rename(LORA_COMBOS_PATH, archive_filename)
        print(f"[INFO] Archived existing lora_combos.json to {archive_filename}")

def clean_response(response):
    filler_phrases = [
        "here is a rewritten prompt", "this is a rewritten prompt",
        "incorporates the descriptions", "this prompt combines",
        "rewritten as", "the following prompt", "generate a"
    ]
    for phrase in filler_phrases:
        response = response.replace(phrase, "").strip()
    patterns = [
        r'^Here',
        r'\*\*Image:\*\*',
        r'^\s+',
        r'\s+$',
        r'\\n+',  # Remove multiple newline characters
        r'\\',    # Remove single backslashes
        r'\"$',   # Remove trailing quotes
        r'\n',    # Remove standalone newline characters
        r'\b(lora\d*)\b',
        r'\bhere is\b',
        r'\bsuggested prompt\b',
        r':\"',
        r'\bhere\'?s the\b',
        r'\bhere is the\b',
        r'\b\'s\b'  # Remove standalone 's
    ]
    for pattern in patterns:
        response = re.sub(pattern, "", response, flags=re.IGNORECASE)
    
    response = re.sub(r'\s{2,}', ' ', response)

    return response.strip()

def main():
    start_time = datetime.now()

    # Check and move LoRA metadata before proceeding
    if not check_and_move_lora_metadata():
        return

    lora1_name = config["LORA1_NAME"]
    if not check_and_move_lora_file(lora1_name):
        return

    archive_lora_combos()

    archived_combos, archived_match_counts = load_archived_lora_combos(ARCHIVE_PATH)

    original_count, final_count = update_lora_metadata(lora_directory=LORA_DIRECTORY)
    print(f"[SUMMARY] LoRA metadata updated: started with {original_count}, ended with {final_count} entries.")

    create_lora_combos_json()

    kill_existing_ollama_service()
    clear_gpu_memory()

    install_and_setup_ollama(MODEL_NAME)

    lora_combos = load_lora_combos()
    global_vars = load_configurations()
    base_prompt = global_vars["OLLAMA_BASE_PROMPT"]

    total_time_spent = 0
    new_combos_to_process = []

    try:
        for iteration_data in lora_combos:
            key = frozenset((iteration_data["LORA1"]["name"], iteration_data["LORA2"]["name"], iteration_data["LORA3"]["name"]))

            if key in archived_combos:
                archived_data = archived_combos[key]["combo"]
                iteration_data.update({
                    "PROMPT_TEXT": archived_data["PROMPT_TEXT"],
                    "time_to_respond": archived_data["time_to_respond"],
                })

                archived_match_counts[archived_combos[key]['file']] += 1

                lora1_name = iteration_data["LORA1"]["name"]
                lora2_name = iteration_data["LORA2"]["name"]
                lora3_name = iteration_data["LORA3"]["name"]
                file_found = archived_combos[key]['file']
                prompt_text = iteration_data['PROMPT_TEXT']

                print(f"[INFO] Found existing combination for iteration {iteration_data['iteration']} using archived data from {file_found}.")
                print(f"LoRA Names: {lora1_name}, {lora2_name}, {lora3_name}")
                print(f"PROMPT_TEXT: {prompt_text}")
            else:
                lora1_name = iteration_data["LORA1"]["name"]
                lora2_name = iteration_data["LORA2"]["name"]
                lora3_name = iteration_data["LORA3"]["name"]

                print(f"[INFO] New combination (lora1: {lora1_name}, lora2: {lora2_name}, lora3: {lora3_name}) for iteration {iteration_data['iteration']} to be processed.")
                new_combos_to_process.append(iteration_data)

        total_iterations = len(new_combos_to_process)

        if new_combos_to_process:
            start_ollama_service()

        for index, iteration_data in enumerate(new_combos_to_process):
            suggested_prompt = iteration_data.get("SUGGESTED_PROMPT_TEXT", "")
            question = f"{base_prompt} {suggested_prompt}"

            start_iteration_time = time()

            try:
                raw_answer = get_story_response_from_model(MODEL_NAME, question)
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Error querying Ollama model: {e}")
                continue
            
            end_iteration_time = time()

            time_to_respond = end_iteration_time - start_iteration_time
            total_time_spent += time_to_respond
            average_time_per_iteration = total_time_spent / (index + 1)
            remaining_iterations = total_iterations - (index + 1)
            estimated_time_remaining = remaining_iterations * average_time_per_iteration

            answer = clean_response(raw_answer)
            answer = cleanse_prompt(answer)

            if answer:
                print(f"[Iteration: {iteration_data['iteration']}]")
                print(f"Prompt: {question}")
                print(f"Response: {answer}")
                print(f"Time to respond: {time_to_respond:.6f} seconds")

                trigger_words = []
                for idx in range(1, 4):
                    lora = iteration_data.get(f"LORA{idx}", {})
                    trigger_word = lora.get("metadata", {}).get("trigger_word", "")
                    if trigger_word:
                        trigger_words.append(trigger_word)

                trigger_words_str = ", ".join(trigger_words)
                updated_prompt_text = f"{trigger_words_str}, {answer}" if trigger_words_str else answer

                iteration_data.update({
                    "PROMPT_TEXT": cleanse_prompt(updated_prompt_text),
                    "time_to_respond": round(time_to_respond, 6),
                })

                save_lora_combos(lora_combos)
                print(f"[INFO] Updated lora_combos.json for iteration {iteration_data['iteration']}")
                print(f"Completed {index + 1} iterations, total time spent: {total_time_spent:.2f} seconds.")
                print(f"Estimated time remaining for {remaining_iterations} iterations: {estimated_time_remaining:.2f} seconds.")
            else:
                print(f"[ERROR] No answer received for iteration: {iteration_data['iteration']}")

    finally:
        stop_ollama_service()
        clear_gpu_memory()

        end_time = datetime.now()
        execution_time = end_time - start_time
        execution_time_str = str(timedelta(seconds=int(execution_time.total_seconds())))

        print("\n====SUMMARY====")
        print("Archived matches:")
        total_archived = 0
        for file, count in archived_match_counts.items():
            print(f"{os.path.basename(file)}: {count}")
            total_archived += count
        print("--")
        print(f"New Ollama created JSON values: {len(new_combos_to_process)}")
        print(f"Time to execute: {execution_time_str}")

if __name__ == "__main__":
    main()