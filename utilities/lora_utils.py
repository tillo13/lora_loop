import os
import json
import itertools
import re
from .logging_utils import log

def load_configurations():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'global_variables.json')
    with open(config_path, 'r', encoding='utf-8') as file:
        configs = json.load(file)
        return {key: value['value'] for key, value in configs.items()}

config = load_configurations()
LORA_DIRECTORY = config['LORA_DIRECTORY']
LORA_METADATA_FILENAME = config['LORA_METADATA_FILENAME']
LORA1_NAME = config['LORA1_NAME']
OLLAMA_BASE_PROMPT = config['OLLAMA_BASE_PROMPT']
PROMPT_TEXT = config['PROMPT_TEXT']
LORA_COMBOS_PATH = config['LORA_COMBOS_PATH']

def cleanse_prompt(text):
    """
    Cleanse the input text by removing non-alphanumeric characters,
    except for spaces and basic punctuation marks.
    """
    # Allow alphanumeric, spaces, and basic punctuation (.,!?')
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s.,!?']", '', text)
    return cleaned_text.strip()

def sort_json_by_iteration(file_path):
    # Read the JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Sort the data based on the 'iteration' field of each dictionary
    sorted_data = sorted(data, key=lambda x: x["iteration"])
    
    # Write the sorted data back to the file
    with open(file_path, 'w') as file:
        json.dump(sorted_data, file, indent=4)

def create_lora_combos_json():
    lora_metadata_path = os.path.join(LORA_DIRECTORY, LORA_METADATA_FILENAME)
    
    with open(lora_metadata_path, 'r') as file:
        lora_metadata = json.load(file)
    
    available_loras = [lora for lora in os.listdir(LORA_DIRECTORY)
                       if lora.endswith('.safetensors') and lora in lora_metadata and lora != LORA1_NAME]
    
    unique_combos = set()
    combos_data = []
    
    for lora2, lora3 in itertools.combinations(available_loras, 2):
        if {(lora2, lora3), (lora3, lora2)}.intersection(unique_combos):
            continue
        
        unique_combos.add((lora2, lora3))
        
        lora1_metadata = lora_metadata.get(LORA1_NAME, {})
        lora2_metadata = lora_metadata.get(lora2, {})
        lora3_metadata = lora_metadata.get(lora3, {})
        
        lora1_desc = lora1_metadata.get('description', '')
        lora2_desc = lora2_metadata.get('description', '')
        lora3_desc = lora3_metadata.get('description', '')
        
        
        suggested_prompt_text = (
            f"{config['OLLAMA_BASE_PROMPT']} Start by synthesizing a unique prompt based on: {config['PROMPT_TEXT']} "
            f"and consider: {config.get('PROMPT_TEXT2', '')}. "
            f"Incorporate the three enhancement descriptions creatively into your overall SUGGESTED PROMPT as follows: "
            f"Enhancement 1: {lora1_desc}, "
            f"Enhancement 2: {lora2_desc}, "
            f"Enhancement 3: {lora3_desc}. "
            f"Ensure the prompt is a creative fusion of these elements without directly copying sentences from any singular one of them."
        )


        prompt_text = f"{lora1_desc} meets {lora2_desc} and {lora3_desc}"
        prompt_text = cleanse_prompt(prompt_text)
        
        combo_info = {
            "iteration": len(combos_data) + 1,
            "LORA1": {
                "name": LORA1_NAME,
                "metadata": lora1_metadata
            },
            "LORA2": {
                "name": lora2,
                "metadata": lora2_metadata
            },
            "LORA3": {
                "name": lora3,
                "metadata": lora3_metadata
            },
            "SUGGESTED_PROMPT_TEXT": cleanse_prompt(suggested_prompt_text),
            "PROMPT_TEXT": prompt_text,
            "time_to_respond": None
        }
        
        combos_data.append(combo_info)
    
    with open(LORA_COMBOS_PATH, 'w') as outfile:
        json.dump(combos_data, outfile, indent=4)
    
    # Sort the file by iteration after creation
    sort_json_by_iteration(LORA_COMBOS_PATH)
    log(f"{LORA_COMBOS_PATH} has been successfully created with unique combinations.")

def update_lora_metadata(lora_directory=LORA_DIRECTORY, metadata_filename=LORA_METADATA_FILENAME):
    lora_metadata_path = os.path.join(lora_directory, metadata_filename)

    # Check if the metadata file exists and stop operation if it does not
    if not os.path.exists(lora_metadata_path):
        log(f"[ERROR] Metadata file {lora_metadata_path} does not exist. Please create the file before proceeding.")
        return 0, 0  # No updates made since the file does not exist

    # Load existing metadata
    existing_metadata = {}
    try:
        with open(lora_metadata_path, 'r') as file:
            existing_metadata = json.load(file)
    except json.JSONDecodeError as e:
        log(f"Error decoding JSON from {lora_metadata_path}: {e}")
        return 0, 0  # No update made due to error in reading

    original_count = len(existing_metadata)

    # List all .safetensors files in the specified directory
    current_lora_files = {
        f for f in os.listdir(lora_directory)
        if f.endswith('.safetensors') and os.path.isfile(os.path.join(lora_directory, f))
    }

    # Check if all files in the current directory are reflected in the metadata
    added_loras = []
    for lora_file in current_lora_files:
        if lora_file not in existing_metadata:
            # Add this new LoRA file to metadata with default placeholder values
            existing_metadata[lora_file] = {
                "trigger_word": "",
                "description": "",
                "url": ""
            }
            added_loras.append(lora_file)

    # Write the updated metadata back to the file if new entries were added
    if added_loras:
        with open(lora_metadata_path, 'w') as file:
            json.dump(existing_metadata, file, indent=4)

    final_count = len(existing_metadata)

    # Log changes
    log(f"Initial LoRA count: {original_count}, Final LoRA count: {final_count}")
    if added_loras:
        log(f"Added LoRAs: {added_loras}")

    return original_count, final_count