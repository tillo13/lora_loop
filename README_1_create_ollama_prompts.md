## Overview

This collection of scripts is designed to automate the process of generating creative text prompts and capturing responses from an AI model (Llama3 via Ollama) using various LoRA (Low-Rank Adaptation) configurations. The primary script, `1_create_ollama_prompts.py`, integrates multiple utility modules to manage AI model interactions, data processing, and detailed logging. In the end, the main goal is to produce a JSON file (`lora_combos.json`) that contains enriched prompts, model-generated responses, and detailed execution logs with the plan that 2_create_

## Expected Output

By executing these scripts, the following key outputs will be generated:

- **`lora_combos.json`**: This file will contain:
  - Combinations of LoRA configurations.
  - Corresponding enriched prompts for each configuration.
  - Captured responses from the Ollama AI model.
  - Timing and performance data for each prompt generation.
  
- **Logs**:
  - **`superhero_test_log.txt`**: Captures sequential logs of events, messages, and errors.
  - **`iteration_log.csv`**: Details each iteration's timing, configurations, and resulting prompts for analytical purposes.

## Prerequisites

Before running the script suite, ensure you have:

- **Python Environment**: Installation of Python 3.7+ on your system.
- **Required Python Libraries**: Install dependencies using `pip install -r requirements.txt` (includes `psutil`, `requests`, etc.).
- **Config Files**: Have `lora_metadata.json` in `comfyui/models/loras` for LoRA attributes, and `global_variables.json` for prompt templates.

## Script Execution

### Running the Script

To execute, open your terminal or command prompt and run:

```bash
python 1_create_ollama_prompts.py
```

### Detailed Script Descriptions

**1. `1_create_ollama_prompts.py`:**

- **Objective**: Serve as the hub for initiating and managing the process of generating prompts and obtaining AI-model responses. Collects results and logs every step in a structured manner.
- **Workflow**:
  - **Initialization**: Starts by halting any running Ollama services and clears memory to optimize resource availability for subsequent processing.
  - **Setup and Execution**: Ensures Ollama is installed and operational. It retrieves prompt configurations (LoRAs) and creates a tailored `lora_combos.json` cataloging these combinations and accompanying prompts.
  - **Interaction and Logging**: Engages the AI model, captures and processes responses for each prompt, and enhances prompts where applicable.
  - **Save and Archive**: Periodically updates the `lora_combos.json` and logs details in text and CSV formats to facilitate result reviews and future reference.

**2. `ollama_utils.py`:**

- **Purpose**: Manage interactions and setup for the Ollama AI model.
- **Key Functions**:
  - **Installation and Service Management**: Handles downloading, installing, and running the Ollama service, with robust error handling and retry logic.
  - **Model Communication**: Sends prompts to the model, retrieves responses, and includes utility functions to verify system resource availability.
  - **Resource Management**: Includes functions for clearing GPU memory and checking for active service ports to prevent conflicts.

**3. `lora_utils.py`:**

- **Purpose**: Facilitate the preparation and management of LoRA configurations.
- **Key Functions**:
  - **Metadata Handling**: Loads and updates LoRA metadata to ensure that configurations reflect the current setup.
  - **Prompt Preparation**: Constructs prompt combinations based on metadata and unique LoRA attributes, enhancing prompts with relevant trigger words.
  - **Maintenance Functions**: Archive previous combinations and ensure that metadata reflects the latest available LoRAs.

**4. `logging_utils.py`:**

- **Purpose**: Provide detailed logging capabilities for the execution process.
- **Key Functions**:
  - **Log Messaging**: Captures general messages, warnings, and errors, ensuring all components provide traceable outputs.
  - **Error Logging**: Delivers detailed stack traces and error descriptions to assist with debugging efforts.
  - **Iteration Tracking**: Logs each generation cycle including performance timings, used configurations, and resultant file paths, aiding in performance analysis.

## Error Handling

- **Installation and Service Errors**: Built-in notifications and retries safeguard against transient issues and help ensure consistent availability of service.
- **Network and Response Handling**: Ensures robustness through retry strategies and error notifications that guide immediate corrective action.
- **Logging**: Continuous and detailed logging aids in real-time observation of script progression and retrospective failure analysis.

## Conclusion

This suite of scripts harmonizes text prompt generation with AI-model responsiveness, managed through detailed logging and robust process management. Designed to support easy execution and in-depth review, it serves as a foundational toolkit for creative text generation projects leveraging AI capabilities. For customization, delving into individual script files and configuration settings will broaden your adaptability of the provided tools.