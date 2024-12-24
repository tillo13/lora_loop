# Creative AI Prompt and Image Generation Suite

## Overview

Welcome to the Creative AI Prompt and Image Generation Suite! This application is designed to harness the power of artificial intelligence to generate creative text prompts and transform them into stunning images. By leveraging modern AI models and innovative Low-Rank Adaptation (LoRA) techniques, this suite empowers users to explore new creative possibilities.

## How It Works

The application operates in two main stages:

### Stage 1: Generate Creative Text Prompts

In the initial phase, the script `1_create_ollama_prompts.py` works with the Llama3 model via the Ollama platform to create enriched text prompts. These prompts are generated using unique combinations of LoRA configurations, which enhance the AI’s ability to produce inventive responses. The results, alongside valuable execution logs, are stored in a JSON file for further use.

### Stage 2: Transform Text Prompts Into Images

Once the prompts are ready, the `2_create_loop_lora.py.py` script takes over. This script uses the ComfyUI environment to turn the generated text prompts into creative images. It applies different LoRA configurations randomly, allowing for diverse and unique image outputs each time the script is run.

## Key Features

- **AI-Driven Text Generation**: Automatically generate creative prompts using advanced AI capabilities.
- **Image Creation**: Convert text prompts into imaginative images with various LoRA configurations.
- **Repeatable Workflows**: Run the image generation process multiple times to explore endless creative combinations.
- **Detailed Logging**: Comprehensive logs capture each step of the process for review and analysis.

## Getting Started

### Prerequisites

- Ensure Python 3.7+ is installed.
- Required libraries must be installed via `pip install -r requirements.txt`.
- Configuration files (`lora_metadata.json` and `global_variables.json`) need to be properly set up.
- **[Install ComfyUI](./README_3_install_comfyui.md)**: Follow the installation guide to set up ComfyUI properly.

### Running the Application

1. **Generate Text Prompts**:
   - Execute the prompts generation script in your command line or terminal:
     ```bash
     python 1_create_ollama_prompts.py
     ```

2. **Create Images**:
   - Once prompts are ready, use the following command to generate images:
     ```bash
     python 2_create_loop_lora.py.py
     ```

## Additional Resources

- For detailed prompt generation and image creation process, refer to `[README_1_create_ollama_prompts.md](./README_1_create_ollama_prompts.md)` and `[README_2_create_images.md](./README_2_create_images.md)`.
- For ComfyUI setup and usage, consult `[README_3_install_comfyui.md](./README_3_install_comfyui.md)`.

## Conclusion

This suite of tools is an exciting way to blend AI with creativity, enabling users to generate and visualize creative ideas effortlessly. Enjoy exploring the bounds of creativity with AI!
