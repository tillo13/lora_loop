# ComfyUI LoRA Training Guide

This guide walks you through the setup and configuration to train LoRA models using ComfyUI. It includes detailed steps specific to a Windows environment, with the interface accessed via a MacBook on the local network.

## Prerequisites

- **Hardware**: I used NVIDIA GPU (e.g., RTX 3060 with 12GB VRAM), Machine: Dell Inspiron 5675, CPU: AMD Ryzen 7 1800X Eight-Core Processor
- **Software**:
  - Git
  - Python 3.11
  - ComfyUI
  - ComfyUI-Manager
  - Models: `flux1-dev-fp8.safetensors`, `t5xxl_fp8_e4m3fn.safetensors`, `clip_l.safetensors`, `ae.safetensors`

## Setup Instructions

### 1. Create a Virtual Environment and Install Dependencies

```bash
# Open PowerShell and navigate to your desired directory
cd path_to_your_directory

# Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI

# Create and activate virtual environment
python -m venv comfy_env
.\comfy_env\Scripts\Activate

# Install required dependencies
pip install -r requirements.txt
pip install protobuf
```

````

### 2. Install ComfyUI-Manager

```bash
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
cd ..
```

### 3. Run ComfyUI

```bash
# Start ComfyUI with network access
python .\main.py --lowvram --listen 0.0.0.0
```

### 4. Open Firewall for External Access (Optional)

If you want to access ComfyUI from another device, you might need to open the firewall on your Windows machine:

1. Open Windows Firewall.
2. Go to Advanced settings.
3. Create a new Inbound Rule to allow TCP traffic on port 8188.

Alternatively, if you are only using it locally, you can access it via 127.0.0.1.

### 5. Access ComfyUI

Open a web browser and navigate to http://<Your_Windows_Machine_IP>:8188 or http://127.0.0.1:8188.

### 6. Download Required Models

Download the following models and place them in their respective directories within ComfyUI:

- `flux1-dev-fp8.safetensors`: https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors

  - Place in `path_to_your_directory/ComfyUI/models/unet/`

- `t5xxl_fp8_e4m3fn.safetensors`: https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors

  - Place in `path_to_your_directory/ComfyUI/models/clip/`

- `clip_l.safetensors`: https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors

  - Place in `path_to_your_directory/ComfyUI/models/clip/`

- `ae.safetensors`: https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors
  - Place in `path_to_your_directory/ComfyUI/models/vae/`

### 7. Install Missing Custom Nodes

In the ComfyUI interface:

1. Click on "Manager" in the sidebar.
2. Go to "Custom Nodes Manager."
3. Search for and install:
   - ComfyUI Flux Trainer by kijai
   - rgthree's ComfyUI Nodes by rgthree
   - Any other missing nodes indicated in the setup process

### 8. Load and Run the Pre-configured Workflow

Download the Pre-configured Workflow:

- Download `workflow_adafactor_splitmode_dimalpha64_3000steps_low10GBVRAM.json` from https://pastebin.com/CjDyMBHh
- Place the file in a directory you can easily access.

Load the Workflow in ComfyUI:

1. Click on "Load" in the sidebar.
2. Navigate to the directory where you placed `workflow_adafactor_splitmode_dimalpha64_3000steps_low10GBVRAM.json` and load it.
   Set Up Your Dataset:

- Place your training images in a directory like `path_to_your_directory/ComfyUI_training/training/input/`.

### 9. Start Training

Once the pre-configured workflow is loaded and your dataset is in place, click "Queue Prompt" to begin the training process.

### Monitoring and Checking Output

- Monitor the training process through the ComfyUI interface.
- Once training is complete, check the output in the specified output directory (e.g., `path_to_your_directory/ComfyUI_training/training/output/`).

### Summary

- Verify Model Paths: Check that the workflow correctly references the model paths.
- Set Up Dataset: Ensure your training images are correctly placed in the input directory.
- Queue Training: Start the training by queuing the prompt.

By following these instructions, you should be able to efficiently set up and run your LoRA training with ComfyUI. If you encounter any issues or need further clarification, feel free to ask!

````
