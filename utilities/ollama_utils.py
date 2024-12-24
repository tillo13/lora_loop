import os
import subprocess
import zipfile
import platform
import psutil
import requests
import socket
import time
from pathlib import Path
import json

OLLAMA_PORT = 11434
OLLAMA_DIRECTORY = Path("ollama")
OLLAMA_EXECUTABLE = OLLAMA_DIRECTORY / "ollama.exe" if platform.system() == "Windows" else OLLAMA_DIRECTORY / "ollama"
OLLAMA_PROCESS = None

DEFAULT_MODELS_DIR = Path("D:/ollama_models") if platform.system() == "Windows" else Path.home() / ".ollama" / "models"

def set_ollama_models_dir():
    models_path = os.getenv("OLLAMA_MODELS", str(DEFAULT_MODELS_DIR))
    models_dir = Path(models_path)

    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        os.environ["OLLAMA_MODELS"] = str(models_dir)
        print(f"Models directory set to {models_dir}")
    except Exception as e:
        print(f"Couldn't use models directory {models_dir}: {e}")
        fallback_dir = Path.home() / ".ollama" / "models"
        try:
            fallback_dir.mkdir(parents=True, exist_ok=True)
            os.environ["OLLAMA_MODELS"] = str(fallback_dir)
            print(f"Fallback models directory set to {fallback_dir}")
        except Exception as fallback_error:
            print(f"Couldn't use fallback models directory {fallback_dir}: {fallback_error}")

def is_windows():
    """Check if the current OS is Windows."""
    return platform.system() == "Windows"

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def ollama_installed():
    return OLLAMA_EXECUTABLE.exists()

def install_ollama():
    print("Ollama not found. Installing now...")
    if is_windows():
        url = "https://ollama.com/download/ollama-windows-amd64.zip"
        local_zip_path = "ollama-windows.zip"
    else:
        url = "https://ollama.com/download/Ollama-darwin.zip"
        local_zip_path = "ollama-darwin.zip"

    extract_dir = OLLAMA_DIRECTORY

    response = requests.get(url)
    with open(local_zip_path, "wb") as file:
        file.write(response.content)

    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    os.remove(local_zip_path)
    print("Ollama installed successfully.")
    return extract_dir

def start_ollama_server(extract_dir):
    global OLLAMA_PROCESS
    os.environ["PATH"] += os.pathsep + str(OLLAMA_EXECUTABLE.parent)

    if is_windows():
        OLLAMA_PROCESS = subprocess.Popen(["cmd", "/k", str(OLLAMA_EXECUTABLE), "serve"], creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        OLLAMA_PROCESS = subprocess.Popen([str(OLLAMA_EXECUTABLE), "serve"])
    return OLLAMA_PROCESS

def retry_ollama_service_start(retries=5, delay=10):
    """Retry the Ollama service start process with retries and delay"""
    for attempt in range(retries):
        proc = start_ollama_service()
        if proc:
            print("Successfully started Ollama service.")
            return True
        print(f"Retrying start Ollama service ({attempt + 1}/{retries}) in {delay} seconds...")
        time.sleep(delay)
    print("Failed to start Ollama service after multiple attempts.")
    return False

def start_ollama_service():
    if is_port_in_use(OLLAMA_PORT):
        print(f"Port {OLLAMA_PORT} is already in use. Assuming Ollama service is running.")
        return True

    if not ollama_installed():
        extract_dir = install_ollama()
    else:
        extract_dir = OLLAMA_DIRECTORY  # Use the existing directory if already installed

    proc = start_ollama_server(extract_dir)

    for _ in range(30):  # Wait up to 30 seconds
        if is_port_in_use(OLLAMA_PORT):
            return proc
        time.sleep(1)

    print("Failed to start Ollama service.")
    return None

def stop_ollama_service():
    global OLLAMA_PROCESS
    if OLLAMA_PROCESS is not None:
        OLLAMA_PROCESS.terminate()
        OLLAMA_PROCESS.wait()
        OLLAMA_PROCESS = None
        print("Ollama service has been stopped.")

def pull_model(model_name):
    try:
        subprocess.run([str(OLLAMA_EXECUTABLE), "pull", model_name], check=True)
        print(f"Model {model_name} pulled successfully.")
    except subprocess.CalledProcessError:
        print(f"Failed to pull the model: {model_name}")

def get_story_response_from_model(model_name, user_message):
    response = requests.post(
        f'http://localhost:{OLLAMA_PORT}/api/generate',
        json={'model': model_name, 'prompt': user_message},
        stream=True
    )
    response.raise_for_status()
    story = ""
    for line in response.iter_lines():
        body = json.loads(line)
        if 'response' in body:
            story += body['response']
    return story

def kill_existing_ollama_service():
    for process in psutil.process_iter(['pid', 'name', 'username']):
        try:
            if process.info['name'] == 'ollama.exe' and process.info['username'] == os.getlogin():
                process.terminate()
                process.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"Skipping process {process.info['name']} (PID: {process.info['pid']}): {e}")

    for process in psutil.process_iter(['pid', 'name', 'username']):
        try:
            if 'ollama' in process.info['name'].lower() and process.info['username'] == os.getlogin():
                process.terminate()
                process.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"Skipping process {process.info['name']} (PID: {process.info['pid']}): {e}")

def clear_gpu_memory():
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True, check=True)
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            if pid:
                try:
                    proc = psutil.Process(int(pid))
                    if proc.username() == os.getlogin():
                        proc.terminate()
                        proc.wait(timeout=5)
                except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError) as e:
                    print(f"Skipping PID {pid}: {e}")

        remaining_pids = [pid for pid in pids if psutil.pid_exists(int(pid))]
        for pid in remaining_pids:
            try:
                proc = psutil.Process(int(pid))
                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError) as e:
                print(f"Skipping PID {pid}: {e}")

        print("GPU memory cleared.")
        
        print("Completed Ollama text generation, on to next step!  Please standby!")
    except Exception as e:
        print(f"Failed to clear GPU memory: {e}")

def install_and_setup_ollama(model_name):
    if not ollama_installed():
        install_ollama()

    kill_existing_ollama_service()
    if not retry_ollama_service_start():
        print("Error: Failed to start Ollama service after multiple attempts. Exiting.")
        return

    try:
        pull_model(model_name)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while pulling the model: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        raise

def generate_new_prompt(ollama_model_name, base_prompt, descriptions):
    """
    Generate a new prompt using the given base prompt and LoRA descriptions.
    
    Args:
        ollama_model_name (str): The name of the Ollama model to use for generating the prompt.
        base_prompt (str): The base prompt to use for generating the final prompt.
        descriptions (str): The combined descriptions of LoRAs to use for the prompt.

    Returns:
        str: Generated prompt text from Ollama.
    """
    user_message = f"{base_prompt} {descriptions}"

    try:
        set_ollama_models_dir()
        kill_existing_ollama_service()
        clear_gpu_memory()
        retry_ollama_service_start()
        
        prompt = get_story_response_from_model(ollama_model_name, user_message)
        stop_ollama_service()
        clear_gpu_memory()
        
        return prompt.strip()
    except Exception as e:
        print(f"Error generating prompt: {e}")
        return None

set_ollama_models_dir()