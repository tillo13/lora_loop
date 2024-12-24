import os
import subprocess
import requests
import time
from datetime import datetime
import psutil

# Configurable constants
SERVER_ADDRESS = 'http://127.0.0.1:8188'
COMFYUI_DIR = r'C:\kumori\dev\ComfyUI_training\ComfyUI'
COMFYUI_MAIN_SCRIPT = 'main.py'
COMFY_ENV_DIR = os.path.join(COMFYUI_DIR, "comfy_env")
START_SCRIPT = f'start cmd.exe /k "cd /d {COMFYUI_DIR} && call {COMFY_ENV_DIR}\\Scripts\\activate && python {COMFYUI_MAIN_SCRIPT} --lowvram --listen 0.0.0.0"'
MAX_ATTEMPTS = 40
ATTEMPT_DELAY = 15

def log(message):
    """Log a message both to the console and to a file."""
    with open('comfy_starter_log.txt', 'a') as f:
        f.write(f"{message}\n")
    print(message)

def run_command(command):
    """Run a command in the subprocess and handle errors."""
    print(f"Running command: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to run command: {e}")
        log(f"Failed to run command: {e}")

def start_comfyui():
    """Start the ComfyUI server in a new terminal window."""
    subprocess.Popen(START_SCRIPT, shell=True)
    log("Started ComfyUI server in a new terminal window.")

def is_comfyui_ready():
    """Check if the ComfyUI server is ready to receive requests."""
    url = f"{SERVER_ADDRESS}/prompt"
    for attempt in range(MAX_ATTEMPTS):
        try:
            log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Attempt {attempt + 1}: Checking if ComfyUI is ready at {url}...")
            response = requests.get(url)
            if response.status_code == 200:
                log(f"ComfyUI is ready after {attempt + 1} attempts.")
                return True
            log(f"Status code received: {response.status_code}, Response: {response.text}")
        except requests.exceptions.ConnectionError as e:
            remaining_attempts = MAX_ATTEMPTS - attempt - 1
            log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Attempt {attempt + 1}: ComfyUI not ready yet. {remaining_attempts} attempts left. Retrying in {ATTEMPT_DELAY} seconds...")
            log(f"ConnectionError: {e}")
            time.sleep(ATTEMPT_DELAY)
    log("ComfyUI is not ready.")
    return False

def is_port_in_use(port):
    """Check if a given port is in use by any process."""
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            return True
    return False

def kill_process_using_port(port):
    """Kill the process currently using the given port."""
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            process = psutil.Process(conn.pid)
            log(f"Killing process {process.name()} (PID: {conn.pid}) using port {port}.")
            process.terminate()
            process.wait()
            break

def initialize_comfyui():
    """Ensure that ComfyUI is running, starting it if necessary."""
    if is_port_in_use(8188):
        log("ComfyUI port is in use. Checking if ComfyUI is ready...")
        if is_comfyui_ready():
            log("ComfyUI is already running and ready.")
            return True
        else:
            log("ComfyUI is not responding correctly. Restarting ComfyUI.")
            kill_process_using_port(8188)
            time.sleep(5)
            start_comfyui()
            if not is_comfyui_ready():
                log("ComfyUI did not start in time.")
                return False
    else:
        log("ComfyUI port is not in use. Starting ComfyUI in a new terminal window.")
        start_comfyui()
        if not is_comfyui_ready():
            log("ComfyUI did not start in time.")
            return False

    return True

if __name__ == "__main__":
    log("Clearing the log file...")
    with open('comfy_starter_log.txt', 'w') as f:
        f.write("")
    initialize_comfyui()
    log("Initialization finished.")