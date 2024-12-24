import requests
import json
from collections import Counter
from datetime import datetime
import os

# -------------------------
# GLOBAL VARIABLES SECTION
# -------------------------

SERVER_ADDRESS = 'http://127.0.0.1:8188'
LOG_DIRECTORY = 'logs'
os.makedirs(LOG_DIRECTORY, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIRECTORY, f'check_queue_log_{timestamp}.txt')

# -------------------------
# HELPER FUNCTIONS SECTION
# -------------------------

def log(message):
    """ Log a message both to the console and to a file. """
    with open(LOG_FILE, 'a') as f:
        f.write(f"{message}\n")
    print(message)

def get_queue_status():
    """ Fetch all prompt statuses from ComfyUI """
    url = f"{SERVER_ADDRESS}/history"
    
    try:
        log(f"Fetching queue status from {url}...")
        response = requests.get(url)
        
        if response.status_code == 200:
            log("Queue status fetched successfully.")
            return response.json()
        else:
            log(f"Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.ConnectionError as e:
        log(f"ConnectionError: {e}")
        return None

def get_prompt_status(prompt_id):
    """ Fetches the status for a given prompt ID from ComfyUI """
    url = f"{SERVER_ADDRESS}/history/{prompt_id}"
    
    try:
        log(f"Fetching status for prompt ID {prompt_id} from {url}...")
        response = requests.get(url)
        
        if response.status_code == 200:
            log("Status fetched successfully.")
            return response.json()
        else:
            log(f"Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.ConnectionError as e:
        log(f"ConnectionError: {e}")
        return None

# -------------------------
# MAIN EXECUTION SECTION
# -------------------------

def main():
    """ Main function to execute the queue check. """
    log("Checking ComfyUI queue status...")
    
    queue_status = get_queue_status()
    if queue_status:
        status_count = Counter()
        
        log("Processing all prompts...")
        
        for prompt_id, details in queue_status.items():
            status = details.get('status', {})
            status_str = status.get('status_str', 'unknown')
            completed = status.get('completed')

            status_info = f"{status_str} (completed: {completed})"
            if status_str == "success" and completed:
                status_info = f"{status_str} (completed: {completed}) - Confirmed"
            status_count[status_info] += 1

            log(f"\nDetails for prompt ID: {prompt_id}")
            log(f"Status: {status_str}")
            log(f"Completed: {completed}")
            log(json.dumps(details, indent=4))

        log("\nSummary of prompt statuses:")
        for status_info, count in status_count.items():
            log(f"{status_info}: {count}")
    else:
        log("Failed to fetch queue status.")

    log("Completed successfully!")

if __name__ == "__main__":
    log("Clearing the log file...")
    with open(LOG_FILE, 'w') as f:
        f.write("")
    main()