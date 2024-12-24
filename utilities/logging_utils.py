import traceback
import csv
import os
from datetime import datetime

# Define constants for log files in the utilities module as they were in the app.py
LOG_FILE = 'superhero_test_log.txt'
ITERATION_LOG_FILE = 'iteration_log.csv'

def log(message):
    """ Log a message both to the console and to a file. """
    with open(LOG_FILE, 'a') as f:
        f.write(f"{message}\n")
    print(message)

def log_error(message):
    """ Log an error message with traceback both to the console and to a file. """
    with open(LOG_FILE, 'a') as f:
        f.write(f"{message}\n")
        f.write(traceback.format_exc() + "\n")
    print(message)
    print(traceback.format_exc())

def log_iteration_details(iter_num, time_start, time_end, inference_steps, latent_batch_amount, scheduler, sampler, file_paths, lora2, lora3, prompt_text):
    """ Log details of each iteration in a CSV file. """
    file_exists = os.path.isfile(ITERATION_LOG_FILE)
    with open(ITERATION_LOG_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Iteration Number", "Start Time", "End Time", "Inference Steps", 
                            "Latent Batch Amount", "Scheduler", "Sampler", 
                            "LORA2", "LORA3", "File Path", "Time to Complete", "Prompt Text"])
        for file_path in file_paths:
            writer.writerow([
                iter_num,
                time_start.strftime("%Y-%m-%d %H:%M:%S"),
                time_end.strftime("%Y-%m-%d %H:%M:%S"),
                inference_steps,
                latent_batch_amount,
                scheduler,
                sampler,
                lora2,
                lora3,
                file_path,
                str(time_end - time_start),
                prompt_text  # Add prompt_text to each row
            ])