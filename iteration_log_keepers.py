import os
import pandas as pd

# Load the CSV file
csv_file_path = "iteration_log.csv"
data = pd.read_csv(csv_file_path)

# Get the list of filenames from the CSV, fixing path styles
csv_filenames = [os.path.basename(path) for path in data['File Path'].tolist()]

# Get the list of files in the api_outputs/ directory
output_directory = "api_outputs/"
directory_filenames = os.listdir(output_directory)

# Identify kept and deleted filenames
kept_filenames = [filename for filename in csv_filenames if filename in directory_filenames]
deleted_filenames = [filename for filename in csv_filenames if filename not in kept_filenames]

# Count the number of kept files vs. deleted files
kept_count = len(kept_filenames)
deleted_count = len(deleted_filenames)
print(f"Kept: {kept_count}, Deleted: {deleted_count}\n")

# Create kept and deleted dataframes
kept_data = data[data['File Path'].apply(lambda x: os.path.basename(x)).isin(kept_filenames)]
deleted_data = data[data['File Path'].apply(lambda x: os.path.basename(x)).isin(deleted_filenames)]

kept_data.to_csv("kept_iterations.csv", index=False)
print("kept_iterations.csv has been created.\n")

# Helper function for percentage calculation
def calculate_percentage(kept, deleted):
    total = kept + deleted
    return (kept / total) * 100 if total > 0 else 0

# Usage Counts
def get_combined_lora_counts(data):
    lora_combined = pd.concat([data['LORA2'], data['LORA3']])
    return lora_combined.value_counts()

# Calculate and print scheduler usage
print("Analyzing Kept Files:")

scheduler_counts_kept = kept_data['Scheduler'].value_counts()
sampler_counts_kept = kept_data['Sampler'].value_counts()

print("\nScheduler Usage (Kept):")
print(scheduler_counts_kept.sort_values(ascending=False))

print("\nSampler Usage (Kept):")
print(sampler_counts_kept.sort_values(ascending=False))

lora_counts_kept = get_combined_lora_counts(kept_data)
print("\nLORA Usage (Kept):")
print(lora_counts_kept.sort_values(ascending=False))

# Top combinations
kept_combination_counts = kept_data.groupby(['Scheduler', 'Sampler', 'LORA2', 'LORA3']).size().reset_index(name='Counts')
top_kept_combinations = kept_combination_counts.sort_values(by='Counts', ascending=False).head(10)

print("\nTop 10 Combinations (Kept):")
print(top_kept_combinations)

print("\nAnalyzing Deleted Files:")

# Calculate and print scheduler usage
scheduler_counts_deleted = deleted_data['Scheduler'].value_counts()
sampler_counts_deleted = deleted_data['Sampler'].value_counts()

print("\nScheduler Usage (Deleted):")
print(scheduler_counts_deleted.sort_values(ascending=False))

print("\nSampler Usage (Deleted):")
print(sampler_counts_deleted.sort_values(ascending=False))

lora_counts_deleted = get_combined_lora_counts(deleted_data)
print("\nLORA Usage (Deleted):")
print(lora_counts_deleted.sort_values(ascending=False))

# Top combinations
deleted_combination_counts = deleted_data.groupby(['Scheduler', 'Sampler', 'LORA2', 'LORA3']).size().reset_index(name='Counts')
top_deleted_combinations = deleted_combination_counts.sort_values(by='Counts', ascending=False).head(10)

print("\nTop 10 Combinations (Deleted):")
print(top_deleted_combinations)

# Calculate and display usage percentages
print("\nLORA Usage Percentage Kept vs. Deleted:")
lora_percentages = pd.DataFrame({
    'Kept': lora_counts_kept,
    'Deleted': lora_counts_deleted
}).fillna(0)
lora_percentages['Kept %'] = lora_percentages.apply(lambda row: calculate_percentage(row['Kept'], row['Deleted']), axis=1)
print(lora_percentages.sort_values(by='Kept %', ascending=False))

print("\nScheduler Usage Percentage Kept vs. Deleted:")
scheduler_percentages = pd.DataFrame({
    'Kept': scheduler_counts_kept,
    'Deleted': scheduler_counts_deleted
}).fillna(0)
scheduler_percentages['Kept %'] = scheduler_percentages.apply(lambda row: calculate_percentage(row['Kept'], row['Deleted']), axis=1)
print(scheduler_percentages.sort_values(by='Kept %', ascending=False))

print("\nSampler Usage Percentage Kept vs. Deleted:")
sampler_percentages = pd.DataFrame({
    'Kept': sampler_counts_kept,
    'Deleted': sampler_counts_deleted
}).fillna(0)
sampler_percentages['Kept %'] = sampler_percentages.apply(lambda row: calculate_percentage(row['Kept'], row['Deleted']), axis=1)
print(sampler_percentages.sort_values(by='Kept %', ascending=False))