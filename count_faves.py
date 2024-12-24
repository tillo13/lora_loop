import os
import json
from collections import Counter

# Load configurations from the global variable JSON file
def load_configurations():
    with open('global_variables.json', 'r', encoding='utf-8') as f:
        configs = json.load(f)
        return {key: value['value'] for key, value in configs.items()}

config = load_configurations()
best_samplers_schedulers = config['BEST_SAMPLERS_SCHEDULERS']

# Generate all possible combos
known_samplers = ["euler", "lms", "dpmpp_2m", "dpm_adaptive", 
                  "dpmpp_2s_ancestral", "ddim", "deis", "ipndm_v", 
                  "heun", "heunpp2", "uni_pc_bh2", "lcm"]

known_schedulers = ["simple", "beta", "sgm_uniform", 
                    "ddim_uniform", "karras", "normal"]

all_possible_combos = {
    f"{sampler}_{scheduler}" 
    for sampler in known_samplers 
    for scheduler in known_schedulers
}

# Convert best combos from list of lists to set of strings for comparison
attempted_combos = {f"{sampler}_{scheduler}" for sampler, scheduler in best_samplers_schedulers}

# Determine excluded combos
excluded_combos = all_possible_combos - attempted_combos

def extract_combo_from_filename(filename):
    parts = filename.split('_')
    sampler = None
    scheduler = None

    for i in range(len(parts) - 1):
        potential_sampler = f"{parts[i]}_{parts[i+1]}"
        if potential_sampler in known_samplers and not sampler:
            sampler = potential_sampler
        
        potential_scheduler = f"{parts[i]}_{parts[i+1]}"
        if potential_scheduler in known_schedulers and not scheduler:
            scheduler = potential_scheduler

    combined_string = "_".join(parts)
    for s in known_samplers:
        if s in combined_string and not sampler:
            sampler = s
    for sch in known_schedulers:
        if sch in combined_string and not scheduler:
            scheduler = sch

    return sampler, scheduler

def main():
    api_output_folder = 'api_outputs/'
    if not os.path.exists(api_output_folder):
        print(f"Folder '{api_output_folder}' does not exist.")
        return

    combos_counter = Counter()
    samplers_counter = Counter()
    schedulers_counter = Counter()
    best_combos_counter = Counter()  # Initialize the best_combos_counter
    directory_counts = Counter()
    total_files = 0
    unmatched_files = []

    for root, dirs, files in os.walk(api_output_folder):
        file_count = 0
        for filename in files:
            total_files += 1
            file_path = os.path.join(root, filename)
            print(f"Processing file: {file_path}")

            if filename.endswith(('.png', '.jpg', '.jpeg')):
                sampler, scheduler = extract_combo_from_filename(filename)
                if sampler and scheduler:
                    combo = f"{sampler}_{scheduler}"
                    combos_counter[combo] += 1
                    samplers_counter[sampler] += 1
                    schedulers_counter[scheduler] += 1

                    # Update best_combos_counter if the combo is in best_samplers_schedulers
                    if combo in attempted_combos:
                        best_combos_counter[combo] += 1
                        
                    print(f" - Full combo found: {combo}")
                elif sampler:
                    samplers_counter[sampler] += 1
                    print(f" - Sampler found: {sampler}")
                elif scheduler:
                    schedulers_counter[scheduler] += 1
                    print(f" - Scheduler found: {scheduler}")
                else:
                    unmatched_files.append(filename)
                    print(" - No valid scheduler/sampler combo found")
                file_count += 1
        directory_counts[root] = file_count

    total_matched_images = sum(combos_counter.values())
    total_best_matched_images = sum(best_combos_counter.values())  # Calculate total best matched images

    print("\nTop Combos ALL:")
    sorted_combos = combos_counter.most_common()
    for rank, (combo, count) in enumerate(sorted_combos, 1):
        percentage = (count / total_matched_images) * 100 if total_matched_images > 0 else 0
        print(f" {rank}. {combo}: {percentage:.2f}% ({count} count)")

    print("\nTop Combos BEST_SCHEDULER/SAMPLER from global_variables.json:")
    sorted_best_combos = best_combos_counter.most_common()
    for rank, (combo, count) in enumerate(sorted_best_combos, 1):
        percentage = (count / total_best_matched_images) * 100 if total_best_matched_images > 0 else 0
        print(f" {rank}. {combo}: {percentage:.2f}% ({count} count)")

    # Code to compare Top Combos ALL vs BEST_SCHEDULER/SAMPLER top combos
    top_combos_set = set(combos_counter.keys())
    best_scheduler_combos_set = set(best_combos_counter.keys())

    common_combos = [
        (combo, combos_counter[combo], best_combos_counter[combo])
        for combo in top_combos_set.intersection(best_scheduler_combos_set)
    ]
    # Sort common combos by count in descending order
    sorted_common_combos = sorted(common_combos, key=lambda x: x[1], reverse=True)

    print("\nComparison of Top Combos ALL and BEST_SCHEDULER/SAMPLER Combos:")
    print("Common Combos:")
    for combo, count_all, count_best in sorted_common_combos:
        percentage_all = (count_all / total_matched_images) * 100 if total_matched_images > 0 else 0
        percentage_best = (count_best / total_best_matched_images) * 100 if total_best_matched_images > 0 else 0
        print(f" {combo}: ALL - {count_all} ({percentage_all:.2f}%), BEST - {count_best} ({percentage_best:.2f}%)")

    print("\nUnique to Top Combos ALL:")
    for combo in top_combos_set - best_scheduler_combos_set:
        count_all = combos_counter[combo]
        percentage_all = (count_all / total_matched_images) * 100 if total_matched_images > 0 else 0
        print(f" {combo}: {count_all} ({percentage_all:.2f}%)")

    print("\nUnique to BEST_SCHEDULER/SAMPLER Combos:")
    # The `exclusive_best_scheduler_combos` list should theoretically be empty
    for combo in best_scheduler_combos_set - top_combos_set:
        count_best = best_combos_counter[combo]
        percentage_best = (count_best / total_best_matched_images) * 100 if total_best_matched_images > 0 else 0
        print(f" {combo}: {count_best} ({percentage_best:.2f}%)")

    print("\nUnused samplers:")
    unused_samplers = [sampler for sampler in known_samplers if sampler not in samplers_counter]
    for sampler in unused_samplers:
        print(f" {sampler}")

    print("\nUnused schedulers:")
    unused_schedulers = [scheduler for scheduler in known_schedulers if scheduler not in schedulers_counter]
    for scheduler in unused_schedulers:
        print(f" {scheduler}")

    print("\nMost used samplers:")
    total_samplers_count = sum(samplers_counter.values())
    sorted_samplers = samplers_counter.most_common()
    for rank, (sampler, count) in enumerate(sorted_samplers, 1):
        percentage = (count / total_samplers_count) * 100 if total_samplers_count > 0 else 0
        print(f" {rank}. {sampler}: {percentage:.2f}% ({count} count)")

    print("\nMost used schedulers:")
    total_schedulers_count = sum(schedulers_counter.values())
    sorted_schedulers = schedulers_counter.most_common()
    for rank, (scheduler, count) in enumerate(sorted_schedulers, 1):
        percentage = (count / total_schedulers_count) * 100 if total_schedulers_count > 0 else 0
        print(f" {rank}. {scheduler}: {percentage:.2f}% ({count} count)")

    print("\nDirectory file counts:")
    for directory, count in directory_counts.items():
        print(f"{directory}: {count} file(s)")

    print("\n=== Final Counts ===")
    print(f"Total files processed: {total_files}")
    print(f"Total matched combos: {total_matched_images}")

    print("\nUnmatched files:")
    for unmatched_file in unmatched_files:
        print(unmatched_file)

    # Suggested New Combos:
    suggested_new_combos = [
        (combo, combos_counter[combo])
        for combo in top_combos_set - best_scheduler_combos_set
    ]
    sorted_suggested_new_combos = sorted(suggested_new_combos, key=lambda x: x[1], reverse=True)

    print("\nSuggested New Combos (not in BEST_SCHEDULER/SAMPLER, by top counts):")
    for combo, count in sorted_suggested_new_combos:
        percentage = (count / total_matched_images) * 100 if total_matched_images > 0 else 0
        print(f" {combo}: {count} ({percentage:.2f}%)")

if __name__ == "__main__":
    main()