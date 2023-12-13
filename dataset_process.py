import json
import glob
import os
from collections import OrderedDict

# List of parent directories

# parent_directories = [
#     'data/data_vs_cfr_first_order_fixed_seed_position0',
#     # Add other directory paths here
# ]

parent_directories = os.listdir('data')
print(parent_directories)

# Define postfixes
postfixes = [
    'belief.json', 'long_memory.json', 'obs.json', 'act.json',
    'opponent_act.json', 'opponent_obs.json', 
    'pattern_model.json', 'plan.json'
]



def get_opponent_action_round_id(obs_round_id):
    game_number, round_number = obs_round_id.split('_')
    opponent_action_round_number = int(round_number) + 1
    return f"{game_number}_{opponent_action_round_number}"
# Function to load JSON objects from a file
def load_json_objects(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {file_path}: {e}")

# Process each directory and accumulate the results
all_rounds_data = []

for base_directory in parent_directories:
    combined_data = {}
    data = {postfix.split('.')[0]: {} for postfix in postfixes}

    # Load data from each file in the current directory
    for postfix in postfixes:
        pattern = os.path.join('data/'+base_directory, f'*{postfix}')
        files = glob.glob(pattern)
        print(files)
        if files:
            filepath = files[0]
            for json_object in load_json_objects(filepath):
                for key, value in json_object.items():
                    if key != "message":  # Assuming 'message' is not needed
                        data[postfix.split('.')[0]][key] = value

    # Combine data for each round
    rounds = set()
    for key in data:
        rounds.update(data[key].keys())

    for round_id in rounds:
        if 'belief' in data['belief'].get(round_id, {}) and 'plan' in data['plan'].get(round_id, {}):
            combined_round_data = {
                'round_id': round_id,
                'parent_directory': base_directory,
                'data': {
                    'obs': data.get('obs', {}).get(round_id, {}).get('readable_text_obs', ''),
                    'opponent_obs': data.get('opponent_obs', {}).get(get_opponent_action_round_id(round_id), {}).get('raw_obs', {}),
                    'pattern': data.get('pattern_model', {}).get(round_id, ''),
                    'belief': data.get('belief', {}).get(round_id, {}).get('belief', ''),
                    'plan': data.get('plan', {}).get(round_id, {}).get('plan', ''),
                    'action': data.get('act', {}).get(round_id, {}).get('act', ''),
                    'opponent_action': data.get('opponent_act', {}).get(get_opponent_action_round_id(round_id), {}).get('act', ''),
                    'long_memory': data.get('long_memory', {}).get(round_id.split("_")[0], {}).get('long_memory', '')
                }
            }
            all_rounds_data.append(combined_round_data)

def round_id_sort_key(round_data):
    game_number, round_number = map(int, round_data['round_id'].split('_'))
    return game_number, round_number

def sort_data_by_round_id(data_list):
    def round_id_sort_key(round_data):
        game_number, round_number = map(int, round_data['round_id'].split('_'))
        return game_number, round_number
    return sorted(data_list, key=round_id_sort_key)

# Group data by parent_directory
grouped_data = {}
for item in all_rounds_data:
    parent_dir = item['parent_directory']
    if parent_dir not in grouped_data:
        grouped_data[parent_dir] = []
    grouped_data[parent_dir].append(item)

# Sort data within each group and concatenate
sorted_grouped_data = []
for parent_dir in grouped_data:
    sorted_data = sort_data_by_round_id(grouped_data[parent_dir])
    sorted_grouped_data.extend(sorted_data)

# Save the sorted and grouped data to a new file
with open('sorted_grouped_combined_game_process.json', 'w') as f:
    json.dump(sorted_grouped_data, f, indent=4)

print("Sorted and grouped game processes saved to 'sorted_grouped_combined_game_process.json'")
# Save all rounds data to a new file
with open('all_combined_game_process.json', 'w') as f:
    json.dump(all_rounds_data, f, indent=4)

print("All game processes combined and saved to 'all_combined_game_process.json'")
