import json
import os
import random

embedding_file_path = 'embedding.json'
selections_folder = 'selections'

max_indices = {
    "Room Style": 17,
    "Room Type": 2,
    "Furniture ID": 72070,
    "Furniture Style": 75,
    "Furniture Width": 3105,
    "Furniture Depth": 1515,
    "Furniture Height": 17975,
    "Furniture Per Room": 3
}

def load_json(file):
    if os.path.exists(file):
        with open(file, 'r') as file:
            return json.load(file)
    return {}

def folder_existence_check(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def max_indice_check(folder_path):
    max_indices_file_path = os.path.join(folder_path, 'max_indices.json')
    if not os.path.exists(max_indices_file_path):
        with open(max_indices_file_path, 'w') as file:
            json.dump(max_indices, file, indent=4)

def generate_random_furniture(embedding):
    furniture_id = str(random.randint(-1, 72070))
    if furniture_id in embedding:
        furniture_data = embedding[furniture_id]
        furniture_style = furniture_data["Style"]
        furniture_width = furniture_data["Width"]
        furniture_depth = furniture_data["Depth"]
        furniture_height = furniture_data["Height"]
    else:
        furniture_style = random.randint(-1, 75)
        furniture_width = random.randint(10, 100)
        furniture_depth = random.randint(10, 100)
        furniture_height = random.randint(10, 100)

    return {
        "ID": int(furniture_id),
        "Style": furniture_style,
        "Width": furniture_width,
        "Depth": furniture_depth,
        "Height": furniture_height
    }

def generate_random_arrangement(embedding):
    room_style = random.randint(0, 17)
    room_type = random.randint(0, 1)

    num_furniture = random.randint(3, 15)
    furniture_list = [generate_random_furniture(embedding) for _ in range(num_furniture)]

    for _ in range(15 - num_furniture):
        furniture_list.append({
            "ID": -1,
            "Style": -1,
            "Width": 0,
            "Depth": 0,
            "Height": 0
        })

    arrangement = {
        "Room": {
            "Style": room_style,
            "Type": room_type
        },
        "Furniture": furniture_list
    }
    return arrangement

def generate_selection(selections_folder, embedding, index):
    selection_file_name = f'selection_{index}.json'
    selection_file_path = os.path.join(selections_folder, selection_file_name)

    arrangement = generate_random_arrangement(embedding)

    with open(selection_file_path, 'w') as file:
        json.dump(arrangement, file, indent=4)

def main(embedding_file_path, selections_folder):
    folder_existence_check(selections_folder)
    max_indice_check(selections_folder)

    embedding = load_json(embedding_file_path)

    existing_files = os.listdir(selections_folder)
    next_index = max([int(f.split('_')[1].split('.')[0]) for f in existing_files if f.startswith('selection_') and f.endswith('.json')], default=0) + 1

    for i in range(next_index, next_index + 5000):
        generate_selection(selections_folder, embedding, i)

main(embedding_file_path, selections_folder)
