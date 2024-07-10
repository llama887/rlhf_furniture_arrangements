import json
import math
import os
import random
import re
from typing import Dict, List, Tuple, Union

arrangement_path = "arrangements/"
os.makedirs(arrangement_path, exist_ok=True)
MIN_ROOM_WIDTH = 60
MIN_ROOM_LENGTH = 60
MIN_ROOM_HEIGHT = 100
MIN_FURNITURE_NUMBER = 2
MIN_FURNITURE_STEP = 5
MIN_THETA_STEP = 5


def json_to_embeddings(
    file_path: str,
) -> Tuple[
    Dict[int, Dict[str, Union[str, int]]],
    Dict[int, str],
    int,
    Dict[int, str],
    int,
    int,
    int,
    int,
]:
    """
    Convert a JSON file of furniture data to indexed embeddings.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        Tuple containing:
            - indexed_data (Dict[int, Dict[str, Union[str, int]]]): Indexed furniture data.
            - id_embedding (Dict[int, str]): Indexed IDs.
            - max_id_index (int): Maximum index for IDs.
            - style_embedding (Dict[int, str]): Indexed styles.
            - max_style_index (int): Maximum index for styles.
            - max_width (int): Maximum width value among the furniture items.
            - max_depth (int): Maximum depth value among the furniture items.
            - max_height (int): Maximum height value among the furniture items.
    """

    def append_if_unique(
        embedding: Dict[int, str], item: str, current_index: int
    ) -> Tuple[Dict[int, str], int]:
        if item not in embedding.values():
            embedding[current_index] = item
            current_index += 1
        return embedding, current_index

    with open(file_path, "r") as file:
        data = json.load(file)
    indexed_data = {
        -1: {"Style": -1, "Width": 0, "Depth": 0, "Height": 0}
    }  # used for padding
    id_embedding, style_embedding = {-1: "NULL"}, {-1: "NULL"}
    max_id_index, max_style_index = 0, 0
    max_width, max_depth, max_height = 0, 0, 0

    for item in data:
        id_embedding, max_id_index = append_if_unique(
            id_embedding, item["ID"], max_id_index
        )
        style_embedding, max_style_index = append_if_unique(
            style_embedding, item["Style"], max_style_index
        )
        match = re.match(
            r"W (\d+\.?\d*) / D (\d+\.?\d*) / H (\d+\.?\d*)", item["Dimensions"]
        )
        if match:
            width, depth, height = match.groups()
            dimension = (
                math.ceil(float(width)),
                math.ceil(float(depth)),
                math.ceil(float(height)),
            )
            max_width = max(max_width, dimension[0])
            max_depth = max(max_depth, dimension[1])
            max_height = max(max_height, dimension[2])
        else:
            dimension = (0, 0, 0)
        indexed_data[max_id_index] = {
            "Style": max_style_index,
            "Width": dimension[0],
            "Depth": dimension[1],
            "Height": dimension[2],
        }
    with open("embeddings.json", "w") as file:
        json.dump(
            [
                {"max_id_index": max_id_index},
                id_embedding,
                {"max_style_index": max_style_index},
                style_embedding,
                {
                    "max_width": max_width,
                    "max_depth": max_depth,
                    "max_height": max_height,
                },
                indexed_data,
            ],
            file,
            indent=4,
        )
    return (
        indexed_data,
        id_embedding,
        max_id_index,
        style_embedding,
        max_style_index,
        max_width,
        max_depth,
        max_height,
    )


def generate_arrangements(
    indexed_data: Dict[int, Dict[str, Union[str, int]]],
    max_id_index: int,
    max_style_index: int,
    number_of_arrangements: int,
    max_room_width: float,
    max_room_length: float,
    max_room_height: float,
    max_furniture_number: int,
) -> List[Dict[str, Union[Dict[str, int], List[Dict[str, int]]]]]:
    """
    Generate random room arrangements based on indexed furniture data.

    Args:
        indexed_data (Dict[int, Dict[str, Union[str, int]]]): Indexed furniture data.
        max_id_index (int): Maximum index for IDs.
        max_style_index (int): Maximum index for styles.
        number_of_arrangements (int): Number of arrangements to generate.
        max_room_width (float): Maximum room width.
        max_room_length (float): Maximum room length.
        max_room_height (float): Maximum room height.
        max_furniture_number (int): Maximum number of furniture items per room.

    Returns:
        List[Dict[str, Union[Dict[str, int], List[Dict[str, int]]]]]: List of generated arrangements.
    """
    arrangements = []
    for i in range(number_of_arrangements):
        room_width = random.randint(MIN_ROOM_WIDTH, max_room_width)
        room_length = random.randint(MIN_ROOM_LENGTH, max_room_length)
        room_height = random.randint(MIN_ROOM_LENGTH, max_room_height)
        furniture_count = random.randint(
            MIN_FURNITURE_NUMBER, max_furniture_number
        )
        style = random.randint(0, max_style_index)
        arrangement_json = {
            "Room": {
                "Width": room_width,
                "Length": room_length,
                "Height": room_height,
                "Style": style,
            },
            "Furniture": [],
        }
        for _ in range(furniture_count):
            id = random.randint(0, max_id_index)
            try:
                arrangement_json["Furniture"].append(
                    {
                        "ID": id,
                        "Style": indexed_data[id]["Style"],
                        "Width": math.ceil(indexed_data[id]["Width"]),
                        "Depth": math.ceil(indexed_data[id]["Depth"]),
                        "Height": math.ceil(indexed_data[id]["Height"]),
                        "X": random.randint(0, room_width // MIN_FURNITURE_STEP)
                        * MIN_FURNITURE_STEP,
                        "Y": random.randint(
                            0, room_length // MIN_FURNITURE_STEP
                        )
                        * MIN_FURNITURE_STEP,
                        "Theta": random.randint(0, 360 // MIN_THETA_STEP)
                        * MIN_THETA_STEP,
                    }
                )
            except KeyError:
                arrangement_json["Furniture"].append(
                    {
                        "ID": id,
                        "Style": indexed_data[str(id)]["Style"],
                        "Width": math.ceil(indexed_data[str(id)]["Width"]),
                        "Depth": math.ceil(indexed_data[str(id)]["Depth"]),
                        "Height": math.ceil(indexed_data[str(id)]["Height"]),
                        "X": random.randint(0, room_width // MIN_FURNITURE_STEP)
                        * MIN_FURNITURE_STEP,
                        "Y": random.randint(
                            0, room_length // MIN_FURNITURE_STEP
                        )
                        * MIN_FURNITURE_STEP,
                        "Theta": random.randint(0, 360 // MIN_THETA_STEP)
                        * MIN_THETA_STEP,
                    }
                )
        arrangements.append(arrangement_json)
    with open(f"arrangements.json", "w") as file:
        json.dump(arrangements, file, indent=4)
    return arrangements


def reward(
    arrangement: Dict[str, Union[Dict[str, int], List[Dict[str, int]]]]
) -> float:
    """
    Given an arrangement, returns a synthetic reward value.

    Args:
        arrangement (Dict[str, Union[Dict[str, int], List[Dict[str, int]]]]): A room arrangement that includes the room dimensions and a list of furniture items and their placement.

    Returns:
        float: A synthetic reward value currently based on furniture being close to the center.
    """

    def distance(
        point1: Tuple[float, float], point2: Tuple[float, float]
    ) -> float:
        return (
            (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
        ) ** 0.5

    def check_collisions(
        furniture: Dict[str, int], furniture_list: List[Dict[str, int]]
    ) -> bool:
        for item in furniture_list:
            if (
                abs(furniture["X"] - item["X"]) < furniture["Width"]
                and abs(furniture["Y"] - item["Y"]) < furniture["Depth"]
            ):
                return True
        return False

    room_center = [
        arrangement["Room"]["Width"] // 2,
        arrangement["Room"]["Length"] // 2,
    ]
    room_style = arrangement["Room"]["Style"]
    total_reward = 0

    for furniture in arrangement["Furniture"]:
        if (furniture["ID"] == -1) or (furniture["Style"] == -1):
            break
        if check_collisions(furniture, arrangement["Furniture"]):
            total_reward -= 1
        furniture_location = [furniture["X"], furniture["Y"]]
        total_reward += (int(furniture["Style"] == room_style) + 1) / distance(
            room_center, furniture_location
        )
    return total_reward


def generate_database(
    arrangements_path: str = None,
    arrangements: List[
        Dict[str, Union[Dict[str, int], List[Dict[str, int]]]]
    ] = None,
) -> None:
    """
    Generate a database of arrangements pairs and their rewards.

    Args:
        arrangements_path (str): Path to arrangement json.
        arrangements (List[Dict[str, Union[Dict[str, int], List[Dict[str, int]]]]]): List of arrangements.

    Returns:
        None
    """
    if arrangements_path is not None:
        with open(arrangements_path, "rb") as file:
            arrangements = json.load(file)
    database = []
    random.shuffle(arrangements)
    for i in range(0, len(arrangements), 2):
        database.append(
            {
                "arrangement1": arrangements[i],
                "arrangement2": arrangements[i + 1],
                "preference": int(
                    reward(arrangements[i]) > reward(arrangements[i + 1])
                )
                + 1,
                "reward1": reward(arrangements[i]),
                "reward2": reward(arrangements[i + 1]),
            }
        )
    with open("database.json", "w") as file:
        json.dump(database, file, indent=4)


if __name__ == "__main__":
    indexed_data, _, max_id_index, _, max_style_index, _, _, _ = (
        json_to_embeddings("furniture.json")
    )
    with open("embeddings.json", "rb") as file:
        data = json.load(file)
    max_id_index = data[0]["max_id_index"]
    max_style_index = data[2]["max_style_index"]
    indexed_data = data[5]
    arrangements = generate_arrangements(
        indexed_data, max_id_index, max_style_index, 100, 144, 144, 120, 5
    )
    generate_database(arrangements=arrangements)
