import json
import os
import random
import time
from random import randint

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

MIN_ROOM_WIDTH = 60
MIN_ROOM_LENGTH = 60
MIN_ROOM_HEIGHT = 100
MIN_FURNITURE_NUMBER = 2
MIN_FURNITURE_STEP = 5
MIN_THETA_STEP = 5


class ArrangementEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(
        self, selections_directory, max_room_width, max_room_length, max_room_height
    ):
        super(ArrangementEnv, self).__init__()
        self.selections_directory = selections_directory
        self.max_room_width = max_room_width
        self.max_room_length = max_room_length
        self.max_room_height = max_room_height
        with open(f"{selections_directory}/max_indices.json", "r") as f:
            self.max_indices = json.load(f)
        self.max_room_style = self.max_indices["Room Style"]
        self.max_room_type = self.max_indices["Room Type"]
        self.max_furniture_id = self.max_indices["Furniture ID"]
        self.max_furniture_style = self.max_indices["Furniture Style"]
        self.max_furniture_width = self.max_indices["Furniture Width"]
        self.max_furniture_depth = self.max_indices["Furniture Depth"]
        self.max_furniture_height = self.max_indices["Furniture Height"]
        self.max_furniture_per_room = self.max_indices["Furniture Per Room"]
        self.max_selection_number = self.max_indices["Number of Selections"]
        self.room_observation = None
        self.furniture_observation = None
        self.current_furniture_number = 0
        self.reward = 0
        self.truncated = False
        self.terminated = False
        # Action space: [Terminate episode, X Position, Y Position, Angle]
        # Action space moves the furniture at the current_furniture_number to the given position
        self.action_space = spaces.MultiDiscrete(
            [
                2,  # Terminate episode (0: continue, 1: terminate)
                max_room_width // MIN_FURNITURE_STEP + 1,  # X Position
                max_room_length // MIN_FURNITURE_STEP + 1,  # Y Position
                360 // MIN_THETA_STEP + 1,  # Angle
            ],
            dtype=int,
        )
        # Room properties: [Width, Length, Height, Style, Type]
        # Each furniture item: [ID, Style, Width, Depth, Height, X, Y, Theta]
        self.observation_space = spaces.MultiDiscrete(
            [
                max_room_width + 1 - MIN_ROOM_WIDTH,  # Width
                max_room_length + 1 - MIN_ROOM_WIDTH,  # Length
                max_room_height + 1 - MIN_ROOM_WIDTH,  # Height
                self.max_room_style + 1,  # Style (+1 since the index is non_inclusive)
                self.max_room_type
                + 1,  # Room Type (Livingroom, Bedroom etc.) (+1 since the index is non_inclusive)
            ]
            + (
                [
                    self.max_furniture_id
                    + 1
                    + 1,  # ID (+1 since -1 is used for empty slot) -1 -> 0; addition +1 for noninclusivity
                    self.max_furniture_style
                    + 1
                    + 1,  # Style (+1 since -1 is used for empty slot) -1 -> 0; addition +1 for noninclusivity
                    self.max_furniture_width + 1,  # Width
                    self.max_furniture_depth + 1,  # Depth
                    self.max_furniture_height + 1,  # Height
                    max_room_width // MIN_FURNITURE_STEP + 1,  # X Position
                    max_room_length // MIN_FURNITURE_STEP + 1,  # Y Position
                    360 // MIN_THETA_STEP + 1,  # Angle
                ]
                * self.max_furniture_per_room  # Repeat for each furniture item
            ),
            dtype=int,
        )

    def step(self, action):
        self.terminated = bool(action[0] == 1)
        self.truncated = bool(
            (np.any(self.current_furniture_number >= self.max_furniture_per_room))
            or (np.any(self.furniture_observation[self.current_furniture_number] == -1))
        )

        if self.terminated or self.truncated:
            info = {}
            return (
                np.concatenate(
                    (
                        self.room_observation,
                        self.furniture_observation.flatten(),
                    )
                ),
                self.reward,
                self.terminated,
                self.truncated,
                info,
            )

        furniture_x = action[1]
        furniture_y = action[2]
        furniture_theta = action[3]
        furniture_x_index_in_observation_space = 5
        furniture_y_index_in_observation_space = 6
        furniture_theta_index_in_observation_space = 7
        self.furniture_observation[self.current_furniture_number][
            furniture_x_index_in_observation_space
        ] = furniture_x
        self.furniture_observation[self.current_furniture_number][
            furniture_y_index_in_observation_space
        ] = furniture_y
        self.furniture_observation[self.current_furniture_number][
            furniture_theta_index_in_observation_space
        ] = furniture_theta
        # assert 0 <= furniture_x < self.max_room_width//MIN_FURNITURE_STEP + 1, f"furniture x:{furniture_x} is larger than the max: {self.max_room_width//MIN_FURNITURE_STEP + 1}"
        # assert 0 <= furniture_y < self.max_room_length//MIN_FURNITURE_STEP + 1, f"furniture y:{furniture_y} is larger than the max: {self.max_room_length//MIN_FURNITURE_STEP + 1}"
        # assert 0 <= furniture_theta < 360//MIN_THETA_STEP + 1, f"furniture theta:{furniture_theta} is larger than the max: {360//MIN_THETA_STEP + 1}"

        room_observations_dictionary = {
            "Width": self.room_observation[0] + MIN_ROOM_WIDTH,
            "Length": self.room_observation[1] + MIN_ROOM_LENGTH,
            "Height": self.room_observation[2] + MIN_ROOM_HEIGHT,
            "Style": self.room_observation[
                3
            ],  # no -1 since the room style is always defined
            "Type": self.room_observation[4],
        }
        furniture_observations_dictionary = []
        for f in self.furniture_observation:
            furniture_observations_dictionary.append(
                {
                    "ID": f[0]
                    - 1,  # -1 since observation space back to embedding space is -1 since observations start a 0
                    "Style": f[1]
                    - 1,  # -1 since observation space back to embedding space is -1 since observations start a 0
                    "Width": f[2],
                    "Depth": f[3],
                    "Height": f[4],
                    "X": f[5] * MIN_FURNITURE_STEP,
                    "Y": f[6] * MIN_FURNITURE_STEP,
                    "Theta": f[7] * MIN_THETA_STEP,
                }
            )
        arrangement = {
            "Room": room_observations_dictionary,
            "Furniture": furniture_observations_dictionary,
        }
        arrangement_reward = reward(arrangement, self.current_furniture_number)
        self.current_furniture_number += 1
        # assert(not np.isnan(arrangement_reward))
        # print("reward:", arrangement_reward)
        self.reward += arrangement_reward
        info = {}
        return (
            np.concatenate(
                (self.room_observation, self.furniture_observation.flatten())
            ),
            self.reward,
            self.terminated,
            self.truncated,
            info,
        )

    def reset(self, seed=None, selection_file=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.truncated = False
        room_width = randint(0, self.max_room_width - MIN_ROOM_WIDTH)
        room_length = randint(0, self.max_room_length - MIN_ROOM_LENGTH)
        room_height = randint(0, self.max_room_height - MIN_ROOM_HEIGHT)

        # If a specific selection file is provided, use it. Otherwise, choose randomly.
        if selection_file:
            selection_path = os.path.join(self.selections_directory, selection_file)
        else:
            selection_path = os.path.join(
                self.selections_directory,
                f"selection_{randint(1, self.max_selection_number)}.json",
            )
        with open(selection_path, "r") as f:
            selection = json.load(f)
        style = selection["Room"]["Style"]
        type = selection["Room"]["Type"]
        furnitures = selection["Furniture"]
        # initializing positions to the furnitures
        furnitures = [
            {
                "ID": furniture["ID"]
                + 1,  # +1 since -1 is used for empty slot -1 -> 0;
                "Style": furniture["Style"]
                + 1,  # +1 since -1 is used for empty slot) -1 -> 0;
                "Width": furniture["Width"],
                "Depth": furniture["Depth"],
                "Height": furniture["Height"],
                "X": 0,
                "Y": 0,
                "Theta": 0,
            }
            for furniture in furnitures
        ]

        # assert (
        #     len(furnitures) == self.max_furniture_per_room
        # ), f"the size of the selected furnitures: {furnitures} does not match the max furniture per room: {self.max_furniture_per_room}"
        # assert (
        #     furnitures[0]["ID"] != -1
        # ), f"the first value in the selected furnitures: {furnitures} is -1"

        for f in furnitures:
            break
            # assert (
            #     0 <= f["ID"] < self.max_furniture_id + 1 + 1
            # ), f"furniture id: {f['ID']} is larger than the max: {self.max_furniture_id + 1 + 1}"
            # assert (
            #     0 <= f["Style"] < self.max_furniture_style + 1 + 1
            # ), f"furniture style: {f['Style']} is larger than the max: {self.max_furniture_style + 1 + 1}"
            # assert (
            #     0 <= f["Width"] < self.max_furniture_width + 1
            # ), f"furniture width: {f['Width']} is larger than the max: {self.max_furniture_width}"
            # assert (
            #     0 <= f["Depth"] < self.max_furniture_depth + 1
            # ), f"furniture depth: {f['Depth']} is larger than the max: {self.max_furniture_depth}"
            # assert (
            #     0 <= f["Height"] < self.max_furniture_height + 1
            # ), f"furniture height: {f['Height']} is larger than the max: {self.max_furniture_height}"
        self.current_furniture_number = 0
        self.reward = 0
        self.room_observation = np.array(
            [room_width, room_length, room_height, style, type]
        )
        self.furniture_observation = np.array(
            [list(furniture.values()) for furniture in furnitures], dtype=int
        )
        info = {}
        # print(
        #     f"INITIAL OBSERVATION: {np.concatenate((self.room_observation, self.furniture_observation.flatten()))}"
        # )
        # print(f"Room observation shape: {self.room_observation.shape}")
        # print(f"Furniture observation shape: {self.furniture_observation.shape}")
        return (
            np.concatenate(
                (self.room_observation, self.furniture_observation.flatten())
            ),
            info,
        )


def reward(arrangement, current_furniture_index):
    def is_collision(furniture1, furniture2):
        # Bottom middle as the reference point for furniture1 (current furniture)
        f1_x = furniture1["X"]
        f1_y = furniture1["Y"]
        f1_width = furniture1["Width"]
        f1_depth = furniture1["Depth"]

        # Bottom middle as the reference point for furniture2 (other furniture)
        f2_x = furniture2["X"]
        f2_y = furniture2["Y"]
        f2_width = furniture2["Width"]
        f2_depth = furniture2["Depth"]

        # Check for overlap in the X and Y axes
        overlap_x = abs(f1_x - f2_x) < (f1_width + f2_width) / 2
        overlap_y = abs(f1_y - f2_y) < (f1_depth + f2_depth) / 2
        return overlap_x and overlap_y

    def is_outside_room(furniture, room_width, room_length):
        # Check if the furniture is outside the room based on the bottom middle reference
        f_x = furniture["X"]
        f_y = furniture["Y"]
        f_width = furniture["Width"]
        f_depth = furniture["Depth"]

        # Ensure the furniture stays within the room boundaries
        return not (
            0 <= f_x - f_width / 2 <= room_width
            and 0 <= f_y - f_depth / 2 <= room_length
        )

    current_furniture = arrangement["Furniture"][current_furniture_index]
    room_width = arrangement["Room"]["Width"]
    room_length = arrangement["Room"]["Length"]

    # Check if the current furniture is outside the room
    if is_outside_room(current_furniture, room_width, room_length):
        return -1  # Penalize for being outside the room

    # Check for collisions with other furniture
    for i, other_furniture in enumerate(arrangement["Furniture"]):
        if i != current_furniture_index and is_collision(
            current_furniture, other_furniture
        ):
            return -1  # Penalize for collisions

    # Reward for being inside the room and no collisions
    return 1


if __name__ == "__main__":
    # assert embeddings is not None, "Embeddings failed to load"
    env = ArrangementEnv("./selections", 144, 144, 120)
    check_env(env)
    models_path = f"models/{int(time.time())}/"
    log_path = f"logs/{int(time.time())}/"
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    model = PPO("MlpPolicy", env, ent_coef=0.01, verbose=1, tensorboard_log=log_path)
    TIMESTEPS = 10000
    episodes = 0
    while episodes < 1000000:
        episodes += 1
        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            tb_log_name="PPO",
        )
        model.save(f"{models_path}/{TIMESTEPS*episodes}")
