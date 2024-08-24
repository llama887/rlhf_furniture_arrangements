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
import torch


from arrangements import (
    MIN_FURNITURE_STEP,
    MIN_ROOM_HEIGHT,
    MIN_ROOM_LENGTH,
    MIN_ROOM_WIDTH,
    MIN_THETA_STEP,
    reward,
)


class ArrangementEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(
        self,
        training_data_json,
        max_room_width,
        max_room_length,
        max_room_height
    ):
        super(ArrangementEnv, self).__init__()
        self.max_room_width = max_room_width
        self.max_room_length = max_room_length
        self.max_room_height = max_room_height
        with open(training_data_json, "r") as f:
            self.training_data_json = json.load(f)
        self.max_room_style = self.training_data_json["Max Indices"]["Room Style"]
        self.max_room_type = self.training_data_json["Max Indices"]["Room Type"]
        self.max_furniture_id = self.training_data_json["Max Indices"]["Furniture ID"]
        self.max_furniture_style = self.training_data_json["Max Indices"]["Furniture Style"]
        self.max_furniture_width = self.training_data_json["Max Indices"]["Furniture Width"]
        self.max_furniture_depth = self.training_data_json["Max Indices"]["Furniture Depth"]
        self.max_furniture_height = self.training_data_json["Max Indices"]["Furniture Height"]
        self.max_furniture_per_room = self.training_data_json["Max Indices"]["Furniture Per Room"]
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
                self.max_room_type + 1, # Room Type (Livingroom, Bedroom etc.) (+1 since the index is non_inclusive)
            ]
            + (
                [
                    self.max_furniture_id
                    + 1 + 1,  # ID (+1 since -1 is used for empty slot) -1 -> 0; addition +1 for noninclusivity
                    self.max_furniture_style
                    + 1 + 1,  # Style (+1 since -1 is used for empty slot) -1 -> 0; addition +1 for noninclusivity
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
            (np.any(self.current_furniture_number >= self.max_furniture_per_room)) or 
            (np.any(self.furniture_observation[self.current_furniture_number] == -1))
        )

        if (
            self.terminated or self.truncated
        ):
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
        self.furniture_observation[self.current_furniture_number][furniture_x_index_in_observation_space] = furniture_x
        self.furniture_observation[self.current_furniture_number][furniture_y_index_in_observation_space] = furniture_y
        self.furniture_observation[self.current_furniture_number][furniture_theta_index_in_observation_space] = furniture_theta
        # assert 0 <= furniture_x < self.max_room_width//MIN_FURNITURE_STEP + 1, f"furniture x:{furniture_x} is larger than the max: {self.max_room_width//MIN_FURNITURE_STEP + 1}"
        # assert 0 <= furniture_y < self.max_room_length//MIN_FURNITURE_STEP + 1, f"furniture y:{furniture_y} is larger than the max: {self.max_room_length//MIN_FURNITURE_STEP + 1}"
        # assert 0 <= furniture_theta < 360//MIN_THETA_STEP + 1, f"furniture theta:{furniture_theta} is larger than the max: {360//MIN_THETA_STEP + 1}"
        
        self.current_furniture_number += 1
        room_observations_dictionary = {
            "Width": self.room_observation[0] + MIN_ROOM_WIDTH,
            "Length": self.room_observation[1] + MIN_ROOM_LENGTH,
            "Height": self.room_observation[2] + MIN_ROOM_HEIGHT,
            "Style": self.room_observation[
                3
            ],  # no -1 since the room style is always defined
            "Type": self.room_observation[4]
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
        arrangement_reward = reward(arrangement)
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

    def reset(self, seed=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.truncated = False
        room_width = randint(0, self.max_room_width - MIN_ROOM_WIDTH)
        room_length = randint(0, self.max_room_length - MIN_ROOM_LENGTH)
        room_height = randint(0, self.max_room_height - MIN_ROOM_HEIGHT)
        arrangements = self.training_data_json["Selections"]
        current_selection = random.choice(arrangements)
        style = current_selection["Room"]["Style"]
        type = current_selection["Room"]["Type"]
        furnitures = current_selection["Furniture"]
        # initializing positions to the furnitures
        furnitures = [{**furniture, "X": 0, "Y": 0, "Theta": 0} for furniture in furnitures]
        # assert len(furnitures) == self.max_furniture_per_room, f"the size of the selected furnitures: {furnitures} does not match the max furniture per room: {self.max_furniture_per_room}"
        # assert furnitures[0]["ID"] != -1, f"the first value in the selected furnitures: {furnitures} is -1"

        for f in furnitures:
            break
            # assert 0 <= f["ID"] < self.max_furniture_id + 1 + 1, f"furniture id: {f['ID']} is larger than the max: {self.max_furniture_id + 1 + 1}"
            # assert 0 <= f["Style"] < self.max_furniture_style + 1 + 1, f"furniture style: {f['Style']} is larger than the max: {self.max_furniture_style + 1 + 1}"
            # assert 0 <= f["Width"] < self.max_furniture_width + 1, f"furniture width: {f['Width']} is larger than the max: {self.max_furniture_width}"
            # assert 0 <= f["Depth"] < self.max_furniture_depth + 1, f"furniture depth: {f['Depth']} is larger than the max: {self.max_furniture_depth}"
            # assert 0 <= f["Height"] < self.max_furniture_height + 1, f"furniture height: {f['Height']} is larger than the max: {self.max_furniture_height}"


        
        self.current_furniture_number = 0
        self.reward = 0
        self.room_observation = np.array(
            [room_width, room_length, room_height, style, type]
        )
        self.furniture_observation = np.array(
            [list(furniture.values()) for furniture in furnitures], dtype=int
        )
        info = {}
        # print(f"INITIAL OBSERVATION: {np.concatenate((self.room_observation, self.furniture_observation.flatten()))}")
        return (
            np.concatenate(
                (self.room_observation, self.furniture_observation.flatten())
            ),
            info,
        )

if __name__ == "__main__":
    # file check
    with open("selections.json", "r") as f:
        embeddings = json.load(f)
    # assert embeddings is not None, "Embeddings failed to load"

    env = ArrangementEnv("selections.json", 144, 144, 120)
    check_env(env)
    models_path = f"models/{int(time.time())}/"
    log_path = f"logs/{int(time.time())}/"
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
    TIMESTEPS = 10000
    episodes = 0
    while episodes < 1000000:
        episodes += 1
        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            tb_log_name=f"PPO",
        )
        model.save(f"{models_path}/{TIMESTEPS*episodes}")
