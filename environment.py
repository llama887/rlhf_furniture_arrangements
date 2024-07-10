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
        embeddings_json,
        max_room_width,
        max_room_length,
        max_room_height,
        max_furniture_number,
    ):
        super(ArrangementEnv, self).__init__()
        with open(embeddings_json, "r") as f:
            self.furniture_embeddings = json.load(f)
        self.max_id_index = self.furniture_embeddings[0]["max_id_index"]
        self.max_style_index = self.furniture_embeddings[2]["max_style_index"]
        self.max_room_width = max_room_width
        self.max_room_length = max_room_length
        self.max_room_height = max_room_height
        self.max_furniture_number = max_furniture_number
        self.max_furniture_width = self.furniture_embeddings[4]["max_width"]
        self.max_furniture_depth = self.furniture_embeddings[4]["max_depth"]
        self.max_furniture_height = self.furniture_embeddings[4]["max_height"]
        self.room_observation = None
        self.furniture_observation = None
        self.current_furniture_number = 0
        self.reward = 0
        self.truncated = False
        self.terminated = False
        # Action space: [Terminate episode, Furniture ID, X Position, Y Position, Angle]
        self.action_space = spaces.MultiDiscrete(
            [
                2,  # Terminate episode (0: continue, 1: terminate)
                self.max_id_index,  # Furniture ID
                max_room_width // MIN_FURNITURE_STEP,  # X Position
                max_room_length // MIN_FURNITURE_STEP,  # Y Position
                360 // MIN_THETA_STEP,  # Angle
            ],
            dtype=int,
        )
        # Room properties: [Width, Length, Height, Style]
        # Each furniture item: [ID, Style, Width, Depth, Height, X, Y, Theta]
        self.observation_space = spaces.MultiDiscrete(
            [
                max_room_width - MIN_ROOM_WIDTH,  # Width
                max_room_length - MIN_ROOM_WIDTH,  # Length
                max_room_height - MIN_ROOM_WIDTH,  # Height
                self.max_style_index,  # Style (no +1 since the rooms style is always defined)
            ]
            + (
                [
                    self.max_id_index
                    + 1,  # ID (+1 since -1 is used for empty slot) -1 -> 0
                    self.max_style_index
                    + 1,  # Style (+1 since -1 is used for empty slot) -1 -> 0
                    self.max_furniture_width,  # Width
                    self.max_furniture_depth,  # Depth
                    self.max_furniture_height,  # Height
                    max_room_width // MIN_FURNITURE_STEP,  # X Position
                    max_room_length // MIN_FURNITURE_STEP,  # Y Position
                    360 // MIN_THETA_STEP,  # Angle
                ]
                * max_furniture_number  # Repeat for each furniture item
            ),
            dtype=int,
        )

    def step(self, action):
        self.terminated = bool(action[0] == 1)
        self.truncated = bool(
            self.current_furniture_number >= self.max_furniture_number
        )
        if (
            action[0] == 1
            or self.current_furniture_number >= self.max_furniture_number
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
        furniture_id = action[
            1
        ]  # no +1 since -1 is not legal in the action space
        furniture_style = self.furniture_embeddings[5][str(furniture_id)][
            "Style"
        ]  # no +1 since -1 is not legal in the action space
        furniture_width = self.furniture_embeddings[5][str(furniture_id)][
            "Width"
        ]
        furniture_depth = self.furniture_embeddings[5][str(furniture_id)][
            "Depth"
        ]
        furniture_height = self.furniture_embeddings[5][str(furniture_id)][
            "Height"
        ]
        furniture_x = action[2]
        furniture_y = action[3]
        furniture_theta = action[4]
        self.furniture_observation[self.current_furniture_number] = np.array(
            [
                furniture_id
                + 1,  # +1 converts from the action space to the observation space
                furniture_style
                + 1,  # +1 converts from the action space to the observation space
                furniture_width,
                furniture_depth,
                furniture_height,
                furniture_x,
                furniture_y,
                furniture_theta,
            ]
        )
        self.current_furniture_number += 1
        room_observations_dictionary = {
            "Width": self.room_observation[0] + MIN_ROOM_WIDTH,
            "Length": self.room_observation[1] + MIN_ROOM_LENGTH,
            "Height": self.room_observation[2] + MIN_ROOM_HEIGHT,
            "Style": self.room_observation[
                3
            ],  # no -1 since the room style is always defined
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
        self.reward += reward(arrangement)
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
        style = randint(1, self.max_style_index)
        self.current_furniture_number = 0
        self.reward = 0
        self.room_observation = np.array(
            [room_width, room_length, room_height, style]
        )
        self.furniture_observation = np.zeros(
            (self.max_furniture_number, 8), dtype=int
        )
        self.furniture_observation[:, :2] = 0
        info = {}
        return (
            np.concatenate(
                (self.room_observation, self.furniture_observation.flatten())
            ),
            info,
        )


if __name__ == "__main__":
    env = ArrangementEnv("embeddings.json", 144, 144, 120, 1)
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
