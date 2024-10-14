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

MIN_ROOM_WIDTH = 100
MIN_ROOM_LENGTH = 100
MIN_ROOM_HEIGHT = 100
MIN_FURNITURE_NUMBER = 2


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
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([1, self.max_room_width, self.max_room_length, 360]),
            dtype=np.float32,
        )

        # Observation space
        self.observation_space = spaces.Box(
            low=np.concatenate(
                [
                    np.array([MIN_ROOM_WIDTH, MIN_ROOM_LENGTH, MIN_ROOM_HEIGHT, 0, 0]),
                    np.tile(
                        np.array([0, 0, 0, 0, 0, 0, 0, 0]), self.max_furniture_per_room
                    ),
                ]
            ),
            high=np.concatenate(
                [
                    np.array(
                        [
                            self.max_room_width,
                            self.max_room_length,
                            self.max_room_height,
                            self.max_room_style,
                            self.max_room_type,
                        ]
                    ),
                    np.tile(
                        np.array(
                            [
                                self.max_furniture_id + 1,
                                self.max_furniture_style + 1,
                                self.max_furniture_width,
                                self.max_furniture_depth,
                                self.max_furniture_height,
                                self.max_room_width,
                                self.max_room_length,
                                360,
                            ]
                        ),
                        self.max_furniture_per_room,
                    ),
                ]
            ),
            dtype=np.float32,
        )

    def step(self, action):
        self.terminated = bool(action[0] == 1)
        self.truncated = self.current_furniture_number >= self.max_furniture_per_room

        if self.terminated or self.truncated:
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

        furniture_x = action[1]
        furniture_y = action[2]
        furniture_theta = action[3]
        self.furniture_observation[self.current_furniture_number][5] = furniture_x
        self.furniture_observation[self.current_furniture_number][6] = furniture_y
        self.furniture_observation[self.current_furniture_number][7] = furniture_theta

        arrangement = {
            "Room": {
                "Width": self.room_observation[0],
                "Length": self.room_observation[1],
                "Height": self.room_observation[2],
                "Style": self.room_observation[3],
                "Type": self.room_observation[4],
            },
            "Furniture": [
                {
                    "ID": f[0],
                    "Style": f[1],
                    "Width": f[2],
                    "Depth": f[3],
                    "Height": f[4],
                    "X": f[5],
                    "Y": f[6],
                    "Theta": f[7],
                }
                for f in self.furniture_observation
            ],
        }

        arrangement_reward = reward(arrangement, self.current_furniture_number)
        self.current_furniture_number += 1
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
        room_width = np.random.uniform(MIN_ROOM_WIDTH, self.max_room_width)
        room_length = np.random.uniform(MIN_ROOM_LENGTH, self.max_room_length)
        room_height = np.random.uniform(MIN_ROOM_HEIGHT, self.max_room_height)

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

        furnitures = [
            {
                "ID": furniture["ID"] + 1,
                "Style": furniture["Style"] + 1,
                "Width": furniture["Width"],
                "Depth": furniture["Depth"],
                "Height": furniture["Height"],
                "X": 0,
                "Y": 0,
                "Theta": 0,
            }
            for furniture in furnitures
        ]

        self.current_furniture_number = 0
        self.reward = 0
        self.room_observation = np.array(
            [room_width, room_length, room_height, style, type], dtype=np.float32
        )
        self.furniture_observation = np.array(
            [list(furniture.values()) for furniture in furnitures], dtype=np.float32
        )
        info = {}
        return np.concatenate(
            (self.room_observation, self.furniture_observation.flatten())
        ), info


def reward(arrangement, current_furniture_index):
    current_furniture = arrangement["Furniture"][current_furniture_index]
    room_width = arrangement["Room"]["Width"]
    room_length = arrangement["Room"]["Length"]

    def get_rectangle_corners(furniture):
        cx, cy = furniture["X"], furniture["Y"]
        w, h = furniture["Width"] / 2, furniture["Depth"] / 2
        theta = np.radians(furniture["Theta"])

        corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]])
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        rotated_corners = np.dot(corners, rotation_matrix) + [cx, cy]
        return rotated_corners

    def is_outside_room(furniture, room_width, room_length):
        corners = get_rectangle_corners(furniture)
        for corner in corners:
            if not (0 <= corner[0] <= room_width and 0 <= corner[1] <= room_length):
                return True
        return False

    def is_collision(furniture1, furniture2):
        rect1_corners = get_rectangle_corners(furniture1)
        rect2_corners = get_rectangle_corners(furniture2)

        for rect_corners in [rect1_corners, rect2_corners]:
            for i in range(4):
                edge = rect_corners[i] - rect_corners[i - 1]
                axis = np.array([-edge[1], edge[0]])

                proj1 = np.dot(rect1_corners, axis)
                proj2 = np.dot(rect2_corners, axis)

                if max(proj1) < min(proj2) or max(proj2) < min(proj1):
                    return False

        return True

    if is_outside_room(current_furniture, room_width, room_length):
        return -1

    for i, other_furniture in enumerate(arrangement["Furniture"]):
        if i != current_furniture_index and is_collision(
            current_furniture, other_furniture
        ):
            return -1

    return 1


if __name__ == "__main__":
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
            total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO"
        )
        model.save(f"{models_path}/{TIMESTEPS*episodes}")
