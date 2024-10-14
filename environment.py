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

        # Action space: [-1, 1] for X, Y, Theta, plus Terminate flag
        self.action_space = spaces.Box(
            low=-1 * np.ones(3 * self.max_furniture_per_room + 1),
            high=1 * np.ones(3 * self.max_furniture_per_room + 1),
            dtype=np.float32,
        )

        # Observation space: Normalize everything to [-1, 1]
        self.observation_space = spaces.Box(
            low=np.concatenate(
                [
                    -1
                    * np.ones(
                        5
                    ),  # Room properties (Width, Length, Height, Style, Type)
                    -1
                    * np.ones(8 * self.max_furniture_per_room),  # Furniture properties
                ]
            ),
            high=np.ones(5 + 8 * self.max_furniture_per_room),
            dtype=np.float32,
        )

    def step(self, action):
        self.terminated = bool(
            action[-1] > 0
        )  # Termination flag is now normalized, use >0 for terminate
        self.truncated = False

        if self.terminated:
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

        # Rescale actions from normalized [-1, 1] range back to the original range
        for i in range(self.max_furniture_per_room):
            furniture_x = (
                (action[3 * i] + 1) / 2 * self.max_room_width
            )  # Rescale to [0, max_room_width]
            furniture_y = (
                (action[3 * i + 1] + 1) / 2 * self.max_room_length
            )  # Rescale to [0, max_room_length]
            furniture_theta = (action[3 * i + 2] + 1) / 2 * 360  # Rescale to [0, 360]

            self.furniture_observation[i][5] = furniture_x
            self.furniture_observation[i][6] = furniture_y
            self.furniture_observation[i][7] = furniture_theta

        # Evaluate the reward based on all furniture at once
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

        arrangement_reward = reward(arrangement)
        self.reward = arrangement_reward  # Since all furniture is placed at once, the reward is calculated based on the final placement

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
        self.room_observation = self._normalize_room_observation(
            np.array(
                [room_width, room_length, room_height, style, type], dtype=np.float32
            )
        )
        self.furniture_observation = self._normalize_furniture_observation(
            np.array(
                [list(furniture.values()) for furniture in furnitures], dtype=np.float32
            )
        )
        info = {}
        print(
            np.concatenate(
                ((self.room_observation, self.furniture_observation.flatten()))
            )
        )
        return np.concatenate(
            (self.room_observation, self.furniture_observation.flatten())
        ), info

    def _concatenate(self, value):
        if value < -1:
            return -1
        if value > 1:
            return 1
        return value

    def _normalize_room_observation(self, room_observation):
        # Normalize room properties to [-1, 1]
        room_observation[0] = self._concatenate(
            (room_observation[0] - MIN_ROOM_WIDTH)
            / (self.max_room_width - MIN_ROOM_WIDTH)
            * 2
            - 1
        )
        room_observation[1] = self._concatenate(
            (room_observation[1] - MIN_ROOM_LENGTH)
            / (self.max_room_length - MIN_ROOM_LENGTH)
            * 2
            - 1
        )
        room_observation[2] = self._concatenate(
            (room_observation[2] - MIN_ROOM_HEIGHT)
            / (self.max_room_height - MIN_ROOM_HEIGHT)
            * 2
            - 1
        )
        room_observation[3] = self._concatenate(
            (room_observation[3] / self.max_room_style) * 2 - 1
        )
        room_observation[4] = self._concatenate(
            (room_observation[4] / self.max_room_type) * 2 - 1
        )
        return room_observation

    def _normalize_furniture_observation(self, furniture_observation):
        # Normalize furniture properties to [-1, 1]
        for furniture in furniture_observation:
            furniture[0] = self._concatenate(
                (furniture[0] - 1) / (self.max_furniture_id + 1) * 2 - 1
            )  # ID
            furniture[1] = self._concatenate(
                (furniture[1] - 1) / (self.max_furniture_style + 1) * 2 - 1
            )  # Style
            furniture[2] = self._concatenate(
                (furniture[2] / self.max_furniture_width) * 2 - 1
            )  # Width
            furniture[3] = self._concatenate(
                (furniture[3] / self.max_furniture_depth) * 2 - 1
            )  # Depth
            furniture[4] = self._concatenate(
                (furniture[4] / self.max_furniture_height) * 2 - 1
            )  # Height
            furniture[5] = self._concatenate(
                (furniture[5] / self.max_room_width) * 2 - 1
            )  # X position
            furniture[6] = self._concatenate(
                (furniture[6] / self.max_room_length) * 2 - 1
            )  # Y position
            furniture[7] = self._concatenate((furniture[7] / 360) * 2 - 1)  # Theta
        return furniture_observation


def reward(arrangement):
    room_width = arrangement["Room"]["Width"]
    room_length = arrangement["Room"]["Length"]
    furniture_list = arrangement["Furniture"]

    total_reward = 0

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

    def is_outside_room(furniture):
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

    for i, furniture in enumerate(furniture_list):
        # Check if the furniture is outside the room
        if is_outside_room(furniture):
            total_reward -= 1
        else:
            non_colliding = True
            # Check for collisions with all other furniture
            for j, other_furniture in enumerate(furniture_list):
                if i != j and is_collision(furniture, other_furniture):
                    non_colliding = False
                    total_reward -= 1  # Penalize for each collision
                    break
            if non_colliding:
                total_reward += 1  # Reward for no collisions

    return total_reward


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
