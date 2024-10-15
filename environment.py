import json
import os
import random
from random import randint

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import observations

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
                    * np.ones(3),  # Continuous Room properties (Width, Length, Height)
                    np.zeros(2),  # Categorical Room properties (Style, Type)
                    -1
                    * np.ones(
                        8 * self.max_furniture_per_room
                    ),  # Furniture properties (8 properties per furniture)
                ]
            ),
            high=np.concatenate(
                [
                    np.ones(3),  # Continuous Room properties (Width, Length, Height)
                    np.array(
                        [self.max_room_style, self.max_room_type]
                    ),  # Max for Categorical Room properties (Style, Type)
                    np.tile(
                        np.concatenate(
                            [
                                np.array(
                                    [self.max_furniture_id, self.max_furniture_style]
                                ),  # Categorical Furniture properties (ID, Style)
                                np.ones(
                                    6
                                ),  # Continuous Furniture properties (Width, Height, Depth, X, Y, Theta)
                            ]
                        ),
                        (
                            self.max_furniture_per_room,
                            1,
                        ),  # Tile to match the number of furniture items
                    ).flatten(),  # Flatten the tiled array
                ]
            ),
            dtype=np.float32,
        )

    def step(self, action):
        self.terminated = bool(action[-1] > 0)
        self.truncated = False

        if self.terminated:
            info = {}
            return (
                np.concatenate(
                    (
                        self.room_observation.get_normalized_array(),
                        self.furniture_observation.get_normalized_array.flatten(),
                    )
                ),
                self.reward,
                self.terminated,
                self.truncated,
                info,
            )

        arrangement_reward = reward(self.room_observation, self.furniture_observation)
        self.reward = arrangement_reward  # Since all furniture is placed at once, the reward is calculated based on the final placement

        info = {}

        return (
            np.concatenate(
                (
                    self.room_observation.get_normalized_array(),
                    self.furniture_observation.get_normalized_array().flatten(),
                )
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

        self.current_furniture_number = 0
        self.reward = 0
        self.room_observation = observations.RoomObservations(
            np.array(
                [room_width, room_length, room_height, style, type], dtype=np.float32
            ),
            self.max_room_width,
            self.max_room_length,
            self.max_room_height,
        )
        self.furniture_observation = observations.FurnitureObservations(
            np.array(
                [
                    [
                        furniture["ID"],
                        furniture["Style"],
                        furniture["Width"],
                        furniture["Depth"],
                        furniture["Height"],
                        0,
                        0,
                        0,
                    ]
                    for furniture in furnitures
                ]
            ).flatten(),
            self.max_furniture_id,
            self.max_furniture_style,
            self.max_furniture_width,
            self.max_furniture_height,
            self.max_furniture_depth,
            self.max_room_width,
            self.max_room_length,
        )
        info = {}
        return np.concatenate(
            (
                self.room_observation.get_normalized_array(),
                self.furniture_observation.get_normalized_array().flatten(),
            )
        ), info


def reward(room_observation, furniture_observation):
    room_width = room_observation.width
    room_length = room_observation.width

    total_reward = 0

    def get_rectangle_corners(furniture):
        cx, cy = furniture.x, furniture.y
        w, h = furniture.width / 2, furniture.depth / 2
        theta = np.radians(furniture.theta)

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

    for i, furniture in enumerate(furniture_observation.furnitures):
        # Check if the furniture is outside the room
        if is_outside_room(furniture):
            total_reward -= 1
        else:
            non_colliding = True
            # Check for collisions with all other furniture
            for j, other_furniture in enumerate(furniture_observation.furnitures):
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

    model_path = "models/ppo_arrangement"
    log_path = "logs/ppo_arrangement"
    checkpoint_path = "checkpoints/ppo_arrangement"

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    # Load existing model if available
    try:
        model = PPO.load(f"{checkpoint_path}/last_checkpoint", env=env)
        print("Loaded model from checkpoint.")
    except FileNotFoundError:
        model = PPO(
            "MlpPolicy", env, ent_coef=0.01, verbose=1, tensorboard_log=log_path
        )
        print("Training a new model.")

    TIMESTEPS = 10000
    save_interval = 100000  # Save every 100,000 timesteps
    total_timesteps = 1000000

    for episode in range(1, total_timesteps // TIMESTEPS + 1):
        model.learn(
            total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO"
        )

        # Save the model at checkpoints
        if episode * TIMESTEPS % save_interval == 0:
            model.save(f"{model_path}/ppo_arrangement_{episode * TIMESTEPS}")
            model.save(
                f"{checkpoint_path}/last_checkpoint"
            )  # Always overwrite last checkpoint
            print(f"Model saved at {episode * TIMESTEPS} timesteps.")
