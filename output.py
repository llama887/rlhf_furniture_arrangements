import json
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from environment import ArrangementEnv


def generate_arrangement_from_model(
    selections_directory,
    model_path,
    max_room_width,
    max_room_length,
    max_room_height,
    specific_selection=None,
):
    # Initialize the custom environment
    env = ArrangementEnv(
        selections_directory, max_room_width, max_room_length, max_room_height
    )

    # Load the trained model
    model = PPO.load(model_path)

    # Reset the environment with the specific selection if provided
    obs, _ = env.reset(selection_file=specific_selection)

    # Step through the environment until termination
    done = False
    cumulative_reward = 0  # Track cumulative reward over the episode
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward  # Accumulate reward
        done = terminated or truncated

    # Unscale room observations before outputting
    room_observation = env.room_observation.get_unnormalized_array()
    room_observations_dictionary = {
        "Width": int(room_observation[0]),
        "Length": int(room_observation[1]),
        "Height": int(room_observation[2]),
        "Style": int(room_observation[3]),  # Unscale Room Style
        "Type": int(room_observation[4]),  # Unscale Room Type
    }

    furniture_observations = env.furniture_observation.furnitures
    furniture_observations_dictionary = []
    for f in furniture_observations:
        unnormalized_furniture = f.get_unnormalized_array()
        furniture_observations_dictionary.append(
            {
                "ID": int(unnormalized_furniture[0]),  # Unscale Furniture ID
                "Style": int(unnormalized_furniture[1]),  # Unscale Furniture Style
                "Width": int(unnormalized_furniture[2]),  # Unscale Furniture Width
                "Depth": int(unnormalized_furniture[3]),  # Unscale Furniture Depth
                "Height": int(unnormalized_furniture[4]),  # Unscale Furniture Height
                "X": int(unnormalized_furniture[5]),  # Unscale X Position
                "Y": int(unnormalized_furniture[6]),  # Unscale Y Position
                "Theta": int(unnormalized_furniture[7]),  # Unscale Theta
            }
        )

    arrangement = {
        "Room": room_observations_dictionary,
        "Furniture": furniture_observations_dictionary,
    }

    # Output the arrangement as JSON
    output_path = "final_arrangement.json"
    print(arrangement)
    with open(output_path, "w") as f:
        json.dump(arrangement, f, indent=4)

    print(f"Arrangement saved to {output_path}")
    return arrangement, cumulative_reward


# Helper function to get the four corners of a rectangle after rotation
def get_rectangle_corners(f):
    # Center of the rectangle
    cx, cy = f["X"], f["Y"]
    w, h = f["Width"] / 2, f["Depth"] / 2
    theta = np.radians(f["Theta"])

    # The four corners relative to the center, before rotation
    corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]])

    # Rotation matrix
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    # Rotate and translate corners
    rotated_corners = np.dot(corners, rotation_matrix) + [cx, cy]

    return rotated_corners


# Function to check if two rotated rectangles overlap using the Separating Axis Theorem
def check_overlap(f1, f2):
    # Get the corners of the two rectangles
    rect1_corners = get_rectangle_corners(f1)
    rect2_corners = get_rectangle_corners(f2)

    # Check for separation on both rectangles' axes
    for rect_corners in [rect1_corners, rect2_corners]:
        for i in range(4):
            # Get the edge vector
            edge = rect_corners[i] - rect_corners[i - 1]
            # Get the perpendicular vector (axis to project onto)
            axis = np.array([-edge[1], edge[0]])

            # Project both rectangles onto the axis
            proj1 = np.dot(rect1_corners, axis)
            proj2 = np.dot(rect2_corners, axis)

            # Check for overlap in the projections
            if max(proj1) < min(proj2) or max(proj2) < min(proj1):
                return False  # Found a separating axis

    return True  # No separating axis found, the rectangles must overlap


# Visualization function for furniture arrangement with Theta (rotation)
def visualize_furniture(arrangement, reward):
    fig, ax = plt.subplots()

    # Room dimensions
    room_width = arrangement["Room"]["Width"]
    room_length = arrangement["Room"]["Length"]

    # Draw the room
    ax.add_patch(
        patches.Rectangle(
            (0, 0), room_width, room_length, fill=None, edgecolor="blue", lw=2
        )
    )

    furniture = arrangement["Furniture"]
    rects = []

    # Add furniture to plot and apply rotation using Theta
    for f in furniture:
        # Get the rectangle's corners (rotated)
        rect_corners = get_rectangle_corners(f)

        # Plot the furniture by creating a Polygon from its corners
        polygon = patches.Polygon(
            rect_corners, closed=True, edgecolor="green", facecolor="green", alpha=0.5
        )
        ax.add_patch(polygon)
        rects.append((f, polygon))

    # Check for overlaps and highlight them
    collision_detected = False
    for i, (f1, polygon1) in enumerate(rects):
        for j, (f2, polygon2) in enumerate(rects):
            if i != j and check_overlap(f1, f2):
                # Change the polygon color to red if there is a collision
                polygon1.set_edgecolor("red")
                polygon2.set_edgecolor("red")
                polygon1.set_facecolor("red")
                polygon2.set_facecolor("red")
                collision_detected = True

    # Set axis limits and labels
    ax.set_xlim(0, room_width)
    ax.set_ylim(0, room_length)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Furniture Arrangement with Reward {reward}")

    # Create custom legend to show meaning of colors
    furniture_patch = patches.Patch(
        edgecolor="green", facecolor="green", label="Furniture", alpha=0.5
    )
    if collision_detected:
        collision_patch = patches.Patch(
            edgecolor="red", facecolor="red", label="Collision", alpha=0.5
        )
        ax.legend(handles=[furniture_patch, collision_patch])
    else:
        ax.legend(handles=[furniture_patch])

    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


if __name__ == "__main__":
    selections_directory = "./selections"
    max_indices_path = os.path.join(selections_directory, "max_indices.json")
    model_path = "models/ppo_arrangement/ppo_arrangement_300000.zip"  # Replace with your trained model's path
    specific_selection = "selection_10.json"  # Use the specific selection file you want
    max_room_width = 144
    max_room_length = 144
    max_room_height = 120

    arrangement, reward = generate_arrangement_from_model(
        selections_directory,
        model_path,
        max_room_width,
        max_room_length,
        max_room_height,
        specific_selection,
    )
    print(json.dumps(arrangement, indent=4))
    visualize_furniture(arrangement, reward)
