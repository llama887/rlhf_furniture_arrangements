import json
import os
from stable_baselines3 import PPO
from environment import ArrangementEnv, MIN_ROOM_WIDTH, MIN_ROOM_LENGTH, MIN_ROOM_HEIGHT, MIN_FURNITURE_STEP, MIN_THETA_STEP

def generate_arrangement_from_model(selections_directory, model_path, max_room_width, max_room_length, max_room_height, specific_selection=None):
    
    # Initialize the custom environment
    env = ArrangementEnv(selections_directory, max_room_width, max_room_length, max_room_height)
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Reset the environment with the specific selection if provided
    obs, _ = env.reset(selection_file=specific_selection)

    # Step through the environment until termination
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Collect the final arrangement
    room_observations_dictionary = {
        "Width": int(env.room_observation[0] + MIN_ROOM_WIDTH),
        "Length": int(env.room_observation[1] + MIN_ROOM_LENGTH),
        "Height": int(env.room_observation[2] + MIN_ROOM_HEIGHT),
        "Style": int(env.room_observation[3]),  # no -1 since the room style is always defined
        "Type": int(env.room_observation[4])
    }
    furniture_observations_dictionary = []
    for f in env.furniture_observation:
        furniture_observations_dictionary.append(
            {
                "ID": int(f[0] - 1),  # -1 since observation space back to embedding space is -1
                "Style": int(f[1] - 1),  # -1 since observation space back to embedding space is -1
                "Width": int(f[2]),
                "Depth": int(f[3]),
                "Height": int(f[4]),
                "X": int(f[5] * MIN_FURNITURE_STEP),
                "Y": int(f[6] * MIN_FURNITURE_STEP),
                "Theta": int(f[7] * MIN_THETA_STEP),
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
    return arrangement

if __name__ == "__main__":
    selections_directory = "./selections"
    max_indices_path = os.path.join(selections_directory, "max_indices.json")
    model_path = "./models/1724614065/160000.zip"  # Replace with your trained model's path
    specific_selection = "selection_5.json"  # Use the specific selection file you want
    max_room_width = 144
    max_room_length = 144
    max_room_height = 120

    arrangement = generate_arrangement_from_model(
        selections_directory, model_path, max_room_width, max_room_length, max_room_height, specific_selection
    )
    print(json.dumps(arrangement, indent=4))
