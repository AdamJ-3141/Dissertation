import json
import time
from stable_baselines3 import PPO
from pool_env import PoolEnv


def main():
    print("Initializing Pool Environment...")
    env = PoolEnv(render_mode="none")

    print("Loading trained AI brain...")
    # Make sure this matches your final save file or a specific checkpoint
    model = PPO.load("pool_model_1600000_steps")

    print("AI loaded! Broadcasting shots to watch_live.py...")
    print("Make sure watch_live.py is running in another terminal!")

    shot_count = 0

    while True:
        obs, info = env.reset()

        # Ask the AI for its chosen action based on the current table layout
        # deterministic=True disables random exploration noise so it plays its absolute best
        action, _states = model.predict(obs, deterministic=True)

        # Package the pre-shot data for the visualizer
        shot_data = {
            "positions": env.sim.positions.tolist(),
            "in_play": env.sim.in_play.tolist(),
            "colours": env.sim.colours.tolist(),
            "action": action.tolist()
        }

        # Broadcast it to watch_live.py
        with open("live_shot.json", "w") as f:
            json.dump(shot_data, f)

        shot_count += 1
        print(f"Broadcasted Shot #{shot_count}")

        # Actually resolve the physics so the environment can calculate the next layout
        obs, reward, terminated, truncated, info = env.step(action)

        # Wait for the visualizer to finish its 2-second pause and animation
        time.sleep(4.0)


if __name__ == '__main__':
    main()