import json
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import warnings
warnings.filterwarnings("ignore", message=".*avx2 capable.*")
from .pool_env import PoolEnv
from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining starts at 1.0 and goes to 0.0
        return progress_remaining * initial_value
    return func

class LiveTelemetryCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_broadcast = 0

    def _on_step(self) -> bool:
        # Broadcast a shot roughly every 500 timesteps
        if self.num_timesteps - self.last_broadcast >= 500:
            try:
                # Safely extract the data from CPU Core #0
                shot_data = self.training_env.get_attr('latest_shot_data', indices=[0])[0]
                if shot_data:
                    with open("live_shot.json", "w") as f:
                        json.dump(shot_data, f)
            except Exception:
                pass
            self.last_broadcast = self.num_timesteps
        return True


def run_training():
    print("Initializing background environments...")

    # Define how many CPU cores you want to dedicate to physics simulation
    total_cores = os.cpu_count() or 8
    num_cpu = max(1, total_cores - 2)

    # make_vec_env automatically builds the multiprocessing infrastructure
    env = make_vec_env(
        PoolEnv,
        n_envs=num_cpu,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"render_mode": "none"}
    )
    custom_lr = linear_schedule(0.0003)

    model = PPO("MlpPolicy", env, learning_rate=custom_lr, verbose=1, tensorboard_log="./ppo_pool_tensorboard/")

    print(f"Background training started on {num_cpu} cores...")

    # Create the checkpoint auto-saver
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,  # Saves every 100,000 steps
        save_path="./ppo_checkpoints/",
        name_prefix="pool_model"
    )

    # Combine your live viewer callback with the auto-saver
    callback_list = CallbackList([LiveTelemetryCallback(), checkpoint_callback])

    try:
        model.learn(total_timesteps=16_000_000, callback=callback_list)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Final save just in case
        model.save("ppo_pool_model")
        print("Final model saved successfully.")


if __name__ == '__main__':
    run_training()