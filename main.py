import multiprocessing as mp
from evolution import run_training, watch_telemetry
import warnings
warnings.filterwarnings("ignore", message=".*avx2 capable.*")

def main():
    print("Starting Deep RL Pipeline...")

    # 1. Spawn the PyTorch training loop in a separate background process
    training_process = mp.Process(target=run_training)
    training_process.start()

    try:
        # 2. Run the Pygame visualizer in the main thread
        print("Launching live telemetry viewer...")
        watch_telemetry()

    except KeyboardInterrupt:
        print("\nPipeline stopped by user.")

    finally:
        # 3. Graceful shutdown: If the Pygame window is closed, kill the trainer
        print("Shutting down background training process...")
        training_process.terminate()
        training_process.join()
        print("Shutdown complete. Goodbye!")


if __name__ == '__main__':
    # Required for Windows multiprocessing safety
    main()