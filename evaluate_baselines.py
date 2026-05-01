import json
import time
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from pool_simulation.physics.engine import Simulation
from match import Match
from agent import Agent, RandomAgent, GreedyAgent, Human


def load_weights(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def play_evaluation_frame(p1_type, p1_weights, p2_type, p2_weights, frame_idx):
    """Worker function to simulate a single frame between two specified agents."""
    sim = Simulation(n_obj_balls=15)
    # custom_setup=False ensures randomised break state and preserves Open Table
    match = Match(sim, play_break=False, custom_setup=False)

    def create_agent(a_type, a_weights, sim_instance):
        if a_type == "trained" or a_type == "default":
            return Agent(sim_instance, a_weights)
        elif a_type == "greedy":
            return GreedyAgent(sim_instance)
        elif a_type == "random":
            return RandomAgent(sim_instance)

    agent_1 = create_agent(p1_type, p1_weights, sim)
    agent_2 = create_agent(p2_type, p2_weights, sim)

    # Alternate the break[cite: 9]
    match.turn = frame_idx % 2
    current_turn = 0
    turn_limit = 100

    while match.turn_state != 3 and current_turn < turn_limit:
        active_agent = agent_1 if match.turn == 0 else agent_2
        match.play_turn(active_agent)
        current_turn += 1

    # Return 0 if Agent 1 won, 1 if Agent 2 won, or -1 for a timeout/draw
    return match.winner if match.winner is not None else -1


def run_ai_evaluations(trained_weights, default_weights, num_frames=30, cores=10):
    matchups = [
        ("Random Baseline", "trained", trained_weights, "random", None),
        ("Greedy Baseline", "trained", trained_weights, "greedy", None),
        ("Default Weights Baseline", "trained", trained_weights, "default", default_weights)
    ]

    for opponent_name, p1_type, p1_weights, p2_type, p2_weights in matchups:
        print(f"\n--- Starting {num_frames} frames: Evolved Agent vs {opponent_name} ---")
        p1_wins = 0
        p2_wins = 0
        draws = 0

        # Distribute the frames across the available cores
        with ProcessPool(max_workers=cores) as pool:
            futures = {}
            for frame_idx in range(num_frames):
                future = pool.schedule(
                    play_evaluation_frame,
                    args=[p1_type, p1_weights, p2_type, p2_weights, frame_idx],
                    timeout=900  # 15 minute OS-level timeout
                )
                futures[future] = frame_idx

            for future in futures:
                frame_idx = futures[future]
                try:
                    winner = future.result()
                    if winner == 0:
                        print("Evolved Agent wins the frame.")
                        p1_wins += 1
                    elif winner == 1:
                        print("Other Agent wins the frame")
                        p2_wins += 1
                    else:
                        print("Draw.")
                        draws += 1
                except TimeoutError:
                    print(f"Frame {frame_idx} timed out!")
                    draws += 1
                except Exception as e:
                    print(f"Frame {frame_idx} crashed: {e}")
                    draws += 1

        print(f"Result - Evolved Agent: {p1_wins} | {opponent_name}: {p2_wins} | Draws/Timeouts: {draws}")


def run_human_matches(trained_weights, num_frames=10):
    print(f"\n--- Starting {num_frames} frames: Evolved Agent vs Human ---")

    # Delayed import to avoid headless server issues if running purely AI
    from pool_simulation.render import Renderer

    p1_wins = 0
    p2_wins = 0

    for frame_idx in range(num_frames):
        sim = Simulation(n_obj_balls=15)
        renderer = Renderer(sim)
        match = Match(sim, play_break=False, custom_setup=False)

        agent = Agent(sim, trained_weights)
        human = Human(sim, renderer)

        # Alternate breaks
        match.turn = frame_idx % 2
        print(f"\nFrame {frame_idx + 1} starting. {'Agent' if match.turn == 0 else 'Human'} to break.")

        while match.turn_state != 3:
            if match.turn == 0:
                print("Agent is thinking...")
                match.play_turn(agent, frame_callback=lambda s: renderer.render(flip=True))
            else:
                print("Your turn!")
                match.play_turn(human, frame_callback=lambda s: renderer.render(flip=True))

        if match.winner == 0:
            print("Agent wins the frame.")
            p1_wins += 1
        else:
            print("Human wins the frame.")
            p2_wins += 1

    print(f"Final Human Series Result - Evolved Agent: {p1_wins} | Human: {p2_wins}")


if __name__ == '__main__':
    # Load the weights
    try:
        best_weights = load_weights('best_gen_19.json')
        default_weights = load_weights('planner/defaults.json')
    except FileNotFoundError as e:
        print(f"Error loading weight files: {e}")
        exit()

    # Run the AI vs AI benchmarks on 10 cores
    run_ai_evaluations(best_weights, default_weights, num_frames=30, cores=10)

    # Optional: Run the Human matches sequentially
    user_input = input("\nAI benchmarks complete. Start Human matches? (y/n): ")
    if user_input.lower() == 'y':
        run_human_matches(best_weights, num_frames=10)