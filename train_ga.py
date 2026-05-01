import concurrent.futures
import json
import random
import copy
import os
import time
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from pool_simulation.physics.engine import Simulation
from match import Match
from agent import Agent

# Load base defaults to act as the genetic baseline
with open('planner/defaults.json', 'r') as f:
    BASE_WEIGHTS = json.load(f)


def generate_random_weights():
    """Mutates the baseline to create a unique individual."""
    new_weights = copy.deepcopy(BASE_WEIGHTS)
    for key in new_weights:
        # Shift each weight by a random amount between -30% and +30%
        shift = random.uniform(-0.3, 0.3)
        new_weights[key] *= (1.0 + shift)
    return new_weights


def simulate_matchup(gen_id, match_id, p1_idx, p2_idx, weights_p1, weights_p2):
    """
    Plays a Best-of-3 match between two agents on a single CPU process.
    First to 2 wins. Tracks total turns and elapsed execution time.
    """
    print(f"({time.strftime('%H:%M:%S', time.gmtime(time.time()))}) "
          f"Starting match ({match_id}) Agent {p1_idx} vs Agent {p2_idx}")
    p1_wins = 0
    p2_wins = 0
    frames_played = 0
    total_turns = 0

    pid = os.getpid()
    start_time = time.time()

    match_filename = f"logs/{gen_id}_Ag{p1_idx}_vs_Ag{p2_idx}.txt"
    with open(match_filename, 'w') as mf:
        mf.write(f"{time.strftime('%H:%M:%S', time.gmtime(time.time()))}\nMatch: Agent {p1_idx} vs Agent {p2_idx}\n")
    # Best-of-3: First to 2 wins
    while p1_wins < 2 and p2_wins < 2:
        frames_played += 1
        sim = Simulation(n_obj_balls=15)
        # Using custom_setup=False to ensure randomized break state and preserve Open Table
        match = Match(sim, play_break=False, custom_setup=False)

        agent_1 = Agent(sim, weights_p1)
        agent_2 = Agent(sim, weights_p2)

        # Alternate who breaks/starts
        match.turn = 0 if frames_played % 2 != 0 else 1

        turn_limit = 100
        current_turn = 0

        with open(match_filename, 'a') as mf:
            mf.write(f"Initial State:\nPositions: {sim.positions.tolist()}\n"
                     f"Colours: {sim.colours.tolist()}\nIn_Play: {sim.in_play.tolist()}\n")


        while match.turn_state != 3 and current_turn < turn_limit:

            active_agent = agent_1 if match.turn == 0 else agent_2

            with open(match_filename, 'a') as mf:
                mf.write(
                    f"{time.strftime('%H:%M:%S', time.gmtime(time.time()))} \n"
                    f"Turn {current_turn}: Agent {[p1_idx, p2_idx][match.turn]} to play.\n"
                    f"Positions: {sim.positions.tolist()}\nColours: {sim.colours.tolist()}\n"
                    f"In_Play: {sim.in_play.tolist()}\n"
                    f"TurnState: {match.turn_state}\n\n")

            res = match.play_turn(active_agent, frame_callback=None)
            with open(match_filename, 'a') as mf:
                mf.write(f"Result: {res}\n\n")

            current_turn += 1

        total_turns += current_turn

        frame_winner = match.winner if match.winner is not None else random.choice([0, 1])
        if frame_winner == 0:
            p1_wins += 1
        else:
            p2_wins += 1
        with open(match_filename, 'a') as mf:
            mf.write(f"Agent {[p1_idx, p2_idx][match.turn]} wins frame {frames_played}.\n\n")
        print(f"({time.strftime('%H:%M:%S', time.gmtime(time.time()))}) "
              f"[Worker {pid:05d} | FRAME COMPLETE] Agent {p1_idx} vs Agent {p2_idx}: {p1_wins}-{p2_wins}")

    elapsed_time = time.time() - start_time

    print(f"({time.strftime('%H:%M:%S', time.gmtime(time.time()))}) "
          f"[Worker {pid:05d} | MATCH OVER] Agent {p1_idx} vs Agent {p2_idx} | Final Score: {p1_wins}-{p2_wins} | "
          f"Time: {elapsed_time:.2f}s | Total Turns: {total_turns}")

    return p1_wins, p2_wins, total_turns, frames_played


def run_tournament(gen, pop):
    import itertools
    matchups = list(itertools.combinations(range(len(pop)), 2))
    results = {i: 0 for i in range(len(pop))}

    total_tournament_turns = 0
    total_tournament_frames = 0
    print(f"Starting tournament on 14 cores")

    # Use Pebble's ProcessPool to actively terminate frozen OS processes
    with ProcessPool(max_workers=14) as pool:
        futures = {}
        for ind, (p1_idx, p2_idx) in enumerate(matchups):
            w1 = pop[p1_idx]
            w2 = pop[p2_idx]
            # Schedule the task with a hard OS-level timeout
            future = pool.schedule(simulate_matchup, args=[gen, ind, p1_idx, p2_idx, w1, w2], timeout=2400)
            futures[future] = (p1_idx, p2_idx)

        for future in futures:
            p1_idx, p2_idx = futures[future]
            try:
                p1_wins, p2_wins, match_turns, match_frames = future.result()

                results[p1_idx] += p1_wins
                results[p2_idx] += p2_wins
                total_tournament_turns += match_turns
                total_tournament_frames += match_frames

            except TimeoutError:
                print(f"\n[TIMEOUT] Worker terminated for Ag {p1_idx} vs Ag {p2_idx}! Both penalized.")
                results[p1_idx] -= 2
                results[p2_idx] -= 2

            except Exception as e:
                import traceback
                print(f"\n[WORKER CRASH] Agent {p1_idx} vs Agent {p2_idx} crashed:", e)
                traceback.print_exc()

    avg_turns_per_frame = total_tournament_turns / total_tournament_frames if total_tournament_frames > 0 else 0
    return results, avg_turns_per_frame


def crossover_and_mutate(w1, w2):
    """Combines winning traits and adds mutation."""
    child = {}
    for key in BASE_WEIGHTS:
        # 50/50 chance to take a trait from Parent 1 or Parent 2
        child[key] = w1[key] if random.random() > 0.5 else w2[key]

        # 10% chance for a severe random mutation
        if random.random() < 0.1:
            child[key] *= random.uniform(0.5, 1.5)
    return child


if __name__ == '__main__':
    POP_SIZE = 8
    START_GEN = 5  # TWEAK: Start at index 10 (Generation 11)
    TOTAL_GENERATIONS = 20  # TWEAK: Train up to 20 total generations

    if START_GEN > 0:
        print(f"Resuming training from Generation {START_GEN + 1}...")

        # Load the reigning champion from the previous run
        with open(f"best_gen_{START_GEN - 1}.json", "r") as f:
            champion = json.load(f)

        # Rebuild the population using the champion
        population = [copy.deepcopy(champion), copy.deepcopy(champion)]  # Keep the exact champion

        # Fill the middle with mutated children of the champion
        while len(population) < POP_SIZE - 2:
            population.append(crossover_and_mutate(champion, champion))

        # Add fresh random weights to prevent genetic stagnation
        population.append(generate_random_weights())
        population.append(generate_random_weights())

        # Load the existing history so we can append to it
        with open("training_history.json", "r") as f:
            training_history = json.load(f)
    else:
        # Standard fresh start
        population = [generate_random_weights() for _ in range(POP_SIZE)]
        training_history = []

    # TWEAK: Update the loop to use the new boundaries
    for gen in range(START_GEN, TOTAL_GENERATIONS):
        print(f"\n--- Starting Generation {gen + 1} ---")

        with open(f"debug_pop_gen_{gen}.json", "w") as f:
            json.dump(population, f, indent=4)

        scores, avg_turns = run_tournament(gen, population)

        # Sort indices by their score
        ranked_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        best_score = scores[ranked_indices[0]]
        print(f"Top Score this Gen: {best_score} wins | Avg Turns/Frame: {avg_turns:.2f}")

        # Select Top 2
        top_2 = [population[i] for i in ranked_indices[:2]]

        champion_weights = copy.deepcopy(top_2[0])

        # Save average turns to the JSON history alongside the standard metrics
        training_history.append({
            "generation": gen + 1,
            "top_score": best_score,
            "avg_turns_per_frame": avg_turns,
            "weights": champion_weights
        })

        with open(f"best_gen_{gen}.json", "w") as f:
            json.dump(champion_weights, f, indent=4)

        # Build next generation
        new_population = []
        new_population.extend(top_2)

        while len(new_population) < POP_SIZE - 2:
            p1, p2 = top_2
            new_population.append(crossover_and_mutate(p1, p2))

        new_population.append(generate_random_weights())
        new_population.append(generate_random_weights())

        population = new_population

    with open("training_history.json", "w") as f:
        json.dump(training_history, f, indent=4)

    print("\nTraining Complete! History saved to training_history.json")