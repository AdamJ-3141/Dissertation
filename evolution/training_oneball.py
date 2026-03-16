import neat
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import numpy as np
from enum import IntEnum
import pickle
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
from datetime import datetime, timedelta
from pool_simulation.physics import Simulation
from agent import Agent

class TurnState(IntEnum):
    NORMAL = 0
    BALL_IN_HAND = 1
    BALL_IN_HAND_BAULK = 2
    GAME_OVER = 3


# Fitness tuning weights
REWARD_POT = 100.0
REWARD_FIRST_HIT_TARGET = 10.0

PENALTY_INVALID_SHOT = -20.0  # Cue clipping through table/balls
PENALTY_INVALID_PLACE = -20.0  # Dropping the cue ball on top of another ball
PENALTY_FOUL = -40.0  # Scratching or knocking balls off the table


def eval_genome(genome, config):
    """The NEAT training loop. Evaluates every bot in the generation."""
    fitness = 0.0

    # 1. Give the agent its neural brain
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    sim = Simulation(start_break=False)
    sim.reset()
    sim.in_play.fill(False)

    agent = Agent(sim, net)

    # ==========================================
    # SET UP THE DRILL (3 Target Balls)
    # ==========================================
    target_color = 1
    sim.colours[1] = target_color
    for i in range(10):
        # Place 1 target ball
        rx = np.random.uniform(-sim.table_width / 2.0 + 0.1, sim.table_width / 2.0 - 0.1)
        ry = np.random.uniform(-sim.table_height / 2.0 + 0.1, sim.table_height / 2.0 - 0.1)
        positions = [rx, ry]
        sim.positions[1] = [rx, ry]
        sim.in_play[1] = True
        sim.in_play[0] = True

        # The AI will need to place the cue ball on its first turn
        needs_placement = True
        max_turns = 1

        # ==========================================
        # PLAY THE SHOTS
        # ==========================================
        for turn in range(max_turns):

            # 1. Cue Ball Placement Phase
            if needs_placement:
                # --- OVERRIDE AI PLACEMENT ---
                # Force the cue ball to the exact same starting spot every time
                placement_pos = np.array([-sim.table_width / 4.0, 0.0])

                try:
                    sim.move_cue_ball(placement_pos, baulk=False)
                    needs_placement = False
                except ValueError:
                    fitness += PENALTY_INVALID_PLACE
                    break

            # 2. Shot Calculation Phase
            shot_params = agent.get_shot_parameters(
                sim.colours, sim.in_play, sim.positions,
                target_color, TurnState.NORMAL
            )
            vx, vy, top, side, el = shot_params


            # 4. Execute the Physics!
            valid_shot = sim.strike_cue_ball(vx, vy, top, side, el)
            if not valid_shot:
                fitness += PENALTY_INVALID_SHOT
                break  # Turn ends, genome dies

            shot_data = sim.run()

            # ==========================================
            # FITNESS EVALUATION
            # ==========================================

            # Foul: Ball off table
            if shot_data["error"]:
                fitness += PENALTY_FOUL
                # Safely respot only the object balls that flew off
                for ball_idx in shot_data["error_balls"]:
                    if ball_idx == 0:
                        needs_placement = True  # Cue ball flew off
                    else:
                        # Respot the target ball to its original starting coordinate
                        sim.positions[ball_idx] = positions[ball_idx]

                        # Stop its momentum and put it back in play
                        sim.velocities[ball_idx] = [0.0, 0.0]
                        sim.angular[ball_idx] = [0.0, 0.0, 0.0]
                        sim.in_play[ball_idx] = True

            # Reward: Good aiming (hit a legal ball first)
            first_hit = shot_data["first_ball_hit"]
            if first_hit is not None and sim.colours[first_hit] == target_color:
                fitness += REWARD_FIRST_HIT_TARGET

                # Check if ANY of the potted balls match the target color
                potted_target = any(sim.colours[b] == target_color for b in shot_data["balls_potted"])

                if potted_target:
                    fitness += REWARD_POT
                else:
                    # Only calculate proximity if no target ball was potted
                    target_pos = sim.positions[first_hit]
                    min_dist = float('inf')
                    for pocket in sim.pockets:
                        dist = np.linalg.norm(target_pos - pocket[:2])
                        if dist < min_dist:
                            min_dist = dist

                    if min_dist < 0.3:
                        proximity_bonus = 50.0 * np.exp(-15.0 * min_dist)
                        fitness += proximity_bonus
    return fitness


class ETAReporter(neat.reporting.BaseReporter):
    def __init__(self, total_generations):
        self.total_generations = total_generations
        self.generation_times = []
        self.last_gen_start_time = None
        self.current_gen = 0

    def start_generation(self, generation):
        current_time = time.time()

        # Measure the exact time elapsed since the start of the PREVIOUS generation
        if self.last_gen_start_time is not None:
            gen_time = current_time - self.last_gen_start_time
            self.generation_times.append(gen_time)

        self.last_gen_start_time = current_time
        self.current_gen = generation

    def post_evaluate(self, config, population, species, best_genome):
        # Fallback for Generation 0 before we have a full cycle's data
        if not self.generation_times:
            avg_time = time.time() - self.last_gen_start_time
        else:
            # Rolling average of the last 10 full generations for stability
            recent_times = self.generation_times[-10:]
            avg_time = sum(recent_times) / len(recent_times)

        generations_left = self.total_generations - (self.current_gen + 1)

        if generations_left > 0:
            eta_seconds = avg_time * generations_left
            eta_clock_time = datetime.now() + timedelta(seconds=eta_seconds)

            time_left_str = str(timedelta(seconds=int(eta_seconds)))
            print(f" ⏳ ETA: {time_left_str} left. (Finishes around {eta_clock_time.strftime('%I:%M %p')})\n")


def run_neat():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    p.add_reporter(neat.Checkpointer(50, filename_prefix='checkpoints/neat-checkpoint-'))
    MAX_GENERATIONS = 1000  # Set your target limit here
    p.add_reporter(ETAReporter(MAX_GENERATIONS))
    print("Spawning Generation 0...")

    num_cores = max(1, mp.cpu_count() - 1)
    print(f"🔥 Firing up {num_cores} CPU cores for parallel training...")

    # Create the Parallel Evaluator, passing it our new single-genome function
    pe = neat.ParallelEvaluator(num_cores, eval_genome)

    # Run evolution using pe.evaluate instead of our old function!
    winner = p.run(pe.evaluate, MAX_GENERATIONS)

    print('\nEvolution Complete!')
    print(f'Best Genome ID: {winner.key}')
    print(f'Best Fitness: {winner.fitness}')

    with open('best_pool_bot.pkl', 'wb') as f:
        pickle.dump(winner, f)  # noqa

    print("Saved the champion genome to 'best_pool_bot.pkl'")

    print("Generating performance graph...")

    # Extract the data from the NEAT stats reporter
    generation = range(len(stats.most_fit_genomes))
    best_fitness = [c.fitness for c in stats.most_fit_genomes]
    avg_fitness = stats.get_fitness_mean()

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(generation, best_fitness, label="Best Fitness (Champion)", color='blue', linewidth=2)
    plt.plot(generation, avg_fitness, label="Average Fitness (Population)", color='red', alpha=0.7)

    # Format the graph
    plt.title("AI Pool Bot Training Performance Over Time")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="best")

    # Save it as an image and show it on screen!
    plt.savefig("fitness_history.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    run_neat()