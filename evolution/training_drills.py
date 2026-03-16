import neat
import os
import numpy as np
from enum import IntEnum
import pickle

from pool_simulation.physics import Simulation
from agent import Agent

class TurnState(IntEnum):
    NORMAL = 0
    BALL_IN_HAND = 1
    BALL_IN_HAND_BAULK = 2
    GAME_OVER = 3


# Fitness tuning weights
REWARD_POT = 50.0
REWARD_FIRST_HIT_TARGET = 10.0
REWARD_CLEAR_TABLE = 100.0

PENALTY_INVALID_SHOT = -20.0  # Cue clipping through table/balls
PENALTY_INVALID_PLACE = -20.0  # Dropping the cue ball on top of another ball
PENALTY_FOUL = -10.0  # Scratching or knocking balls off the table


def eval_genomes(genomes, config):
    """The NEAT training loop. Evaluates every bot in the generation."""

    for genome_id, genome in genomes:
        genome.fitness = 0.0

        # 1. Give the agent its neural brain
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        agent = Agent(net)

        sim = Simulation(start_break=False)
        sim.reset()
        sim.in_play.fill(False)

        # ==========================================
        # SET UP THE DRILL (3 Target Balls)
        # ==========================================
        positions = [[0.3, 0.2], [0.5, -0.2], [0.7, 0.0]]
        target_color = 1
        sim.colours[1:4] = target_color

        # Place 3 easy target balls
        sim.positions[1:4] = positions
        sim.in_play[1:4] = True

        # The AI will need to place the cue ball on its first turn
        needs_placement = True
        max_turns = 5

        # ==========================================
        # PLAY THE SHOTS
        # ==========================================
        for turn in range(max_turns):

            # 1. Cue Ball Placement Phase
            if needs_placement:
                placement_pos = agent.get_cue_ball_in_hand_position(
                    sim.colours, sim.in_play, sim.positions,
                    target_color, TurnState.BALL_IN_HAND
                )
                try:
                    # The engine will raise a ValueError if this is illegal
                    sim.move_cue_ball(placement_pos, baulk=False)
                    needs_placement = False
                except ValueError:
                    genome.fitness += PENALTY_INVALID_PLACE
                    break  # Turn ends, genome dies

            # 2. Shot Calculation Phase
            shot_params = agent.get_shot_parameters(
                sim.colours, sim.in_play, sim.positions,
                target_color, TurnState.NORMAL
            )
            vx, vy, top, side, el = shot_params


            # 4. Execute the Physics!
            valid_shot = sim.strike_cue_ball(vx, vy, top, side, el)
            if not valid_shot:
                genome.fitness += PENALTY_INVALID_SHOT
                break  # Turn ends, genome dies

            shot_data = sim.run()

            # ==========================================
            # FITNESS EVALUATION
            # ==========================================

            # Foul: Ball off table
            if shot_data["error"]:
                genome.fitness += PENALTY_FOUL
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
                genome.fitness += REWARD_FIRST_HIT_TARGET

            # Evaluate Potted Balls
            for potted_idx in shot_data["balls_potted"]:
                if potted_idx == 0:
                    # Scratched the cue ball
                    genome.fitness += PENALTY_FOUL
                    needs_placement = True
                elif sim.colours[potted_idx] == target_color:
                    # Potted a target ball!
                    genome.fitness += REWARD_POT

            # Check for Drill Completion
            if not np.any(sim.in_play[1:4]):
                genome.fitness += REWARD_CLEAR_TABLE
                break  # Drill complete!


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

    # NEW: Save a backup of the entire population every 10 generations
    p.add_reporter(neat.Checkpointer(10, filename_prefix='neat-checkpoint-'))

    print("Spawning Generation 0...")

    # Run evolution
    winner = p.run(eval_genomes, 300)

    print('\nEvolution Complete!')
    print(f'Best Genome ID: {winner.key}')
    print(f'Best Fitness: {winner.fitness}')

    # NEW: Save the ultimate winner to a file!
    with open('best_pool_bot.pkl', 'wb') as f:
        pickle.dump(winner, f)  # noqa

    print("Saved the champion genome to 'best_pooExpected type 'SupportsWrite[bytes]', got 'BufferedWriter' insteadl_bot.pkl'")

if __name__ == '__main__':
    run_neat()