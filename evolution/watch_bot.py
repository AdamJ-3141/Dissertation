import pygame
import pickle
import neat
import os
import numpy as np

from pool_simulation.physics.engine import Simulation
from pool_simulation.render import Renderer
from evolution import Human, Match, TurnState
from agent import Agent  # Import your new Agent class!


def main():
    # ==========================================
    # 1. LOAD THE AI BRAIN
    # ==========================================
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')

    # Load the exact NEAT configuration used for training
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Load the saved champion genome
    with open('best_pool_bot.pkl', 'rb') as f:
        winner_genome = pickle.load(f)

    # Rebuild the neural network and give it to the Agent
    net = neat.nn.FeedForwardNetwork.create(winner_genome, config)

    # ==========================================
    # 2. SETUP THE MATCH
    # ==========================================
    # Start the simulation (we will override the 15 balls in a second)
    sim = Simulation(n_obj_balls=15, start_break=False)
    renderer = Renderer(sim)

    # Recreate the exact 1-ball drill the AI mastered
    rx = np.random.uniform(-sim.table_width / 2.0 + 0.1, sim.table_width / 2.0 - 0.1)
    ry = np.random.uniform(-sim.table_height / 2.0 + 0.1, sim.table_height / 2.0 - 0.1)
    positions = [rx, ry]
    target_color = 1
    sim.colours[1] = target_color

    # Place 3 easy target balls
    sim.positions[1] = positions
    sim.in_play.fill(False)
    sim.in_play[:2] = True

    match = Match(sim, play_break=False, custom_setup=True)

    # Force the AI to be Player 1 so we immediately see its shot
    match.player_colours[0] = 1
    ai_player = Agent(sim, net)
    # p1 is the AI, p2 is Human (or you can make them both AI!)
    p1 = ai_player
    p2 = Human(sim, renderer)
    players = {0: p1, 1: p2}

    # ==========================================
    # 3. MAIN GAME LOOP
    # ==========================================
    while match.turn_state != TurnState.GAME_OVER:
        active_player = players[match.turn]

        playback_frames = []

        def record_frame(simulation_instance):
            playback_frames.append({
                'positions': simulation_instance.positions.copy(),
                'angular': simulation_instance.angular.copy(),
                'in_play': simulation_instance.in_play.copy(),
                'ball_states': simulation_instance.ball_states.copy()
            })

        print(f"Player {match.turn + 1}'s turn. Waiting for shot...")

        # Ask the referee to take a turn.
        match.play_turn(active_player, frame_callback=record_frame)

        final_referee_state = {
            'positions': sim.positions.copy(),
            'angular': sim.angular.copy(),
            'in_play': sim.in_play.copy(),
            'ball_states': sim.ball_states.copy()
        }

        # Playback the physical simulation that just occurred
        for frame in playback_frames:
            sim.positions = frame['positions']
            sim.angular = frame['angular']
            sim.in_play = frame['in_play']
            sim.ball_states = frame['ball_states']

            renderer.render(fps=60, flip=True)
            renderer.clock.tick(60)

            for evt in pygame.event.get():
                if evt.type == pygame.QUIT:
                    pygame.quit()
                    exit()

        sim.positions = final_referee_state['positions']
        sim.angular = final_referee_state['angular']
        sim.in_play = final_referee_state['in_play']
        sim.ball_states = final_referee_state['ball_states']

        print("Shot complete. Referee evaluating...")

        # Stop the game after the AI takes its one perfect shot so we can analyze it
        if match.turn == 0:
            print("AI shot complete. Exiting for analysis.")
            break

    pygame.quit()


if __name__ == "__main__":
    main()