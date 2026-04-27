import json
import pygame
import threading
import numpy as np
from pool_simulation.physics.engine import Simulation
from pool_simulation.render.pygame_renderer import Renderer
from agent import Agent
from match import Match, TurnState


def main():

    with open("debug_pop_gen_0.json", "r") as f:
        population = json.load(f)

    p1_weights = population[0]
    p2_weights = population[7]

    # 2. Initialize simulation and restore state
    sim = Simulation(n_obj_balls=15)

    # Clear default setup
    sim.positions.fill(0.0)
    sim.velocities.fill(0.0)
    sim.angular.fill(0.0)
    sim.in_play.fill(False)
    sim.colours.fill(0)
    sim.ball_states.fill("STOPPED")

    sim.positions = np.array([[0.34631553215067684, 0.4210259002562583], [999.0, 999.0], [999.0, 999.0], [999.0, 999.0], [-0.615582726961229, -0.21952095862475843], [999.0, 999.0], [-0.5239853468687499, -0.4268869128453211], [999.0, 999.0], [999.0, 999.0], [0.481596438273, 0.0], [999.0, 999.0], [999.0, 999.0], [999.0, 999.0], [0.8210824203239896, 0.1027700532325726], [999.0, 999.0], [0.532406438273, 0.0]])

    sim.colours = np.array([0, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 3])

    sim.in_play = np.array([True, False, False, False, True, False, True, False, False, True, False, False, False, True, False, True]
)

    # 3. Initialize UI and Match
    renderer = Renderer(sim)
    match = Match(sim, custom_setup=True)
    match.turn = 0
    match.player_colours[0] = 1
    match.player_colours[1] = 2

    # 4. Initialize the Agents
    p1 = Agent(sim, weights=p1_weights)
    p2 = Agent(sim, weights=p2_weights)
    players = {0: p1, 1: p2}

    print(f"Recreating Match: Agent {0} vs Agent {7}")

    # 5. Main Game Loop (Adapted from play.py)
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

        print(f"Player {match.turn + 1}'s turn. Agent is thinking...")

        # A. Handle Ball in Hand manually
        if match.turn_state in [TurnState.BALL_IN_HAND_BAULK, TurnState.BALL_IN_HAND]:
            pos = active_player.get_cue_ball_in_hand_position(
                sim.colours, sim.in_play, sim.positions,
                match.player_colours[match.turn], match.turn_state
            )
            try:
                sim.move_cue_ball(pos, baulk=(match.turn_state == TurnState.BALL_IN_HAND_BAULK))
                match.turn_state = TurnState.NORMAL
            except ValueError:
                match.turn_state = TurnState.BALL_IN_HAND
                match.turn = 1 - match.turn
                continue

        renderer.render(flip=True)
        pygame.display.set_caption(f"Agent {0 if match.turn == 0 else 7} is thinking...")

        # B. Ask the AI for parameters silently on a background thread
        shot_result = []

        def plan_shot():
            params = active_player.get_shot_parameters(
                sim.colours, sim.in_play, sim.positions,
                match.player_colours[match.turn], match.turn_state,
                renderer=None  # Detaches the Monte Carlo visualizer
            )
            shot_result.append(params)

        # Draw the frozen table ONCE before the AI starts messing with the physics engine
        renderer.render(flip=True)
        pygame.display.set_caption(f"Agent {0 if match.turn == 0 else 7} is thinking...")

        think_thread = threading.Thread(target=plan_shot)
        think_thread.start()

        # Keep the Pygame window responsive while the agent thinks, BUT DO NOT re-render
        while think_thread.is_alive():
            for evt in pygame.event.get():
                if evt.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            # Limit the while-loop so it doesn't eat CPU cycles from the AI
            renderer.clock.tick(30)

        think_thread.join()
        vel_x, vel_y, tip_y, tip_x, cue_elev = shot_result[0]

        # C. Execute the shot and record frames for playback
        valid = sim.strike_cue_ball(vel_x, vel_y, tip_y, tip_x, cue_elev)
        shot_data = sim.run(framerate=60, frame_callback=record_frame)
        shot_data["valid"] = valid

        # D. Referee evaluates the result
        match.evaluate_shot(shot_data)

        # E. Save real state, play back the visualizer, then restore real state
        final_referee_state = {
            'positions': sim.positions.copy(),
            'angular': sim.angular.copy(),
            'in_play': sim.in_play.copy(),
            'ball_states': sim.ball_states.copy()
        }

        pygame.display.set_caption("Shot Playback")
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


if __name__ == '__main__':
    main()