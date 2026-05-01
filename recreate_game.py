import json
import pygame
import threading
import numpy as np
from pool_simulation.physics.engine import Simulation
from pool_simulation.render.pygame_renderer import Renderer
from agent import Agent, Human
from match import Match, TurnState


def main():

    with open("debug_pop_gen_0.json", "r") as f:
        pop = json.load(f)

    p1_weights = pop[1]
    p2_weights = pop[3]

    # Initialize simulation and restore state
    sim = Simulation(n_obj_balls=15)

    # Clear default setup
    # sim.set_up_randomly(15)
    sim.reset(positions=np.array([[-0.5302449219472871, -0.19727313983360267], [-0.6853295201975858, -0.14168115958773864], [0.7691805858943301, -0.15159581826214164], [0.33518236931159096, -0.39540886609430226], [-0.3158725494503566, -0.05993844929682879], [0.0013173909159613917, 0.246289778687296], [0.27106626094276787, 0.028974297695481765], [0.699013836158553, 0.10562442177719716], [0.3600086112332974, 0.4212844349190327], [-0.5495729573701006, 0.04743633856882673], [0.3245050521179267, 0.19129406890193418], [0.1781524379127617, -0.13532573595025688], [0.4368664867836123, 0.12894520371291268], [0.008041566706902814, -0.2219609716817335], [-0.6018088899114244, -0.13781756019650887], [0.8456927333846014, 0.34105729548336866]]),
              colours=np.array([0, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 1, 2, 1, 3]),
              in_play=np.array([True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, True]))

    # Initialize UI and Match
    renderer = Renderer(sim)
    match = Match(sim, custom_setup=True)
    match.turn = 0
    match.player_colours[0] = 1
    match.player_colours[1] = 2

    # Initialize the Agents
    p1 = Human(sim, renderer)
    p2 = Agent(sim, weights=None)
    players = {0: p1, 1: p2}

    print(f"Recreating Match: Agent {0} vs Agent {7}")

    # Main Game Loop (Adapted from play.py)
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

        print(f"\n>>> AGENT {0 if match.turn == 0 else 7} IS PLANNING SHOT <<<")
        pygame.display.set_caption(f"Agent {0 if match.turn == 0 else 7} is thinking...")

        # Pass the renderer so the planner can draw its MC thoughts
        params = active_player.get_shot_parameters(
            sim.colours, sim.in_play, sim.positions,
            match.player_colours[match.turn], match.turn_state,
            renderer=renderer
        )
        vel_x, vel_y, tip_y, tip_x, cue_elev = params

        # Execute the shot and record frames for playback
        valid = sim.strike_cue_ball(vel_x, vel_y, tip_y, tip_x, cue_elev)
        shot_data = sim.run(framerate=60, frame_callback=record_frame)
        shot_data["valid"] = valid

        # Referee evaluates the result
        match.evaluate_shot(shot_data)

        # Save real state, play back the visualizer, then restore real state
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