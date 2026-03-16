import pygame
from pool_simulation.physics.engine import Simulation
from pool_simulation.render import Renderer
from evolution import Human, Match, TurnState


def main():
    sim = Simulation(n_obj_balls=15, start_break=True)
    renderer = Renderer(sim)

    # Initialize the Referee
    match = Match(sim, play_break=True)

    # Initialize the players
    p1 = Human(sim, renderer)
    p2 = Human(sim, renderer)
    players = {0: p1, 1: p2}

    # Main Game Loop
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

        # 2. Ask the referee to take a turn.
        # The referee will automatically call active_player.get_shot_parameters(),
        # which will trigger the Pygame UI loop in the Human class!
        print(f"Player {match.turn + 1}'s turn. Waiting for shot...")
        match.play_turn(active_player, frame_callback=record_frame)

        final_referee_state = {
            'positions': sim.positions.copy(),
            'angular': sim.angular.copy(),
            'in_play': sim.in_play.copy(),
            'ball_states': sim.ball_states.copy()
        }

        # 3. Playback the physical simulation that just occurred
        for frame in playback_frames:
            sim.positions = frame['positions']
            sim.angular = frame['angular']
            sim.in_play = frame['in_play']
            sim.ball_states = frame['ball_states']

            renderer.render(fps=60, flip=True)
            renderer.clock.tick(60)

            # Allow quitting during playback
            for evt in pygame.event.get():
                if evt.type == pygame.QUIT:
                    pygame.quit()
                    exit()

        sim.positions = final_referee_state['positions']
        sim.angular = final_referee_state['angular']
        sim.in_play = final_referee_state['in_play']
        sim.ball_states = final_referee_state['ball_states']

        print("Shot complete. Referee evaluating...")

    print(f"Match Over! Player {match.winner + 1} wins!")
    pygame.quit()


if __name__ == "__main__":
    main()