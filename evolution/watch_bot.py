import json
import time
import sys
import os
import numpy as np
import pygame
import warnings
warnings.filterwarnings("ignore", message=".*avx2 capable.*")

from pool_simulation.physics import Simulation
from pool_simulation.render import Renderer


def watch_telemetry():
    sim = Simulation(start_break=False)
    renderer = Renderer(sim)
    last_mod_time = 0

    print("Waiting for the training script to broadcast its first shot...")

    def render_frame(current_sim):
        """Callback passed to sim.run() to draw each frame."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # The renderer's clock.tick(fps) handles the real-time playback speed
        renderer.render(fps=60)

    while True:
        # 1. Wait for a new file update
        if not os.path.exists("live_shot.json"):
            time.sleep(1.0)
            continue

        current_mod_time = os.path.getmtime("live_shot.json")
        if current_mod_time == last_mod_time:
            time.sleep(0.1)
            continue

        # 2. Safely read the JSON
        try:
            with open("live_shot.json", "r") as f:
                shot = json.load(f)

            last_mod_time = current_mod_time

        except (json.JSONDecodeError, PermissionError, ValueError):
            # The file is currently half-written. Wait a tiny bit and try again.
            time.sleep(0.05)
            continue

        # 3. Setup the physical table (with bulletproof explicit datatypes)
        sim.reset()
        sim.positions = np.array(shot["positions"], dtype=np.float64)
        sim.in_play = np.array(shot["in_play"], dtype=bool)
        sim.colours = np.array(shot["colours"], dtype=int)

        # 4. Decode the action
        raw_action = shot["action"]
        vx, vy = raw_action[0] * 7.0, raw_action[1] * 7.0
        theta = raw_action[2] * np.pi
        r = (raw_action[3] + 1) / 2 * 0.75

        topspin = r * np.sin(theta)
        sidespin = r * np.cos(theta)
        elevation = ((raw_action[4] + 1.0) / 2.0) * 60

        # --- Calculate Aim Line Parameters ---
        speed = float(np.hypot(vx, vy))

        # Find a target point along the velocity vector and convert to screen pixels
        if speed > 1e-5:
            target_world_x = sim.positions[0][0] + vx
            target_world_y = sim.positions[0][1] + vy
            aim_screen_x, aim_screen_y = renderer.world_to_screen((target_world_x, target_world_y))
        else:
            aim_screen_x, aim_screen_y = renderer.world_to_screen(sim.positions[0])

        # --- Hold the initial frame for 2 seconds and draw the UI ---
        pause_start = time.time()
        while time.time() - pause_start < 2.0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # 1. Draw the base table and balls WITHOUT flipping the display yet
            renderer.render(fps=60, flip=False)

            # 2. Draw the Ghost Ball and Aim Line
            if speed > 1e-5:
                renderer.draw_aim_line(aim_screen_x, aim_screen_y, speed, topspin, sidespin, elevation)

            # 3. Draw the custom HUD elements
            renderer.draw_power_scale(speed)

            # raw_action[3] is side (-1 to 1), raw_action[2] is top (-1 to 1)
            renderer.draw_spin_ui(tip_x=sidespin, tip_y=topspin)
            renderer.draw_elevation_ui(elevation)

            # 4. Flip the fully composed frame to the monitor
            pygame.display.flip()
            renderer.clock.tick(60)

        # 5. Strike the ball
        sim.strike_cue_ball(vx, vy, topspin, sidespin, elevation)

        # 6. Let the engine handle the rest!
        sim.run(framerate=60.0, frame_callback=render_frame)


if __name__ == '__main__':
    watch_telemetry()