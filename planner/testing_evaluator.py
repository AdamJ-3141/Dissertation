import pygame
import numpy as np
from pool_simulation.physics.engine import Simulation
from pool_simulation.render.pygame_renderer import Renderer
from planner.evaluator import TableEvaluator
from pool_simulation.constants import *


def main():
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("consolas", 24)

    NUM_STATES = 5
    TARGET_COLOUR = 1

    sim = Simulation(n_obj_balls=15)
    renderer = Renderer(sim)

    evaluations = []

    print(f"Generating and evaluating {NUM_STATES} states...")

    for i in range(NUM_STATES):
        sim.set_up_randomly(15)

        # We assume they are all in play for the test
        sim.in_play = np.ones(sim.n_obj_balls + 1, dtype=bool)

        evaluator = TableEvaluator(sim, target_colour=TARGET_COLOUR)

        # 2. Extract Individual Scores
        _, _, _, freedom = evaluator.get_full_heatmap()
        attack = evaluator.direct_pots_score()
        strat = evaluator.cluster_analysis_score()

        weighted_freedom = freedom * evaluator.w["w_easiness"]
        weighted_attack = attack * evaluator.w["w_attackability"]
        weighted_strat = strat * evaluator.w["w_strategic"]
        total_score = weighted_freedom + weighted_attack + weighted_strat

        # 3. Render and Capture
        renderer.render(evaluator=evaluator, flip=False)
        raw_surface = renderer.screen.copy()

        scale_factor = 0.5
        new_w = int(raw_surface.get_width() * scale_factor)
        new_h = int(raw_surface.get_height() * scale_factor)
        state_surface = pygame.transform.smoothscale(raw_surface, (new_w, new_h))

        evaluations.append({
            "total": total_score,
            "freedom": freedom,
            "attack": attack,
            "strat": strat,
            "w_freedom": weighted_freedom,
            "w_attack": weighted_attack,
            "w_strat": weighted_strat,
            "surface": state_surface
        })
    # Sort best to worst
    evaluations.sort(key=lambda x: x["total"], reverse=True)

    table_w = evaluations[0]["surface"].get_width()
    table_h = evaluations[0]["surface"].get_height()

    text_panel_w = 400
    row_h = table_h

    master_surface = pygame.Surface((table_w + text_panel_w, row_h * NUM_STATES))
    master_surface.fill((30, 30, 30))

    for idx, eval_data in enumerate(evaluations):
        y_offset = idx * row_h

        # Blit table image
        master_surface.blit(eval_data["surface"], (0, y_offset))

        # Render Text
        texts = [
            f"Rank: #{idx + 1}",
            f"Total Score: {eval_data['total']:>7.3f}",
            "-" * 30,
            "Raw Metrics:",
            f"  Freedom:   {eval_data['freedom']:>7.3f}",
            f"  Attack:    {eval_data['attack']:>7.3f}",
            f"  Strategic: {eval_data['strat']:>7.3f}",
            "-" * 30,
            "Weighted Contributions:",
            f"  Freedom:   {eval_data['w_freedom']:>7.3f}",
            f"  Attack:    {eval_data['w_attack']:>7.3f}",
            f"  Strategic: {eval_data['w_strat']:>7.3f}",
        ]

        text_y = y_offset + 40
        for line in texts:
            # Color total score green/red based on value roughly
            color = (255, 255, 255) if "Rank" not in line else (100, 255, 100)
            text_surf = font.render(line, True, color)
            master_surface.blit(text_surf, (table_w + 30, text_y))
            text_y += 30

        # Draw a separator line
        pygame.draw.line(master_surface, (100, 100, 100), (0, y_offset + row_h - 1),
                         (table_w + text_panel_w, y_offset + row_h - 1), 2)

    # Save to disk
    out_file = "evaluation_report.png"
    pygame.image.save(master_surface, out_file)
    print(f"Done! Graphic saved to {out_file}")
    pygame.quit()


if __name__ == '__main__':
    main()