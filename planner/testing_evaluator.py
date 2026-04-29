import pygame
import numpy as np
from pool_simulation.physics.engine import Simulation
from pool_simulation.render.pygame_renderer import Renderer
from planner.evaluator import TableEvaluator


def main():
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("consolas", 20)

    NUM_STATES = 5
    TARGET_COLOUR = 1

    sim = Simulation(n_obj_balls=15)
    renderer = Renderer(sim)

    evaluations = []

    print(f"Generating and evaluating {NUM_STATES} states...")

    for i in range(NUM_STATES):
        sim.set_up_randomly(15)
        sim.in_play = np.ones(sim.n_obj_balls + 1, dtype=bool)

        evaluator = TableEvaluator(sim, target_colour=TARGET_COLOUR)

        targets = [idx for idx in range(1, sim.n_obj_balls + 1) if
                   sim.colours[idx] in (TARGET_COLOUR, 3) and sim.in_play[idx]]
        num_targets = max(1, len(targets))

        # 1. Extract Individual Scores
        _, _, _, raw_freedom = evaluator.get_full_heatmap()
        raw_attack = evaluator.direct_pots_score()
        norm_strat = evaluator.cluster_analysis_score()

        try:
            norm_vis = evaluator.visibility_analysis_score()
        except AttributeError:
            norm_vis = 0.0

        # 2. Normalize the remaining scores to a -1.0 to 1.0 scale
        norm_freedom = raw_freedom
        norm_attack = min(1.0, (raw_attack / num_targets) / 1.35)

        # 3. Fetch weights safely from JSON
        w_easiness = evaluator.w.get("w_easiness", 0.33)
        w_attackability = evaluator.w.get("w_attackability", 0.33)
        w_strategic = evaluator.w.get("w_strategic", 0.33)
        w_safety = evaluator.w.get("w_safety", 0.15)

        # 4. Calculate final weighted contributions
        weighted_freedom = norm_freedom * w_easiness
        weighted_attack = norm_attack * w_attackability
        weighted_strat = norm_strat * w_strategic
        weighted_vis = norm_vis * w_safety

        total_score = weighted_freedom + weighted_attack + weighted_strat + weighted_vis

        # Extract the surface
        sim.saved_positions = sim.positions.copy()
        renderer.render(flip=False)
        surface = renderer.screen.copy()

        evaluations.append({
            "surface": surface,
            "total": total_score,
            "norm_freedom": norm_freedom,
            "norm_attack": norm_attack,
            "norm_strat": norm_strat,
            "norm_vis": norm_vis,
            "w_freedom": weighted_freedom,
            "w_attack": weighted_attack,
            "w_strat": weighted_strat,
            "w_vis": weighted_vis
        })

    # Sort by total score (descending)
    evaluations.sort(key=lambda x: x["total"], reverse=True)

    # --- NEW DISPLAY LOGIC: ONE AT A TIME ---
    table_w, table_h = renderer.screen.get_size()

    # Resize the Pygame display to fit one table + 450px for the text panel
    screen = pygame.display.set_mode((table_w + 450, table_h))

    for idx, eval_data in enumerate(evaluations):
        screen.fill((30, 30, 30))

        # Blit the table image
        screen.blit(eval_data["surface"], (0, 0))

        pygame.display.set_caption(f"Evaluator Sandbox | Rank {idx + 1}/{NUM_STATES} | PRESS SPACE FOR NEXT")

        # Render Text
        texts = [
            f"Rank: #{idx + 1} of {NUM_STATES}",
            f"Total Score: {eval_data['total']:>7.3f}",
            "-" * 35,
            "Normalized Metrics (-1.0 to 1.0):",
            f"  Easiness:  {eval_data['norm_freedom']:>7.3f}",
            f"  Attack:    {eval_data['norm_attack']:>7.3f}",
            f"  Strategic: {eval_data['norm_strat']:>7.3f}",
            f"  Safety:    {eval_data['norm_vis']:>7.3f}",
            "-" * 35,
            "Weighted Contributions:",
            f"  Easiness:  {eval_data['w_freedom']:>7.3f}",
            f"  Attack:    {eval_data['w_attack']:>7.3f}",
            f"  Strategic: {eval_data['w_strat']:>7.3f}",
            f"  Safety:    {eval_data['w_vis']:>7.3f}",
            "",
            "[PRESS SPACE TO VIEW NEXT]"
        ]

        text_y = 40
        for line in texts:
            if "Rank" in line:
                color = (100, 255, 100)
            elif "Total Score" in line:
                color = (255, 215, 0) if eval_data['total'] > 0 else (255, 100, 100)
            elif "SPACE" in line:
                color = (150, 150, 255)
            else:
                color = (200, 200, 200)

            text_surf = font.render(line, True, color)
            screen.blit(text_surf, (table_w + 20, text_y))
            text_y += 25

        pygame.display.flip()

        # Wait for user to press SPACE or QUIT
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False

    pygame.quit()


if __name__ == "__main__":
    main()