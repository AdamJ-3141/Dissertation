import matplotlib.pyplot as plt
import numpy as np

def generate_win_percentage_chart():
    """
    Generates a publication-ready bar chart of the evaluation results.
    Replace these hardcoded numbers with your actual evaluation output!
    """
    # [Evolved Wins, Opponent Wins, Draws/Timeouts]
    results = {
        "Random Baseline": {"evolved": 28, "opp": 2, "draws": 0, "total": 30},
        "Greedy Baseline": {"evolved": 25, "opp": 5, "draws": 0, "total": 30},
        "Default Weights": {"evolved": 14, "opp": 14, "draws": 2, "total": 30},
        "Human (Author)": {"evolved": 2, "opp": 8, "draws": 0, "total": 10}
    }

    labels = list(results.keys())

    # Calculate Win % (Ignoring draws/timeouts to focus on decisive frames)
    evolved_win_pct = []
    opp_win_pct = []

    for key, data in results.items():
        decisive_games = data["total"] - data["draws"]
        if decisive_games > 0:
            evolved_win_pct.append((data["evolved"] / decisive_games) * 100)
            opp_win_pct.append((data["opp"] / decisive_games) * 100)
        else:
            evolved_win_pct.append(0)
            opp_win_pct.append(0)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Professional color palette
    rects1 = ax.bar(x - width / 2, evolved_win_pct, width, label='Evolved Agent (Gen 19)', color='#2ca02c',
                    edgecolor='black')
    rects2 = ax.bar(x + width / 2, opp_win_pct, width, label='Opponent', color='#d62728', edgecolor='black')

    ax.set_ylabel('Win Percentage (%)', fontsize=12)
    ax.set_title('Evaluation Results: Evolved Agent vs Baselines (Decisive Frames)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)  # Give space for labels above 100
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add text labels on top of the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig("win_percentage_chart.png", dpi=300)
    print("Saved win_percentage_chart.png")

if __name__ == "__main__":
    generate_win_percentage_chart()
