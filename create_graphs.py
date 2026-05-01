import json
import matplotlib.pyplot as plt
import pandas as pd


def generate_graphs():
    # Load the data
    with open('training_history.json', 'r') as f:
        data = json.load(f)

    # Convert to a Pandas DataFrame for easy plotting
    df = pd.DataFrame(data)

    # Extract the nested weights into separate columns
    weights_df = pd.json_normalize(df['weights'])
    df = pd.concat([df.drop('weights', axis=1), weights_df], axis=1)

    # Set global plotting style for academic write-ups
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

    # GRAPH 1: The Fitness Curve (Top Score)
    plt.figure(figsize=(8, 5))
    plt.plot(df['generation'], df['top_score'], marker='o', color='#2ca02c', linewidth=2, markersize=8)
    plt.title('Genetic Algorithm Fitness Progression', fontsize=14, fontweight='bold')
    plt.xlabel('Generation')
    plt.ylabel('Top Score (Tournament Wins)')
    plt.xticks(df['generation'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('dissertation_fitness_curve.png')
    plt.close()

    # GRAPH 2: The Efficiency Curve (Average Turns)
    plt.figure(figsize=(8, 5))
    plt.plot(df['generation'], df['avg_turns_per_frame'], marker='s', color='#d62728', linewidth=2, markersize=8)
    plt.title('Match Efficiency Evolution', fontsize=14, fontweight='bold')
    plt.xlabel('Generation')
    plt.ylabel('Average Turns per Frame')
    plt.xticks(df['generation'])
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add a trendline to emphasize the drop
    z = np.polyfit(df['generation'], df['avg_turns_per_frame'], 1)
    p = np.poly1d(z)
    plt.plot(df['generation'], p(df['generation']), "r--", alpha=0.5, label="Trend")
    plt.legend()

    plt.tight_layout()
    plt.savefig('dissertation_efficiency_curve.png')
    plt.close()

    # GRAPH 3: Gene Evolution (Behavioral Weights)
    plt.figure(figsize=(10, 6))

    # Select the most interesting heuristic weights to plot
    key_weights = ['w_easiness', 'w_attackability', 'w_strategic', 'aggression_threshold', 'w_risk', 'w_safety']
    colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#8c564b', '#fe2222', '#dddd00']

    for weight, color in zip(key_weights, colors):
        plt.plot(df['generation'], df[weight], marker='^', linewidth=2, label=weight, color=color)

    plt.title('Evolution of Key Agent Heuristics', fontsize=14, fontweight='bold')
    plt.xlabel('Generation')
    plt.ylabel('Weight Value')
    plt.xticks(df['generation'])
    plt.legend(loc='lower left', frameon=True, shadow=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('dissertation_gene_evolution.png')
    plt.close()

    print("Graphs successfully generated: fitness_curve.png, efficiency_curve.png, gene_evolution.png")


if __name__ == "__main__":
    import numpy as np  # Needed for the trendline

    generate_graphs()