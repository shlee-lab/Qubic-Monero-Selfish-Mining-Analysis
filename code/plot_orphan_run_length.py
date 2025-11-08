import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_visualizations(res):
    """Create a single (right) visualization: Orphan Count vs Run Length (scatter).
       Exports as a square PDF."""
    # Ensure output directory exists
    os.makedirs('fig', exist_ok=True)

    fig, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    ax2.scatter(res['length_qubic_run'], res['total_orphans_on_run'],
                alpha=0.6, color='red', s=30)
    ax2.set_xlabel('Qubic Run Length', fontsize=14)
    ax2.set_ylabel('Total Orphans in Run', fontsize=14)
    ax2.grid(True, alpha=0.3)

    if len(res) > 0:
        max_x = int(res['length_qubic_run'].max())
        max_y = int(res['total_orphans_on_run'].max())
        ax2.set_xticks(range(1, max_x + 1))
        ax2.set_yticks(range(0, max_y + 1))

    plt.tight_layout()
    plt.show()
    plt.savefig('fig/orphan_run_length.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
	# Load data from CSV file
	csv_path = 'data/qubic_orphan_start_qubic_continuous_split.csv'
	if not os.path.exists(csv_path):
		print(f'Error: {csv_path} not found.')
		return
	
	print(f'Loading data from {csv_path}...')
	res = pd.read_csv(csv_path)
	print(f'Loaded {len(res)} runs')
	
	if len(res) == 0:
		print('No runs found.')
		return
	
	# Create visualizations
	create_visualizations(res)
	print('Saved visualization to fig/orphan_run_length.pdf')


if __name__ == '__main__':
	main()