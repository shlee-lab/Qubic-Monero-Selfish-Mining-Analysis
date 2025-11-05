import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def load_blocks():
	blocks = pd.read_csv('data/all_blocks.csv')
	blocks['timestamp'] = pd.to_datetime(blocks['timestamp'])
	if blocks['timestamp'].dt.tz is not None:
		blocks['timestamp'] = blocks['timestamp'].dt.tz_localize(None)
	return blocks


def build_height_index(blocks: pd.DataFrame):
	by_h = {h: g.sort_values('timestamp') for h, g in blocks.groupby('height')}
	main_map = {}
	orphan_count = {}
	for h, g in by_h.items():
		main = g[g['is_orphan'] == False]
		main_map[h] = main.iloc[0] if len(main) > 0 else None
		orphan_count[h] = int((g['is_orphan'] == True).sum())
	return by_h, main_map, orphan_count


def find_orphan_start_qubic_continuous_split(by_h, main_map, orphan_count):
	"""Find heights where orphan chains start, check if main is Qubic,
	then count consecutive Qubic main-chain blocks forward (orphan can stop),
	but when orphan reappears, start a new separate run.
	"""
	runs = []
	heights = sorted(main_map.keys())
	
	for i, h in enumerate(heights):
		# Check if this height has orphan chain (multiple blocks)
		if orphan_count.get(h, 0) > 0:
			main = main_map[h]
			# Check if main chain at this height is Qubic
			if main is not None and bool(main['is_qubic']) is True:
				# Start counting from this height
				start_h = h
				start_ts = main['timestamp']
				length = 1
				total_orphans = orphan_count.get(h, 0)
				max_consec_orphan_heights = 1 if orphan_count.get(h, 0) > 0 else 0
				curr_streak = 1 if orphan_count.get(h, 0) > 0 else 0
				
				# Count consecutive Qubic main-chain blocks forward
				# Stop when Qubic continuity breaks OR when orphan reappears (new state)
				j = i + 1
				while j < len(heights):
					hn = heights[j]
					mn = main_map[hn]
					# Check if next height is consecutive and has Qubic main
					if mn is not None and bool(mn['is_qubic']) is True and hn == heights[j-1] + 1:
						# Check if orphan reappears (new state start)
						if orphan_count.get(hn, 0) > 0 and curr_streak == 0:
							# Orphan reappeared after gap - start new run
							break
						
						length += 1
						o = orphan_count.get(hn, 0)
						total_orphans += o
						if o > 0:
							curr_streak += 1
							max_consec_orphan_heights = max(max_consec_orphan_heights, curr_streak)
						else:
							curr_streak = 0
						j += 1
					else:
						break
				
				end_h = heights[j-1]
				end_ts = main_map[end_h]['timestamp'] if main_map[end_h] is not None else None
				runs.append({
					'start_height': start_h,
					'end_height': end_h,
					'length_qubic_run': length,
					'total_orphans_on_run': total_orphans,
					'max_consecutive_orphan_heights': max_consec_orphan_heights,
					'start_ts': start_ts,
					'end_ts': end_ts,
				})
	
	return runs


def create_visualizations(res):
    """Create a single (right) visualization: Orphan Count vs Run Length (scatter).
       Exports as a square PDF."""
    import matplotlib.pyplot as plt

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
    plt.savefig('fig/orphan_run_length.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
	print('=== Orphan start + Qubic continuous (split when orphan reappears) ===')
	blocks = load_blocks()
	by_h, main_map, orphan_count = build_height_index(blocks)
	runs = find_orphan_start_qubic_continuous_split(by_h, main_map, orphan_count)
	res = pd.DataFrame(runs).sort_values(['start_height'])
	res.to_csv('data/qubic_orphan_start_qubic_continuous_split.csv', index=False)
	print(f'Saved qubic_orphan_start_qubic_continuous_split.csv with {len(res)} runs')
	if len(res) == 0:
		print('No runs found.')
		return
	print('\nTop 10 runs by length:')
	print(res.sort_values('length_qubic_run', ascending=False).head(10).to_string(index=False))
	print('\nDistribution (run length -> avg attached orphans):')
	grp = res.groupby('length_qubic_run')['total_orphans_on_run'].mean().reset_index()
	for _, r in grp.iterrows():
		print(f"  len={int(r['length_qubic_run'])}: avg_orphans={r['total_orphans_on_run']:.2f}")
	
	# Create visualizations
	create_visualizations(res)


if __name__ == '__main__':
	main()