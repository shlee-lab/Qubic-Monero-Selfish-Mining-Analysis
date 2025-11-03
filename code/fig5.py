import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def load_blocks():
	"""Load and preprocess block data"""
	blocks = pd.read_csv('data/all_blocks.csv')
	blocks['timestamp'] = pd.to_datetime(blocks['timestamp'])
	if blocks['timestamp'].dt.tz is not None:
		blocks['timestamp'] = blocks['timestamp'].dt.tz_localize(None)
	return blocks


def identify_state_transitions(blocks):
	"""Identify state transitions based on proper state machine logic"""
	# Sort blocks by height and timestamp
	blocks = blocks.sort_values(['height', 'timestamp']).reset_index(drop=True)
	
	states = []
	prev_state = 0  # Start with state 0
	
	for i, block in blocks.iterrows():
		if not block['is_qubic']:
			continue
		
		height = block['height']
		current_timestamp = block['timestamp']
		
		# Get all blocks at this height
		height_blocks = blocks[blocks['height'] == height].sort_values('timestamp')
		qubic_blocks = height_blocks[height_blocks['is_qubic'] == True]
		non_qubic_blocks = height_blocks[height_blocks['is_qubic'] == False]
		
		# Check if there are orphans at this height
		has_orphans = len(height_blocks) > 1
		
		# Determine current state based on state machine logic
		current_state = 0
		transition = "N/A"
		
		if not has_orphans:
			# No orphans - state 0
			current_state = 0
			transition = f"{prev_state}->0"
		else:
			# There are orphans - need to determine state
			first_qubic = qubic_blocks.iloc[0]
			first_non_qubic = non_qubic_blocks.iloc[0] if len(non_qubic_blocks) > 0 else None
			
			if first_non_qubic is not None:
				if first_qubic['timestamp'] < first_non_qubic['timestamp']:
					# Qubic mined first
					if prev_state == 0:
						current_state = 1  # 0 -> 1 (start private chain)
						transition = "0->1"
					else:
						current_state = prev_state + 1  # Extend private chain
						transition = f"{prev_state}->{current_state}"
				else:
					# Others mined first
					if prev_state == 1:
						current_state = -1  # 1 -> -1 (catch-up)
						transition = "1->-1"
					elif prev_state == -1:
						current_state = 0  # -1 -> 0'' (0'' state)
						transition = "-1->0''"
					else:
						current_state = 0  # Reset to 0
						transition = f"{prev_state}->0"
			else:
				# Only Qubic blocks
				current_state = prev_state + 1 if prev_state > 0 else 1
				transition = f"{prev_state}->{current_state}"
		
		states.append({
			'height': height,
			'timestamp': current_timestamp,
			'state': current_state,
			'transition': transition,
			'prev_state': prev_state,
			'is_orphan': block['is_orphan'],
			'block_hash': block['block hash']
		})
		
		prev_state = current_state
	
	return pd.DataFrame(states)


def calculate_weekly_gamma_with_0_prime_estimation(states_df, blocks_df):
	"""Calculate weekly gamma values and estimate 0' state counts"""
	
	# Add weekly time unit
	states_df['week'] = states_df['timestamp'].dt.to_period('W-TUE').apply(lambda p: p.start_time.date())
	blocks_df['week'] = blocks_df['timestamp'].dt.to_period('W-TUE').apply(lambda p: p.start_time.date())
	
	results = []
	
	for week in states_df['week'].unique():
		# Filter data for this week
		week_states = states_df[states_df['week'] == week]
		week_blocks = blocks_df[blocks_df['week'] == week]
		
		# Calculate alpha (Qubic's mining power share)
		total_blocks = len(week_blocks)
		qubic_blocks = len(week_blocks[week_blocks['is_qubic'] == True])
		alpha = qubic_blocks / total_blocks if total_blocks > 0 else 0
		
		# Use the same logic as estimated 0' states for consistency
		# Find all 0' state situations using the same mathematical conditions
		valid_0_prime_situations = []
		
		for i, case in week_states[week_states['state'] == 1].iterrows():
			height = case['height']
			
			# Get all blocks at this height
			height_blocks = week_blocks[week_blocks['height'] == height].sort_values('timestamp')
			qubic_blocks_at_height = height_blocks[height_blocks['is_qubic'] == True]
			non_qubic_blocks_at_height = height_blocks[height_blocks['is_qubic'] == False]
			
			# Use the same mathematical conditions as estimated 0' state:
			# 1. has_orphans = True (len(height_blocks) > 1)
			# 2. first_qubic['timestamp'] < first_non_qubic['timestamp']
			# 3. prev_state == 0
			# 4. transition == "0->1"
			
			has_orphans = len(height_blocks) > 1
			qubic_mined_first = False
			
			if len(qubic_blocks_at_height) > 0 and len(non_qubic_blocks_at_height) > 0:
				first_qubic = qubic_blocks_at_height.iloc[0]
				first_non_qubic = non_qubic_blocks_at_height.iloc[0]
				qubic_mined_first = first_qubic['timestamp'] < first_non_qubic['timestamp']
			
			prev_state_0 = case['prev_state'] == 0
			transition_0_to_1 = case['transition'] == "0->1"
			
			# All conditions must be true for 0' state
			if has_orphans and qubic_mined_first and prev_state_0 and transition_0_to_1:
				# Check if Qubic's block is on main chain
				qubic_block_from_contested_height = qubic_blocks_at_height.iloc[0]
				qubic_wins = not qubic_block_from_contested_height['is_orphan']
				
				# Check if Qubic mined the next block (for gamma calculation)
				next_height = height + 1
				next_height_blocks = week_blocks[week_blocks['height'] == next_height].sort_values('timestamp')
				qubic_main_next = len(next_height_blocks[(next_height_blocks['is_qubic'] == True) & (next_height_blocks['is_orphan'] == False)])
				qubic_mined_next = qubic_main_next > 0
				
				valid_0_prime_situations.append({
					'height': height,
					'timestamp': case['timestamp'],
					'state': case['state'],
					'transition': case['transition'],
					'is_orphan': case['is_orphan'],
					'block_hash': case['block_hash'],
					'qubic_wins': qubic_wins,
					'qubic_mined_next': qubic_mined_next,
					'competition_resolved': True
				})
		
		valid_0_prime_df = pd.DataFrame(valid_0_prime_situations)
		
		# Calculate gamma: Qubic's block is on main chain AND Qubic did NOT mine the next block
		if len(valid_0_prime_df) > 0:
			gamma_cases = valid_0_prime_df[(valid_0_prime_df['qubic_wins'] == True) & (valid_0_prime_df['qubic_mined_next'] == False)]
		else:
			gamma_cases = pd.DataFrame()
		
		# Calculate gamma rate
		total_0_prime = len(valid_0_prime_df)
		gamma_successes = len(gamma_cases)
		gamma_rate = gamma_successes / total_0_prime if total_0_prime > 0 else 0
		
		# Estimate 0' state count using mathematical conditions
		# 0' state estimation: Qubic mined first AND there are orphans AND prev_state == 0
		estimated_0_prime_count = 0
		
		for i, case in week_states[week_states['state'] == 1].iterrows():
			height = case['height']
			
			# Get all blocks at this height
			height_blocks = week_blocks[week_blocks['height'] == height].sort_values('timestamp')
			qubic_blocks_at_height = height_blocks[height_blocks['is_qubic'] == True]
			non_qubic_blocks_at_height = height_blocks[height_blocks['is_qubic'] == False]
			
			# Mathematical conditions for 0' state:
			# 1. has_orphans = True (len(height_blocks) > 1)
			# 2. first_qubic['timestamp'] < first_non_qubic['timestamp']
			# 3. prev_state == 0
			# 4. transition == "0->1"
			
			has_orphans = len(height_blocks) > 1
			qubic_mined_first = False
			
			if len(qubic_blocks_at_height) > 0 and len(non_qubic_blocks_at_height) > 0:
				first_qubic = qubic_blocks_at_height.iloc[0]
				first_non_qubic = non_qubic_blocks_at_height.iloc[0]
				qubic_mined_first = first_qubic['timestamp'] < first_non_qubic['timestamp']
			
			prev_state_0 = case['prev_state'] == 0
			transition_0_to_1 = case['transition'] == "0->1"
			
			# All conditions must be true for 0' state
			if has_orphans and qubic_mined_first and prev_state_0 and transition_0_to_1:
				estimated_0_prime_count += 1
		
		results.append({
			'week': week,
			'alpha': alpha,
			'gamma': gamma_rate,
			'total_0_prime': total_0_prime,
			'gamma_successes': gamma_successes,
			'estimated_0_prime_count': estimated_0_prime_count,
			'blocks': len(week_blocks)
		})
	
	return pd.DataFrame(results)


def create_weekly_dual_axis_chart(weekly_df):
	"""Create weekly chart with dual y-axis showing gamma and 0' state counts"""
	fig, ax1 = plt.subplots(figsize=(10, 10))
	
	# Convert week to string for better x-axis display
	weekly_df['week_str'] = weekly_df['week'].astype(str)
	
	# Create bars for 0' state counts (left y-axis)
	bars = ax1.bar(weekly_df['week_str'], weekly_df['estimated_0_prime_count'], 
				   alpha=0.7, color='lightblue', edgecolor='navy', linewidth=1, label='Estimated 0\' State Count')
	ax1.set_xlabel('Week', fontsize=14)
	ax1.set_ylabel('Estimated 0\' State Count', fontsize=14)
	ax1.tick_params(axis='x', rotation=45)
	
	# Create line for gamma values (right y-axis)
	ax2 = ax1.twinx()
	line = ax2.plot(weekly_df['week_str'], weekly_df['gamma'], 'ro-', linewidth=2, markersize=6, label='γ Rate')
	ax2.set_ylabel('γ Rate', fontsize=14)
	ax2.set_ylim(-0.01, 0.2)  # Set y-axis range with slight offset from 0 for better visibility

	ax1.grid(True, alpha=0.3)
	
	# Add legend in upper right corner
	lines1, labels1 = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)
	
	# Add value labels on gamma line only (with larger font)
	for i, (week, gamma) in enumerate(zip(weekly_df['week_str'], weekly_df['gamma'])):
		if not pd.isna(gamma) and gamma > 0:
			# Position gamma values slightly above the line for better visibility
			ax2.text(i, gamma + 0.005, f'{gamma:.3f}', ha='center', va='bottom', 
					fontsize=10, fontweight='bold')
	
	plt.tight_layout()
	plt.savefig('fig/fig5.pdf', dpi=300, bbox_inches='tight')


def main():
	"""Main analysis function"""
	print("Loading data...")
	blocks_df = load_blocks()
	
	print("Identifying state transitions...")
	states_df = identify_state_transitions(blocks_df.copy())
	
	print("Calculating weekly gamma values with 0' state estimation...")
	weekly_df = calculate_weekly_gamma_with_0_prime_estimation(states_df.copy(), blocks_df.copy())

	# Print summary statistics
	print("=== Weekly Summary Statistics ===")
	print(f"Total weeks analyzed: {len(weekly_df)}")
	print(f"Average gamma rate: {weekly_df['gamma'].mean():.6f} ({weekly_df['gamma'].mean()*100:.4f}%)")
	print(f"Total estimated 0' states: {weekly_df['estimated_0_prime_count'].sum()}")
	print(f"Average 0' states per week: {weekly_df['estimated_0_prime_count'].mean():.2f}")
	print(f"Max 0' states in a week: {weekly_df['estimated_0_prime_count'].max()}")
	print(f"Weeks with 0' states: {len(weekly_df[weekly_df['estimated_0_prime_count'] > 0])}")
	
	# Save results
	weekly_df.to_csv('qubic_gamma_weekly_analysis.csv', index=False)
	
	print("\nCreating weekly dual-axis chart...")
	create_weekly_dual_axis_chart(weekly_df)
	
	print("Analysis complete!")


if __name__ == '__main__':
	main()