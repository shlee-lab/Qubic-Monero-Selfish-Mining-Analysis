"""
Remove duplicate runs from qubic_orphan_start_qubic_continuous_split.csv.
For runs with the same end_height, keep only the one with maximum length_qubic_run.
"""
import pandas as pd
import os

def deduplicate_runs():
    """
    Remove duplicate runs: for each end_height, keep only the run with maximum length_qubic_run.
    """
    input_file = 'data/qubic_orphan_start_qubic_continuous_split.csv'
    output_file = 'data/qubic_orphan_start_qubic_continuous_split.csv'
    
    print("=" * 80)
    print("DEDUPLICATING RUNS")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading {input_file}...")
    df = pd.read_csv(input_file)
    original_count = len(df)
    print(f"  Original run count: {original_count}")
    
    # Group by end_height and keep only the run with maximum length_qubic_run
    print("\nRemoving duplicates (keeping max length for each end_height)...")
    
    # For each end_height, keep the row with maximum length_qubic_run
    # If there are ties, keep the first one (by start_height)
    df_dedup = df.sort_values(['end_height', 'length_qubic_run', 'start_height'], 
                             ascending=[True, False, True])
    df_dedup = df_dedup.drop_duplicates(subset=['end_height'], keep='first')
    df_dedup = df_dedup.sort_values('start_height').reset_index(drop=True)
    
    removed_count = original_count - len(df_dedup)
    print(f"  Removed {removed_count} duplicate runs")
    print(f"  Remaining run count: {len(df_dedup)}")
    
    # Show some examples of removed duplicates
    print("\nExample of duplicate removal:")
    # Find an end_height that had multiple runs
    end_height_counts = df.groupby('end_height').size()
    duplicate_end_heights = end_height_counts[end_height_counts > 1].head(5)
    
    for end_h, count in duplicate_end_heights.items():
        runs_for_end = df[df['end_height'] == end_h].sort_values('length_qubic_run', ascending=False)
        kept = runs_for_end.iloc[0]
        removed = runs_for_end.iloc[1:]
        
        print(f"\n  end_height {int(end_h)}: {count} runs")
        print(f"    KEPT: start={kept['start_height']}, length={kept['length_qubic_run']}, orphans={kept['total_orphans_on_run']}")
        for idx, row in removed.iterrows():
            print(f"    REMOVED: start={row['start_height']}, length={row['length_qubic_run']}, orphans={row['total_orphans_on_run']}")
    
    # Save deduplicated data
    print(f"\nSaving deduplicated data to {output_file}...")
    df_dedup.to_csv(output_file, index=False)
    print(f"  Saved {len(df_dedup)} runs")
    
    print("\n" + "=" * 80)
    print("DEDUPLICATION COMPLETE")
    print("=" * 80)
    print(f"Removed {removed_count} duplicate runs ({removed_count/original_count*100:.1f}% reduction)")
    print(f"Original: {original_count} runs")
    print(f"After deduplication: {len(df_dedup)} runs")


if __name__ == '__main__':
    deduplicate_runs()

