import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta


def filter_inactive_periods(df, all_blocks_df, window_hours=12, min_orphans_per_window=3):
    """
    Filter out periods where Qubic orphan blocks are insufficient over consecutive 12-hour windows.
    Uses a sliding window to identify inactive selfish-mining periods.
    All timestamps are kept UTC-aware to avoid tz-naive/aware comparison errors.
    """
    # Ensure input df ('start_ts') is sorted and UTC-aware upstream (load_data does this)
    df = df.sort_values('start_ts').reset_index(drop=True)

    # Extract Qubic orphan blocks from all_blocks.csv and make timestamps UTC-aware
    qubic_orphan_blocks = all_blocks_df[
        (all_blocks_df['is_qubic'] == True) &
        (all_blocks_df['is_orphan'] == True)
    ].copy()
    qubic_orphan_blocks['timestamp'] = pd.to_datetime(qubic_orphan_blocks['timestamp'], utc=True)

    # Active mask per run & detailed results for logging
    active_mask = np.zeros(len(df), dtype=bool)
    results = []

    # Sliding-window counting
    for i in range(len(df)):
        run_date = df.iloc[i]['start_ts']                     # tz-aware (UTC)
        window_start = run_date - pd.Timedelta(hours=window_hours)
        window_end = run_date

        # Count Qubic orphan blocks in the (window_start, window_end] interval
        window_qubic_orphans = qubic_orphan_blocks[
            (qubic_orphan_blocks['timestamp'] > window_start) &
            (qubic_orphan_blocks['timestamp'] <= window_end)
        ]
        qubic_orphan_count = len(window_qubic_orphans)

        is_active = qubic_orphan_count >= min_orphans_per_window
        active_mask[i] = is_active

        last_orphan_time = window_qubic_orphans['timestamp'].max() if qubic_orphan_count > 0 else None
        time_since_last_orphan = (
            (run_date - last_orphan_time).total_seconds() / 3600
            if last_orphan_time is not None else None
        )

        results.append({
            'date': run_date,
            'count': qubic_orphan_count,
            'window_start': window_start,
            'window_end': window_end,
            'last_orphan_time': last_orphan_time,
            'hours_since_last_orphan': time_since_last_orphan,
            'status': "VALID" if is_active else "EXCLUDED",
        })

    # Compressed log printing (avoid tz conversion issues by converting only for display)
    def _fmt(ts, with_time=True):
        if ts is None or pd.isna(ts):
            return "N/A"
        # display-only conversion; keep internal data tz-aware
        s = ts.tz_convert('UTC')
        return s.strftime('%Y-%m-%d %H:%M') if with_time else s.strftime('%Y-%m-%d')

    print(f"\n  Detailed exclusion log (min {min_orphans_per_window} Qubic orphan blocks per {window_hours}h window):")
    print(f"  {'Date':<12} {'Qubic Orphan':<15} {'Hours Since':<12} {'Window Start':<20} {'Window End':<20} {'Status':<10}")
    print(f"  {'-'*12} {'-'*15} {'-'*12} {'-'*20} {'-'*20} {'-'*10}")

    i = 0
    while i < len(results):
        current_status = results[i]['status']
        start_idx = i
        while i < len(results) and results[i]['status'] == current_status:
            i += 1
        end_idx = i - 1

        # First entry
        r = results[start_idx]
        hours_since = f"{r['hours_since_last_orphan']:.1f}" if r['hours_since_last_orphan'] is not None else "N/A"
        print(f"  {_fmt(r['date'], with_time=False):<12} {r['count']:<15} {hours_since:<12} "
              f"{_fmt(r['window_start']):<20} {_fmt(r['window_end']):<20} {r['status']:<10}")

        # Last entry if span > 1
        if end_idx > start_idx:
            r_end = results[end_idx]
            if end_idx > start_idx + 1:
                print(f"  ... ({end_idx - start_idx - 1} entries skipped) ...")
            hours_since_end = f"{r_end['hours_since_last_orphan']:.1f}" if r_end['hours_since_last_orphan'] is not None else "N/A"
            print(f"  {_fmt(r_end['date'], with_time=False):<12} {r_end['count']:<15} {hours_since_end:<12} "
                  f"{_fmt(r_end['window_start']):<20} {_fmt(r_end['window_end']):<20} {r_end['status']:<10}")

    print()
    return active_mask, results


def _to_utc_aware(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)

def load_data(min_orphans_per_12h=3):

    df = pd.read_csv("data/selfish_mining_blocks.csv")

    if "start_ts" in df.columns:
        df["start_ts"] = _to_utc_aware(df["start_ts"])
    if "end_ts" in df.columns:
        df["end_ts"] = _to_utc_aware(df["end_ts"])

    df = df.sort_values("start_ts").reset_index(drop=True)

    print("  Loading all_blocks.csv to count Qubic orphan blocks...")
    all_blocks_df = pd.read_csv("data/all_blocks.csv")

    all_blocks_df["timestamp"] = _to_utc_aware(all_blocks_df["timestamp"])

    original_count = len(df)

    print(f"  Identifying active selfish mining periods (12-hour sliding window, min {min_orphans_per_12h} Qubic orphan blocks)...")
    active_mask, filter_results = filter_inactive_periods(
        df,
        all_blocks_df,
        window_hours=12,
        min_orphans_per_window=min_orphans_per_12h,
    )

    df = df[active_mask].copy().reset_index(drop=True)
    filtered_count = len(df)

    print(f"  Summary: Filtered out {original_count - filtered_count} runs in inactive periods")
    print(f"  (12-hour window with < {min_orphans_per_12h} Qubic orphan blocks)")
    print(f"  Remaining runs for analysis: {filtered_count}")

    if len(df) == 0:
        raise ValueError("No runs with sufficient Qubic orphan blocks in 12-hour windows found!")

    df["orphans_per_length"] = df["total_orphans_on_run"] / df["length_qubic_run"]
    df["orphans_per_length"] = df["orphans_per_length"].replace([np.inf, -np.inf], 0).fillna(0)

    return df, filter_results


def detect_change_points(data, feature_col, min_segment_size=5, penalty=10, min_orphans_per_window=2):
    """
    Detect change points using a simple statistical approach.
    Uses sliding window to find points where mean changes significantly.
    Only considers windows with sufficient orphan blocks (selfish mining activity).
    """
    values = data[feature_col].values
    orphan_counts = data['total_orphans_on_run'].values
    change_points = [0]  # Start with beginning
    
    i = min_segment_size
    while i < len(values) - min_segment_size:
        # Check if both windows have sufficient orphan blocks
        left_orphans = orphan_counts[i-min_segment_size:i].sum()
        right_orphans = orphan_counts[i:i+min_segment_size].sum()
        
        # Skip if either window has insufficient orphan activity
        if left_orphans < min_orphans_per_window or right_orphans < min_orphans_per_window:
            i += 1
            continue
        
        # Compare statistics of left and right windows
        left_window = values[i-min_segment_size:i]
        right_window = values[i:i+min_segment_size]
        
        # Mann-Whitney U test for distribution difference
        if len(left_window) > 0 and len(right_window) > 0:
            try:
                statistic, p_value = stats.mannwhitneyu(left_window, right_window, alternative='two-sided')
                
                # If significant difference, mark as change point
                if p_value < 0.05:  # Significant change
                    change_points.append(i)
                    i += min_segment_size  # Skip ahead to avoid too many close points
                else:
                    i += 1
            except:
                i += 1
        else:
            i += 1
    
    change_points.append(len(values))  # End with last point
    return sorted(list(set(change_points)))


def compute_segment_statistics(data, start_idx, end_idx):
    """Compute comprehensive statistics for a segment, focusing on orphan/run length ratio."""
    segment = data.iloc[start_idx:end_idx]
    
    if len(segment) == 0:
        return None
    
    # Compute linear regression for orphans_per_length over time (1차 함수)
    segment_sorted = segment.sort_values('start_ts')
    time_numeric = (segment_sorted['start_ts'] - segment_sorted['start_ts'].min()).dt.total_seconds() / 86400  # days
    ratio_values = segment_sorted['orphans_per_length'].values
    
    # Linear regression: ratio = slope * time + intercept
    if len(time_numeric) > 1 and len(ratio_values) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, ratio_values)
    else:
        slope, intercept, r_value, p_value, std_err = 0, ratio_values[0] if len(ratio_values) > 0 else 0, 0, 1, 0
    
    stats_dict = {
        'start_time': segment['start_ts'].min(),
        'end_time': segment['start_ts'].max(),
        'duration_days': (segment['start_ts'].max() - segment['start_ts'].min()).days + 1,
        'num_runs': len(segment),
        # Ratio statistics (primary focus)
        'mean_ratio': segment['orphans_per_length'].mean(),
        'median_ratio': segment['orphans_per_length'].median(),
        'std_ratio': segment['orphans_per_length'].std(),
        'min_ratio': segment['orphans_per_length'].min(),
        'max_ratio': segment['orphans_per_length'].max(),
        # Linear trend (1차 함수)
        'ratio_slope': slope,  # 변화율 (양수면 증가, 음수면 감소)
        'ratio_intercept': intercept,  # 시작점
        'ratio_r_squared': r_value ** 2,  # 선형성 정도
        'ratio_p_value': p_value,  # 선형 관계의 유의성
        # Supporting statistics
        'mean_run_length': segment['length_qubic_run'].mean(),
        'mean_orphans': segment['total_orphans_on_run'].mean(),
    }
    
    return stats_dict


def compute_similarity(stat1, stat2):
    """Compute similarity score between two segments based on ratio pattern."""
    if stat1 is None or stat2 is None:
        return 0.0
    
    # Focus on ratio pattern (1차 함수 특성)
    features = [
        'mean_ratio',  # 평균 비율
        'ratio_slope',  # 변화율 (1차 함수의 기울기)
        'std_ratio',  # 비율의 변동성
    ]
    
    similarity = 0.0
    for feat in features:
        val1 = stat1.get(feat, 0)
        val2 = stat2.get(feat, 0)
        
        if val1 == 0 and val2 == 0:
            similarity += 1.0
        elif val1 == 0 or val2 == 0:
            similarity += 0.0
        else:
            # Use relative difference
            diff = abs(val1 - val2) / max(abs(val1), abs(val2), 0.001)  # Avoid division by zero
            similarity += max(0, 1 - diff)
    
    return similarity / len(features)


def merge_similar_segments(segments, similarity_threshold=0.7, min_segment_size=3):
    """Merge segments that are statistically similar."""
    if len(segments) <= 1:
        return segments
    
    merged = []
    i = 0
    
    while i < len(segments):
        current = segments[i]
        j = i + 1
        
        # Try to merge with following segments
        while j < len(segments):
            next_seg = segments[j]
            similarity = compute_similarity(current['stats'], next_seg['stats'])
            
            if similarity >= similarity_threshold:
                # Merge segments
                current['end_idx'] = next_seg['end_idx']
                current['stats'] = compute_segment_statistics(
                    current['data'], current['start_idx'], current['end_idx']
                )
                j += 1
            else:
                break
        
        merged.append(current)
        i = j
    
    # Filter out segments that are too small
    filtered = [s for s in merged if s['stats']['num_runs'] >= min_segment_size]
    
    return filtered


def create_period_visualizations_by_status(data, filter_results):
    """
    Create scatter plots for VALID periods only.
    Similar to plot_orphan_run_length.py style.
    """
    # Group consecutive periods by status
    periods = []
    i = 0
    while i < len(filter_results):
        current_status = filter_results[i]['status']
        start_date = filter_results[i]['date']
        start_idx = i
        
        # Find consecutive same status
        while i < len(filter_results) and filter_results[i]['status'] == current_status:
            i += 1
        end_idx = i - 1
        end_date = filter_results[end_idx]['date']
        
        periods.append({
            'status': current_status,
            'start_date': start_date,
            'end_date': end_date,
            'start_idx': start_idx,
            'end_idx': end_idx
        })
    
    # Only VALID periods
    valid_periods = [p for p in periods if p['status'] == 'VALID']
    
    # Filter out VALID periods with total orphan count <= 5
    filtered_valid_periods = []
    for period in valid_periods:
        # Get all runs within this period
        period_runs = data[
            (data['start_ts'] >= period['start_date']) & 
            (data['start_ts'] <= period['end_date'])
        ]
        
        # Calculate total orphan count in this period
        # Sum of total_orphans_on_run for all runs in the period
        total_orphans = period_runs['total_orphans_on_run'].sum() if len(period_runs) > 0 else 0
        
        # Only include periods with more than 5 total orphans
        if total_orphans > 5:
            period['total_orphans'] = total_orphans
            filtered_valid_periods.append(period)
        else:
            print(f"  Excluding VALID period {period['start_date'].strftime('%Y-%m-%d')} to {period['end_date'].strftime('%Y-%m-%d')}: only {total_orphans} total orphans (<= 5)")
    
    # Create visualizations for filtered VALID periods only
    if len(filtered_valid_periods) > 0:
        print("\n" + "=" * 80)
        print("VISUALIZING VALID PERIODS (with > 5 total orphans)")
        print("=" * 80)
        print(f"  Showing {len(filtered_valid_periods)} out of {len(valid_periods)} VALID periods")
        create_period_group_visualization(data, filtered_valid_periods, "VALID")
    else:
        print("\n" + "=" * 80)
        print("VISUALIZING VALID PERIODS")
        print("=" * 80)
        print("  No VALID periods with > 5 total orphans found.")


def create_period_group_visualization(data, periods, status_label):
    """
    Create scatter plots for a group of periods (EXCLUDED or VALID).
    """
    import math
    
    num_periods = len(periods)
    if num_periods == 0:
        return
    
    # Determine grid size dynamically
    if num_periods == 1:
        rows, cols = 1, 1
    elif num_periods == 2:
        rows, cols = 1, 2
    elif num_periods <= 4:
        rows, cols = 2, 2
    elif num_periods <= 6:
        rows, cols = 2, 3
    elif num_periods <= 9:
        rows, cols = 3, 3
    elif num_periods <= 12:
        rows, cols = 3, 4
    elif num_periods <= 16:
        rows, cols = 4, 4
    elif num_periods <= 20:
        rows, cols = 4, 5
    elif num_periods <= 25:
        rows, cols = 5, 5
    else:
        # For more than 25 periods, use a more flexible approach
        cols = math.ceil(math.sqrt(num_periods))
        rows = math.ceil(num_periods / cols)
    
    # First pass: collect all data to determine global axis limits
    all_period_runs = []
    for period in periods:
        period_runs = data[
            (data['start_ts'] >= period['start_date']) & 
            (data['start_ts'] <= period['end_date'])
        ]
        if len(period_runs) > 0:
            all_period_runs.append(period_runs)
    
    # Determine global max x and y values for consistent axis limits
    if len(all_period_runs) > 0:
        all_runs = pd.concat(all_period_runs, ignore_index=True)
        global_max_x = int(all_runs['length_qubic_run'].max()) if len(all_runs) > 0 else 1
        global_max_y = int(all_runs['total_orphans_on_run'].max()) if len(all_runs) > 0 else 1
    else:
        global_max_x = 1
        global_max_y = 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    if num_periods == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, period in enumerate(periods):
        ax = axes[idx]
        
        # Filter runs that start within this period
        period_runs = data[
            (data['start_ts'] >= period['start_date']) & 
            (data['start_ts'] <= period['end_date'])
        ]
        
        if len(period_runs) > 0:
            # Count frequency of each (x, y) coordinate pair
            coord_counts = period_runs.groupby(['length_qubic_run', 'total_orphans_on_run']).size().reset_index(name='count')
            
            # Set explicit vmin and vmax to match actual data range (prevents white band at top)
            vmin = coord_counts['count'].min()
            vmax = coord_counts['count'].max()
            
            # Create scatter plot with size and color based on frequency
            # Use norm to explicitly set the range to match data exactly
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            scatter = ax.scatter(coord_counts['length_qubic_run'], coord_counts['total_orphans_on_run'],
                              s=coord_counts['count'] * 50,  # Size proportional to frequency
                              c=coord_counts['count'],  # Color based on frequency
                              cmap='Reds',  # Red color map (darker = more frequent)
                              norm=norm,  # Explicit normalization matching data range exactly
              alpha=0.7, edgecolors='black', linewidths=0.5)
            
            # Add colorbar to show frequency scale
            # Create a ScalarMappable with exact range to ensure colorbar matches exactly
            from matplotlib import cm
            sm = cm.ScalarMappable(norm=norm, cmap='Reds')
            sm.set_array([])  # Empty array, we just need the mappable for colorbar
            
            cbar = plt.colorbar(sm, ax=ax, extend='neither')
            cbar.set_label('Frequency (number of runs)', fontsize=10)
            
            # Ensure colorbar image covers exactly the data range
            # Set both the mappable limits and the axis limits
            cbar.mappable.set_clim(vmin=vmin, vmax=vmax)
            cbar.ax.set_ylim(vmin, vmax)
            
            # Filter ticks to show only integers (remove decimal ticks while maintaining natural spacing)
            current_ticks = cbar.get_ticks()
            integer_ticks = [tick for tick in current_ticks if tick == int(tick) and vmin <= tick <= vmax]
            if len(integer_ticks) > 0:
                cbar.set_ticks(integer_ticks)
            # Format colorbar labels to show only integers
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
        else:
            ax.text(0.5, 0.5, 'No data in this period', 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Set consistent axis limits for all subplots
        ax.set_xlim(0.5, global_max_x + 0.5)
        ax.set_ylim(-0.5, global_max_y + 0.5)
        ax.set_xticks(range(1, global_max_x + 1))
        ax.set_yticks(range(0, global_max_y + 1))
        # Format y-axis to show only integers
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
        
        # Add ideal selfish mining reference line: y = x - 1
        # Calculate x range: from 1 to min(global_max_x, global_max_y + 1)
        x_line_max = min(global_max_x, global_max_y + 1)
        if x_line_max >= 1:
            x_line = np.arange(1, x_line_max + 1)
            y_line = x_line - 1
            ax.plot(x_line, y_line, 'b--', linewidth=2, alpha=0.7, label='Ideal (y=x-1)')
        
        ax.set_xlabel('Qubic Run Length', fontsize=12)
        ax.set_ylabel('Total Orphans in Run', fontsize=12)
        
        # Include hour in date format for precise period boundaries
        period_label = f"{period['start_date'].strftime('%Y-%m-%d %H:%M')} to {period['end_date'].strftime('%Y-%m-%d %H:%M')}"
        
        # Calculate or get total orphans for this period
        if 'total_orphans' in period:
            total_orphans = period['total_orphans']
        else:
            total_orphans = period_runs['total_orphans_on_run'].sum() if len(period_runs) > 0 else 0
        
        ax.set_title(f'Selfish mining period {idx+1} ({total_orphans} cases)\n{period_label}', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_periods, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save to PDF in fig folder
    os.makedirs('fig', exist_ok=True)
    filename = f'fig/orphan_run_length_{status_label.lower()}_periods.pdf'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"  Saved visualization to {filename}")
    
    plt.show()
    plt.close(fig)


def validate_segments(segments, min_runs=3, min_orphans_per_period=5):
    """
    Validate and filter segments based on minimum requirements.
    Note: Duration-based filtering is removed since inactive periods are already filtered.
    """
    validated = []
    
    for seg in segments:
        stats = seg['stats']
        segment_data = seg['data'].iloc[seg['start_idx']:seg['end_idx']]
        total_orphans = segment_data['total_orphans_on_run'].sum()
        
        # Check requirements (duration check removed - already filtered by 24h window)
        if (stats['num_runs'] >= min_runs and
            total_orphans >= min_orphans_per_period):
            validated.append(seg)
        else:
            # Log why segment was excluded
            reasons = []
            if stats['num_runs'] < min_runs:
                reasons.append(f"runs < {min_runs}")
            if total_orphans < min_orphans_per_period:
                reasons.append(f"total orphans < {min_orphans_per_period}")
            print(f"  Excluded segment {stats['start_time'].strftime('%Y-%m-%d')}: {', '.join(reasons)}")
    
    return validated


def analyze_periods():
    """Main analysis function using hybrid approach."""
    print("=" * 80)
    print("PERIOD ANALYSIS: Hybrid Change Point Detection + Statistical Clustering")
    print("=" * 80)
    print()
    
    # Step 1: Load data (filter out periods with insufficient Qubic orphan blocks in 12h windows)
    print("Step 1: Loading data...")
    print("  Filtering out periods with insufficient Qubic orphan blocks in 12-hour windows...")
    # Use 3 Qubic orphan blocks as threshold for 12-hour window
    min_orphans_threshold = 3
    data, filter_results = load_data(min_orphans_per_12h=min_orphans_threshold)
    print(f"  Loaded {len(data)} runs in active selfish mining periods")
    print(f"  Date range: {data['start_ts'].min()} to {data['start_ts'].max()}")
    print()
    
    # Load original data for visualization
    original_data = pd.read_csv('data/selfish_mining_blocks.csv')
    original_data['start_ts'] = pd.to_datetime(original_data['start_ts'], utc=True)
    
    # Create visualizations for EXCLUDED and VALID periods
    create_period_visualizations_by_status(original_data, filter_results)
    
    if len(data) < 10:
        print("ERROR: Not enough data for period analysis (need at least 10 runs)")
        return
    
    # Step 2: Change Point Detection based on orphan/run length ratio
    print("Step 2: Change Point Detection...")
    print("  Detecting change points based on orphan/run length ratio pattern...")
    print("  Only considering windows with sufficient orphan activity...")
    cp_ratio = detect_change_points(data, 'orphans_per_length', min_segment_size=5, penalty=10, min_orphans_per_window=30)
    
    all_cp = sorted(list(set(cp_ratio)))
    print(f"  Found {len(all_cp)-1} potential segments from change point detection")
    print(f"  Change points (indices): {all_cp}")
    print()
    
    # Step 3: Compute segment statistics
    print("Step 3: Computing segment statistics...")
    segments = []
    for i in range(len(all_cp) - 1):
        start_idx = all_cp[i]
        end_idx = all_cp[i + 1]
        segment_data = data.iloc[start_idx:end_idx]
        
        stats = compute_segment_statistics(data, start_idx, end_idx)
        if stats:
            segments.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'data': data,
                'stats': stats
            })
    
    print(f"  Computed statistics for {len(segments)} segments")
    print()
    
    # Step 4: Merge similar segments
    print("Step 4: Merging statistically similar segments...")
    print("  Similarity threshold: 0.7")
    merged_segments = merge_similar_segments(segments, similarity_threshold=0.7, min_segment_size=3)
    print(f"  Merged to {len(merged_segments)} segments")
    print()
    
    # Step 5: Validate segments (exclude periods with insufficient orphan activity)
    print("Step 5: Validating segments...")
    print("  Minimum requirements: 3 runs, 5 total orphan blocks")
    print("  (Inactive periods already filtered by 24-hour window check)")
    validated_segments = validate_segments(merged_segments, min_runs=3, min_orphans_per_period=5)
    print(f"  Validated {len(validated_segments)} segments with sufficient orphan activity")
    print()
    
    # Step 6: Generate detailed report
    print("=" * 80)
    print("DETAILED ANALYSIS REPORT")
    print("=" * 80)
    print()
    
    print(f"Total periods identified: {len(validated_segments)}")
    print()
    
    for idx, seg in enumerate(validated_segments, 1):
        stats = seg['stats']
        print(f"Period {idx}:")
        print(f"  Time Range: {stats['start_time'].strftime('%Y-%m-%d')} to {stats['end_time'].strftime('%Y-%m-%d')}")
        print(f"  Duration: {stats['duration_days']} days")
        print(f"  Number of Runs: {stats['num_runs']}")
        print()
        print("  Orphan/Run Length Ratio Statistics (Primary Focus):")
        print(f"    Mean Ratio: {stats['mean_ratio']:.3f}")
        print(f"    Median Ratio: {stats['median_ratio']:.3f}")
        print(f"    Std Dev: {stats['std_ratio']:.3f}")
        print(f"    Range: [{stats['min_ratio']:.3f}, {stats['max_ratio']:.3f}]")
        print()
        print("  Ratio Trend (1차 함수 패턴):")
        print(f"    Slope (변화율): {stats['ratio_slope']:.6f} {'(증가)' if stats['ratio_slope'] > 0 else '(감소)' if stats['ratio_slope'] < 0 else '(일정)'}")
        print(f"    Intercept (시작점): {stats['ratio_intercept']:.3f}")
        print(f"    R² (선형성): {stats['ratio_r_squared']:.3f}")
        print(f"    P-value (유의성): {stats['ratio_p_value']:.4f} {'(유의함)' if stats['ratio_p_value'] < 0.05 else '(유의하지 않음)'}")
        print()
        # Calculate total orphans in this period
        segment_data = seg['data'].iloc[seg['start_idx']:seg['end_idx']]
        total_orphans = segment_data['total_orphans_on_run'].sum()
        
        print("  Supporting Statistics:")
        print(f"    Mean Run Length: {stats['mean_run_length']:.2f}")
        print(f"    Mean Orphan Count: {stats['mean_orphans']:.2f}")
        print(f"    Total Orphan Blocks: {total_orphans}")
        print()
        
        # Compare with overall statistics
        overall_mean_ratio = data['orphans_per_length'].mean()
        ratio_diff = ((stats['mean_ratio'] - overall_mean_ratio) / overall_mean_ratio) * 100 if overall_mean_ratio > 0 else 0
        
        print("  Comparison to Overall Average:")
        print(f"    Orphan/Run Length Ratio: {ratio_diff:+.1f}% {'(above)' if ratio_diff > 0 else '(below)'}")
        print()
        print("-" * 80)
        print()
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()
    
    if len(validated_segments) > 0:
        period_durations = [s['stats']['duration_days'] for s in validated_segments]
        period_runs = [s['stats']['num_runs'] for s in validated_segments]
        period_mean_ratios = [s['stats']['mean_ratio'] for s in validated_segments]
        period_slopes = [s['stats']['ratio_slope'] for s in validated_segments]
        
        print(f"Period Count: {len(validated_segments)}")
        print(f"Total Coverage: {sum(period_durations)} days")
        print()
        print("Period Duration Statistics:")
        print(f"  Mean: {np.mean(period_durations):.1f} days")
        print(f"  Median: {np.median(period_durations):.1f} days")
        print(f"  Range: [{min(period_durations)}, {max(period_durations)}] days")
        print()
        print("Runs per Period Statistics:")
        print(f"  Mean: {np.mean(period_runs):.1f} runs")
        print(f"  Median: {np.median(period_runs):.1f} runs")
        print(f"  Range: [{min(period_runs)}, {max(period_runs)}] runs")
        print()
        print("Orphan/Run Length Ratio Variation Across Periods:")
        print(f"  Mean: {np.mean(period_mean_ratios):.3f}")
        print(f"  Std Dev: {np.std(period_mean_ratios):.3f}")
        print(f"  Range: [{min(period_mean_ratios):.3f}, {max(period_mean_ratios):.3f}]")
        print()
        print("Ratio Trend Slopes (1차 함수 기울기) Across Periods:")
        print(f"  Mean: {np.mean(period_slopes):.6f}")
        print(f"  Std Dev: {np.std(period_slopes):.6f}")
        print(f"  Range: [{min(period_slopes):.6f}, {max(period_slopes):.6f}]")
        print(f"  Positive slopes (증가): {sum(1 for s in period_slopes if s > 0)} periods")
        print(f"  Negative slopes (감소): {sum(1 for s in period_slopes if s < 0)} periods")
        print(f"  Zero slopes (일정): {sum(1 for s in period_slopes if abs(s) < 1e-6)} periods")
        print()
        
        # Period differences
        if len(validated_segments) > 1:
            print("Period-to-Period Ratio Pattern Changes:")
            for i in range(len(validated_segments) - 1):
                curr = validated_segments[i]['stats']
                next_seg = validated_segments[i + 1]['stats']
                
                ratio_change = ((next_seg['mean_ratio'] - curr['mean_ratio']) / 
                               curr['mean_ratio']) * 100 if curr['mean_ratio'] > 0 else 0
                slope_change = next_seg['ratio_slope'] - curr['ratio_slope']
                
                print(f"  Period {i+1} -> Period {i+2}:")
                print(f"    Mean Ratio Change: {ratio_change:+.1f}%")
                print(f"    Slope Change: {slope_change:+.6f} {'(더 가파름)' if abs(slope_change) > 0.0001 else '(비슷함)'}")
                print(f"    Pattern: {curr['mean_ratio']:.3f} (slope={curr['ratio_slope']:.6f}) -> {next_seg['mean_ratio']:.3f} (slope={next_seg['ratio_slope']:.6f})")
            print()
    
    print("=" * 80)
    print("Analysis Complete")
    print("=" * 80)
    
    return validated_segments


def verify_orphan_counting():
    """
    Verify if orphan blocks are being counted multiple times across different runs.
    Check if the same orphan block appears in multiple runs.
    """
    print("\n" + "=" * 80)
    print("VERIFYING ORPHAN COUNTING METHODOLOGY")
    print("=" * 80)
    
    # Load run data
    df = pd.read_csv('data/selfish_mining_blocks.csv')
    print(f"\nTotal runs in dataset: {len(df)}")
    
    # Load all blocks to get actual orphan blocks
    all_blocks_df = pd.read_csv('data/all_blocks.csv')
    all_blocks_df['timestamp'] = pd.to_datetime(all_blocks_df['timestamp'])
    
    # Get Qubic orphan blocks
    qubic_orphan_blocks = all_blocks_df[
        (all_blocks_df['is_qubic'] == True) & 
        (all_blocks_df['is_orphan'] == True)
    ].copy()
    qubic_orphan_blocks = qubic_orphan_blocks.sort_values('height').reset_index(drop=True)
    
    print(f"Total Qubic orphan blocks: {len(qubic_orphan_blocks)}")
    
    # Check if same orphan blocks appear in multiple runs
    print("\nChecking if same orphan blocks appear in multiple runs...")
    orphan_to_runs = {}  # Map orphan height to list of run indices
    
    for idx, run in df.iterrows():
        start_h = int(run['start_height'])
        end_h = int(run['end_height'])
        
        # Get orphan blocks in this height range
        orphans_in_range = qubic_orphan_blocks[
            (qubic_orphan_blocks['height'] >= start_h) & 
            (qubic_orphan_blocks['height'] <= end_h)
        ]
        
        for _, orphan in orphans_in_range.iterrows():
            orphan_h = int(orphan['height'])
            if orphan_h not in orphan_to_runs:
                orphan_to_runs[orphan_h] = []
            orphan_to_runs[orphan_h].append(idx)
    
    # Count orphans that appear in multiple runs
    multi_count_orphans = {h: runs for h, runs in orphan_to_runs.items() if len(runs) > 1}
    
    print(f"  Total unique orphan heights: {len(orphan_to_runs)}")
    print(f"  Orphan heights appearing in multiple runs: {len(multi_count_orphans)}")
    
    if len(multi_count_orphans) > 0:
        print(f"\n  ⚠️  WARNING: {len(multi_count_orphans)} orphan heights are counted in multiple runs!")
        print("\n  Example orphan blocks counted multiple times:")
        example_count = 0
        for orphan_h, run_indices in list(multi_count_orphans.items())[:5]:
            example_count += 1
            print(f"\n    Orphan at height {orphan_h} appears in {len(run_indices)} runs:")
            for run_idx in run_indices[:3]:
                run = df.iloc[run_idx]
                print(f"      Run {run_idx}: height {run['start_height']}-{run['end_height']}, "
                      f"orphans={run['total_orphans_on_run']}")
            if len(run_indices) > 3:
                print(f"      ... and {len(run_indices) - 3} more runs")
        
        # Calculate total duplicate count
        total_duplicate_count = sum(len(runs) - 1 for runs in multi_count_orphans.values())
        print(f"\n  Total duplicate counts: {total_duplicate_count} (each orphan counted {total_duplicate_count / len(multi_count_orphans):.1f} times on average)")
    else:
        print("\n  ✓ No orphan blocks are counted multiple times. Each orphan is counted exactly once.")
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    if len(multi_count_orphans) > 0:
        print("⚠️  ISSUE DETECTED: Same orphan blocks are being counted in multiple runs.")
        print("   This means the current methodology counts each orphan multiple times")
        print("   when it appears in overlapping runs.")
    else:
        print("✓ No issues detected: Each orphan block is counted only once.")
    print("=" * 80)


if __name__ == '__main__':
    # First verify the counting methodology
    verify_orphan_counting()
    
    segments = analyze_periods()

