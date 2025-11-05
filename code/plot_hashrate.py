import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# Font settings for better display
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def analyze_qubic_mining():
    """
    Analyze Qubic's Monero mining power share on a daily basis and generate graphs.
    """
    # Load CSV file
    print("Loading data...")
    df = pd.read_csv('data/all_blocks.csv')

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['week'] = df['timestamp'].dt.to_period('W-WED')
    df['hour'] = df['timestamp'].dt.floor('H')

    print(f"Total blocks: {len(df)}")
    print(f"Data period: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

    # Check Qubic-mined blocks and orphan blocks
    qubic_blocks = df[df['is_qubic'] == True]
    orphan_blocks = df[df['is_orphan'] == True]
    qubic_orphan_blocks = df[(df['is_qubic'] == True) & (df['is_orphan'] == True)]

    print(f"Qubic-mined blocks: {len(qubic_blocks)}")
    print(f"Orphan blocks: {len(orphan_blocks)}")
    print(f"Qubic-mined orphan blocks: {len(qubic_orphan_blocks)}")

    # Calculate overall statistics
    total_blocks_overall = len(df)
    qubic_blocks_overall = len(df[df['is_qubic'] == True])
    qubic_power_overall = (qubic_blocks_overall / total_blocks_overall) * 100 if total_blocks_overall > 0 else 0

    # Calculate daily statistics
    daily_stats = []
    for date in df['date'].unique():
        date_data = df[df['date'] == date]

        total_blocks = len(date_data)
        qubic_blocks_count = len(date_data[date_data['is_qubic'] == True])
        orphan_blocks_count = len(date_data[date_data['is_orphan'] == True])
        qubic_orphan_count = len(date_data[(date_data['is_qubic'] == True) & (date_data['is_orphan'] == True)])

        non_qubic_orphan_count = orphan_blocks_count - qubic_orphan_count
        non_qubic_regular_count = total_blocks - qubic_blocks_count - non_qubic_orphan_count
        qubic_power_ratio = (qubic_blocks_count / total_blocks) * 100 if total_blocks > 0 else 0

        daily_stats.append({
            'date': date,
            'total_blocks': total_blocks,
            'qubic_blocks': qubic_blocks_count,
            'orphan_blocks': orphan_blocks_count,
            'qubic_orphan_blocks': qubic_orphan_count,
            'non_qubic_orphan_blocks': non_qubic_orphan_count,
            'non_qubic_regular_blocks': non_qubic_regular_count,
            'qubic_power_ratio': qubic_power_ratio
        })

    # Calculate weekly statistics (W-WED)
    weekly_stats = []
    for week in df['week'].unique():
        week_data = df[df['week'] == week]
        total_blocks = len(week_data)
        qubic_blocks_count = len(week_data[week_data['is_qubic'] == True])
        qubic_power_ratio = (qubic_blocks_count / total_blocks) * 100 if total_blocks > 0 else 0
        weekly_stats.append({
            'week': week,
            'total_blocks': total_blocks,
            'qubic_blocks': qubic_blocks_count,
            'qubic_power_ratio': qubic_power_ratio
        })

    # Calculate hourly statistics
    hourly_stats = []
    for hour in df['hour'].unique():
        hour_data = df[df['hour'] == hour]
        total_blocks = len(hour_data)
        qubic_blocks_count = len(hour_data[hourly_data['is_qubic'] == True]) if (hourly_data := hour_data) is not None else 0
        qubic_power_ratio = (qubic_blocks_count / total_blocks) * 100 if total_blocks > 0 else 0
        hourly_stats.append({
            'hour': hour,
            'total_blocks': total_blocks,
            'qubic_blocks': qubic_blocks_count,
            'qubic_power_ratio': qubic_power_ratio
        })

    # Convert to DataFrames
    daily_df = pd.DataFrame(daily_stats)
    daily_df['date'] = pd.to_datetime(daily_df['date'])

    weekly_df = pd.DataFrame(weekly_stats)
    weekly_df['week_end'] = weekly_df['week'].dt.end_time

    hourly_df = pd.DataFrame(hourly_stats)
    hourly_df['hour'] = pd.to_datetime(hourly_df['hour'])

    # x-axis formatting logic (공유)
    total_days = len(daily_df)
    if total_days <= 30:
        interval = 3
    elif total_days <= 60:
        interval = 5
    elif total_days <= 90:
        interval = 7
    else:
        interval = 14

    date_min = daily_df['date'].min()
    date_max = daily_df['date'].max()
    padding = pd.Timedelta(days=2)

    fig1, ax1 = plt.subplots(1, 1, figsize=(15, 6))

    ax1.plot(daily_df['date'], daily_df['qubic_power_ratio'],
             marker='o', linewidth=2, markersize=4, color='red', label='Daily Mining Power Share')

    ax1.plot(weekly_df['week_end'], weekly_df['qubic_power_ratio'],
             marker='s', linewidth=2, markersize=6, color='blue', label='Weekly Mining Power Share')

    hourly_sample = hourly_df[hourly_df.index % 6 == 0]
    ax1.plot(hourly_sample['hour'], hourly_sample['qubic_power_ratio'],
             marker='^', linewidth=1, markersize=3, color='green', alpha=0.7, label='Hourly Mining Power Share')

    ax1.axhline(y=qubic_power_overall, color='black', linestyle='--', linewidth=2,
                label=f'Overall Average ({qubic_power_overall:.2f}%)')

    ax1.set_xlabel('Date', fontsize=14)
    ax1.set_ylabel('Share (%)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax1.set_xlim(date_min - padding, date_max + padding)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.WEDNESDAY, interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    fig1.tight_layout()
    fig1.savefig('fig/hashrate.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    fig2, ax2 = plt.subplots(1, 1, figsize=(15, 6))

    non_qubic_orphan_blocks = daily_df['non_qubic_orphan_blocks']
    qubic_orphan_blocks = daily_df['qubic_orphan_blocks']
    qubic_regular_blocks = daily_df['qubic_blocks'] - daily_df['qubic_orphan_blocks']
    non_qubic_regular_blocks = daily_df['non_qubic_regular_blocks']

    ax2.bar(daily_df['date'], non_qubic_orphan_blocks,
            alpha=0.6, label='Non-Qubic Orphan Blocks', color='orange')
    ax2.bar(daily_df['date'], qubic_orphan_blocks,
            bottom=non_qubic_orphan_blocks,
            alpha=0.9, label='Qubic Orphan Blocks', color='darkred')
    ax2.bar(daily_df['date'], qubic_regular_blocks,
            bottom=non_qubic_orphan_blocks + qubic_orphan_blocks,
            alpha=0.8, label='Qubic Regular Blocks', color='red')
    ax2.bar(daily_df['date'], non_qubic_regular_blocks,
            bottom=non_qubic_orphan_blocks + qubic_orphan_blocks + qubic_regular_blocks,
            alpha=0.7, label='Non-Qubic Regular Blocks', color='lightblue')

    ax2.set_xlabel('Date', fontsize=14)
    ax2.set_ylabel('Number of Blocks', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax2.set_xlim(date_min - padding, date_max + padding)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.WEDNESDAY, interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    fig2.tight_layout()
    fig2.savefig('fig/block_production.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig2)

    print("\n=== Analysis Results Summary ===")
    print(f"Total period: {daily_df['date'].min().strftime('%Y-%m-%d')} ~ {daily_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Average Qubic mining power share: {daily_df['qubic_power_ratio'].mean():.2f}%")
    print(f"Maximum Qubic mining power share: {daily_df['qubic_power_ratio'].max():.2f}%")
    print(f"Minimum Qubic mining power share: {daily_df['qubic_power_ratio'].min():.2f}%")
    print(f"Total Qubic blocks: {daily_df['qubic_blocks'].sum()}")
    print(f"Total blocks: {daily_df['total_blocks'].sum()}")
    print(f"Overall Qubic share: {(daily_df['qubic_blocks'].sum() / daily_df['total_blocks'].sum()) * 100:.2f}%")

    return daily_df

if __name__ == "__main__":
    daily_stats = analyze_qubic_mining()
