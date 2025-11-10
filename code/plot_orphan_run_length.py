import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

ALL_BLOCKS_PATH = "data/all_blocks.csv"
OUT_CSV = "data/qubic_orphan_start_qubic_continuous_split_with_len.csv"
FIG_PATH = "fig/orphan_run_length.pdf"
MIN_ORPHAN_LEN = 1

def load_all_blocks(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["height"] = pd.to_numeric(df["height"], errors="raise").astype(int)
    df["is_orphan"] = df["is_orphan"].astype(bool)
    df["is_qubic"] = df["is_qubic"].astype(bool)
    return df

def build_mainchain_maps(df: pd.DataFrame):
    main = df[~df["is_orphan"]].sort_values(["height", "timestamp"])
    first_main = main.groupby("height").first()
    main_heights = set(first_main.index.astype(int).tolist())
    main_qubic_heights = set(first_main.index[first_main["is_qubic"] == True].astype(int).tolist())
    main_ts = first_main["timestamp"].to_dict()
    return main_heights, main_qubic_heights, main_ts

def consecutive_runs(sorted_heights: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    if sorted_heights.size == 0:
        return runs
    s = int(sorted_heights[0])
    p = int(sorted_heights[0])
    for h in sorted_heights[1:]:
        h = int(h)
        if h == p + 1:
            p = h
        else:
            runs.append((s, p))
            s = h
            p = h
    runs.append((s, p))
    return runs

def qubic_runlength_from_start(start_h: int, main_qubic_heights: set) -> Tuple[int, Optional[int], Optional[int]]:
    qlen = 0
    q_first = None
    q_last = None
    h = int(start_h)
    while h in main_qubic_heights:
        if q_first is None:
            q_first = h
        q_last = h
        qlen += 1
        h += 1
    return qlen, q_first, q_last

def compute_runs(df: pd.DataFrame, min_orphan_len: int = MIN_ORPHAN_LEN) -> pd.DataFrame:
    main_heights, main_qubic_heights, main_ts = build_mainchain_maps(df)

    orphan_df = df[df["is_orphan"]].copy().sort_values(["height", "timestamp"])
    orphan_heights = np.sort(orphan_df["height"].unique().astype(int))
    orphan_runs = consecutive_runs(orphan_heights)  # [(o_start, o_end), ...] 오름차순

    rows = []
    for idx, (o_start, o_end) in enumerate(orphan_runs):
        o_len = o_end - o_start + 1
        if o_len < min_orphan_len:
            continue

        # 해당 orphan 구간의 모든 height가 mainchain qubic으로 대체되었는지 확인
        rng = range(o_start, o_end + 1)
        if not all(h in main_heights for h in rng):
            continue
        if not all(h in main_qubic_heights for h in rng):
            continue

        # orphan 구간의 타임스탬프/개수 집계
        o_slice = orphan_df[(orphan_df["height"] >= o_start) & (orphan_df["height"] <= o_end)]
        o_min_ts = o_slice["timestamp"].min() if not o_slice.empty else pd.NaT
        o_max_ts = o_slice["timestamp"].max() if not o_slice.empty else pd.NaT
        total_orphans_all = int(len(o_slice))
        total_orphans_qubic = int((o_slice["is_qubic"] == True).sum())

        # 1) orphan 시작에서의 qubic 연속 길이(무한 확장)
        q_len_inf, q_first, q_last_inf = qubic_runlength_from_start(o_start, main_qubic_heights)

        # 2) 다음 orphan이 나타나면 그 직전에서 컷
        next_o_start = orphan_runs[idx + 1][0] if idx + 1 < len(orphan_runs) else None
        if next_o_start is not None:
            cap_len = max(0, next_o_start - o_start)  # [o_start, next_o_start) 길이
            q_len = min(q_len_inf, cap_len)
            q_last = o_start + q_len - 1 if q_len > 0 else None
        else:
            q_len = q_len_inf
            q_last = q_last_inf

        q_start_ts = main_ts.get(q_first, pd.NaT) if q_first is not None else pd.NaT
        q_end_ts   = main_ts.get(q_last,  pd.NaT) if q_last  is not None else pd.NaT

        rows.append({
            "start_height": o_start,
            "end_height": o_end,
            "orphan_len": o_len,
            "total_orphans_on_run": total_orphans_qubic,   # qubic orphan만 카운트 유지
            "total_qubic_orphans_on_run": total_orphans_qubic,
            "length_qubic_run": q_len,                     # ← 다음 orphan에서 컷된 길이
            "qubic_start_height": q_first if q_first is not None else np.nan,
            "qubic_end_height":   q_last  if q_last  is not None else np.nan,
            "orphan_start_time":  o_min_ts,
            "orphan_end_time":    o_max_ts,
            "qubic_start_time":   q_start_ts,
            "qubic_end_time":     q_end_ts,
        })

    runs = pd.DataFrame(rows).sort_values(["start_height"]).reset_index(drop=True)
    return runs


def create_visualizations(res: pd.DataFrame, fig_path: str = 'fig/orphan_run_length.pdf'):
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plot_df = res[["length_qubic_run", "orphan_len"]].dropna().copy()
    if plot_df.empty:
        print("Nothing to plot.")
        return
    fig, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    ax2.scatter(plot_df['length_qubic_run'], plot_df['orphan_len'], alpha=0.6, color='red', s=30)
    ax2.set_xlabel('Qubic Run Length', fontsize=14)
    ax2.set_ylabel('Orphan Run Length', fontsize=14)
    ax2.grid(True, alpha=0.3)
    max_x = int(plot_df['length_qubic_run'].max())
    max_y = int(plot_df['orphan_len'].max())
    ax2.set_xticks(range(1, max_x + 1))
    ax2.set_yticks(range(1, max_y + 1))
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight')
    plt.show()

def main():
    print(f"Loading {ALL_BLOCKS_PATH} ...")
    df = load_all_blocks(ALL_BLOCKS_PATH)
    print("Computing orphan runs and qubic run length from orphan START height...")
    runs = compute_runs(df, MIN_ORPHAN_LEN)
    print(f"  Found {len(runs)} runs")
    if not runs.empty:
        print(f"  Max orphan_len: {runs['orphan_len'].max()}, Max length_qubic_run: {runs['length_qubic_run'].max()}")
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    runs.to_csv(OUT_CSV, index=False)
    print(f"Saved CSV -> {OUT_CSV}")
    print("Creating visualization...")
    create_visualizations(runs, FIG_PATH)
    print(f"Saved figure -> {FIG_PATH}")

if __name__ == "__main__":
    main()
