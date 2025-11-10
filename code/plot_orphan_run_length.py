import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALL_BLOCKS_PATH = "data/all_blocks.csv"
FIG_PATH = "fig/orphan_run_length.pdf"
CSV_PATH = "data/selfish_mining_blocks.csv"

def _to_utc(s: pd.Series) -> pd.Series:
    try:
        return s.dt.tz_convert("UTC")
    except Exception:
        return s.dt.tz_localize("UTC")

def load_all_blocks(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["height","timestamp","is_orphan","is_qubic"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["height"] = pd.to_numeric(df["height"], errors="raise").astype(np.int64)
    df["is_orphan"] = df["is_orphan"].astype(bool)
    df["is_qubic"] = df["is_qubic"].astype(bool)
    return df.dropna(subset=["timestamp"])

def build_main(df: pd.DataFrame) -> pd.DataFrame:
    m = df.loc[~df["is_orphan"]].sort_values(["height","timestamp"])
    m = m.groupby("height", as_index=False).first()[["height","timestamp","is_qubic"]]
    m["timestamp"] = _to_utc(pd.to_datetime(m["timestamp"], errors="coerce"))
    return m.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

def build_orphan_by_height(df: pd.DataFrame) -> pd.DataFrame:
    o = df.loc[df["is_orphan"], ["height","timestamp"]].copy()
    o["timestamp"] = _to_utc(pd.to_datetime(o["timestamp"], errors="coerce"))
    o = o.sort_values(["height","timestamp"], kind="mergesort").groupby("height", as_index=False).first()
    return o.sort_values("height").reset_index(drop=True)

def compute_runs_consecutive(orphan_h: pd.DataFrame) -> pd.DataFrame:
    if orphan_h.empty:
        return orphan_h.assign(orphan_run_length=pd.Series(dtype=np.int64),
                               run_first_height=pd.Series(dtype=np.int64),
                               run_last_height=pd.Series(dtype=np.int64),
                               last_orphan_ts=pd.Series(dtype="datetime64[ns]")).iloc[0:0]
    h = orphan_h["height"].to_numpy(np.int64)
    diff = np.diff(h, prepend=h[0]-1)
    gid = np.cumsum((diff != 1).astype(np.int64))
    g = orphan_h.assign(gid=gid)
    runs = g.groupby("gid", as_index=False).agg(
        orphan_run_length=("height","size"),
        run_first_height=("height","min"),
        run_last_height=("height","max"),
        last_orphan_ts=("timestamp","max")
    ).sort_values("run_first_height").reset_index(drop=True)
    return runs

def count_qubic_main_between(main: pd.DataFrame, prev_main_ts: pd.Series, right_ts: pd.Series) -> np.ndarray:
    mts_q = main.loc[main["is_qubic"], "timestamp"].to_numpy("datetime64[ns]")
    if len(mts_q) == 0:
        return np.zeros(len(prev_main_ts), dtype=np.int64)
    left = prev_main_ts.to_numpy("datetime64[ns]")
    right = right_ts.to_numpy("datetime64[ns]")
    left = np.where(pd.isna(left), np.datetime64("1677-09-21T00:12:43.145224192"), left)
    li = np.searchsorted(mts_q, left, side="right")
    ri = np.searchsorted(mts_q, right, side="right")
    return (ri - li).astype(np.int64)

def compute_qubic_only(df: pd.DataFrame):
    main = build_main(df)
    orphan_h = build_orphan_by_height(df)

    main_h = main.set_index("height")[["is_qubic","timestamp"]]
    merged = orphan_h.join(main_h, on="height", how="left", rsuffix="_main")
    orphan_q = merged[merged["is_qubic"] == True][["height","timestamp"]].rename(
        columns={"timestamp":"orphan_ts"}
    ).reset_index(drop=True)

    orphan_q = orphan_q.sort_values("height").reset_index(drop=True)
    runs = compute_runs_consecutive(orphan_q.rename(columns={"orphan_ts":"timestamp"}))

    prev_heights = runs["run_first_height"].to_numpy(np.int64) - 1
    prev_ts = main.set_index("height").reindex(prev_heights)["timestamp"].reset_index(drop=True)

    selfish_len = count_qubic_main_between(main, prev_ts, runs["last_orphan_ts"])

    res = pd.DataFrame({
        "run_first_height": runs["run_first_height"].to_numpy(np.int64),
        "run_last_height": runs["run_last_height"].to_numpy(np.int64),
        "orphan_run_length": runs["orphan_run_length"].to_numpy(np.int64),
        "selfish_mining_run_length": selfish_len
    })
    return res, main, runs, prev_ts, orphan_q

def plot_scatter(res: pd.DataFrame, fig_path: str):
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    if res.empty:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(
        res["selfish_mining_run_length"].to_numpy(),
        res["orphan_run_length"].to_numpy(),
        s=28, alpha=0.75, edgecolors="none", c="red"
    )
    ax.set_xlabel("Selfish Mining Run Length (Qubic only)", fontsize=14)
    ax.set_ylabel("Orphan Run Length (consecutive heights, replaced by Qubic)", fontsize=14)
    ax.grid(True, alpha=0.3)
    max_x = int(res["selfish_mining_run_length"].max())
    max_y = int(res["orphan_run_length"].max())
    if max_x >= 1:
        ax.set_xticks(range(1, max_x + 1))
        ax.set_xlim(0.5, max_x + 0.5)
    if max_y >= 1:
        ax.set_yticks(range(1, max_y + 1))
        ax.set_ylim(0.5, max_y + 0.5)
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

def export_selfish_orphan_height_aligned(main: pd.DataFrame, runs: pd.DataFrame, prev_ts: pd.Series,
                                         orphan_q: pd.DataFrame, out_path: str):
    qubic_main = main.loc[main["is_qubic"], ["height","timestamp"]].rename(
        columns={"height":"selfish_mining_height","timestamp":"selfish_mining_timestamp"}
    ).reset_index(drop=True)

    if qubic_main.empty or runs.empty:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(columns=["selfish_mining_height","selfish_mining_timestamp","orphan_height","orphan_timestamp"]).to_csv(out_path, index=False)
        return

    q_ts = qubic_main["selfish_mining_timestamp"].to_numpy("datetime64[ns]")
    q_h  = qubic_main["selfish_mining_height"].to_numpy(np.int64)

    idx_all = []
    for i in range(len(runs)):
        left = prev_ts.iloc[i]
        right = runs.loc[i, "last_orphan_ts"]
        li = np.searchsorted(q_ts, np.datetime64(left) if pd.notna(left) else np.datetime64("1677-09-21T00:12:43.145224192"), side="right")
        ri = np.searchsorted(q_ts, np.datetime64(right), side="right")
        if ri > li:
            idx_all.append(np.arange(li, ri))

    if len(idx_all):
        idx_all = np.concatenate(idx_all)
    else:
        idx_all = np.array([], dtype=np.int64)

    selfish_df = qubic_main.iloc[idx_all].copy()
    selfish_df = selfish_df.drop_duplicates(subset=["selfish_mining_height"]).sort_values("selfish_mining_height")

    orphan_h = orphan_q.rename(columns={"height":"orphan_height","orphan_ts":"orphan_timestamp"})[
        ["orphan_height","orphan_timestamp"]
    ]

    aligned = selfish_df.merge(
        orphan_h, left_on="selfish_mining_height", right_on="orphan_height", how="left", sort=True
    ).sort_values("selfish_mining_height").reset_index(drop=True)

    aligned = aligned[["selfish_mining_height","selfish_mining_timestamp","orphan_height","orphan_timestamp"]]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    aligned.to_csv(out_path, index=False)

def main():
    df = load_all_blocks(ALL_BLOCKS_PATH)
    res, main_df, runs, prev_ts, orphan_q = compute_qubic_only(df)
    plot_scatter(res, FIG_PATH)
    export_selfish_orphan_height_aligned(main_df, runs, prev_ts, orphan_q, CSV_PATH)

if __name__ == "__main__":
    main()
