## Inside Qubic’s Selfish Mining in Monero

Analyzing Qubic’s behavior and its real impact on Monero’s network

### Overview

The project investigates whether Qubic, a public Monero mining pool, used selfish mining and if such behavior was actually effective.
We collected and analyzed Monero network data between August and October 2025 to measure Qubic’s hash-rate share and mining outcomes.

### What is Selfish Mining?

In normal Proof-of-Work (PoW) systems, miners publish new blocks as soon as they are found so the network stays synchronized.
In selfish mining, a miner keeps newly found blocks private for a short time to get ahead of others. When honest miners find competing blocks, the selfish miner releases its hidden chain to make the others’ work useless.

This strategy can increase a miner’s rewards if it controls a large enough portion of the total hash rate and if other miners prefer its chain.

![Selfish Mining Illustration](figs/selfish_mining.png)

## Main Findings

Qubic’s 51% claim:
Qubic temporarily reached around 51% of Monero’s total hash rate during certain short time windows.
However, this dominance was not stable or sustained. The analysis shows that the control quickly dropped below 50%, meaning it cannot be considered a successful 51% attack.

Effectiveness of the strategy:
The observed behavior partially fits the pattern of selfish mining but did not improve revenue.
On the contrary, Qubic likely lost efficiency compared to honest mining.

Network impact:
While some orphan and fork events coincided with Qubic’s activity, there was no lasting disruption to the Monero network.

### Repository Structure
Folder	Description
data/	Raw and processed blockchain data (blocks.csv, orphan.csv, jobs.csv, etc.)
scripts/	Python scripts for data parsing and plotting (count_qubic.py, daily_alpha_plot.py, etc.)
figs/	Generated figures such as gamma.pdf and alpha_trend.pdf
sec/	LaTeX sources for each section of the paper
results/	Output files summarizing Qubic’s hash-rate share and fork activity

### Key Insights

Selfish mining was detectable but not profitable.

Qubic’s hash-rate advantage was temporary and did not result in network takeover.

The Monero network remained stable overall, showing the limits of selfish mining in a real-world environment.

### 🗂️ Data Collection

The data were collected from:

A Monero full node running in pruning mode

A miner connected directly to Qubic’s public pool

Collection period: August–October 2025