# Crypto Statistical Arbitrage

Wall Street Quants course project: cross-sectional momentum, time-series momentum, a fire-sale reversal and betting-against-beta on Binance global cryptocurrencies, combined by risk parity, net of 20 bps execution costs. The main universe is point in time (each month, the top 30 coins by trailing 90-day dollar volume, stablecoins excluded); a fixed set of 13 large coins is used for comparison. Training April 2021 – August 2025, test September 2025 – October 2026.

- [`report/main_v3.pdf`](report/main_v3.pdf) — report ([`report/main_v3.tex`](report/main_v3.tex) is the LaTeX source, figures in [`report/figures/`](report/figures))
- [`30coins_project_module.ipynb`](30coins_project_module.ipynb) — main analysis, 30-coin point-in-time universe (training, test, strategy failure)
- [`13coins_project_V2_module.ipynb`](13coins_project_V2_module.ipynb) — the same analysis on the 13 coins
- [`statarb.py`](statarb.py) — function definitions (signals, cvxpy optimizer, backtest, risk parity, statistics)
- [`global30_data.ipynb`](global30_data.ipynb) — data download from Binance global (`data.binance.vision`) and coin selection for the 30-coin universe (monthly top 30 by 90-day dollar volume, stablecoins excluded)
- [`data_13coins.ipynb`](data_13coins.ipynb) — data download from Binance.US and selection of the 13 coins
- [`old version/`](old%20version) — earlier version (13 Binance.US coins): `project_final.ipynb` and `report/main.pdf`

The price data (`data/*.pk`, ~1 GB) are not included; the two data notebooks download them.
