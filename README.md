# Statistical Arbitrage in Cryptocurrencies

Wall Street Quants course project: momentum and reversal strategies on 13 liquid cryptocurrencies (Binance.US 4-hour bars, 2020–2025), net of 20 bps execution costs, with an out-of-sample test on Binance global data (2025–2026).

- [`project_final.ipynb`](project_final.ipynb) — full research notebook (data, momentum, reversal, betting against beta, risk-parity combination, out-of-sample test)
- [`report/main.pdf`](report/main.pdf) — written report ([`report/main.tex`](report/main.tex) is the LaTeX source)

The price data (`data/*.pk`, ~175 MB) are not included; they are downloaded with `python-binance` from Binance.US and from Binance's public mirror `data-api.binance.vision`.
