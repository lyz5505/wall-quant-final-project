"""
Shared functions of the crypto statistical-arbitrage project
(30coins_project_module.ipynb and 13coins_project_V2_module.ipynb).

Data conventions: `r` is a DataFrame of 4-hour returns (index = bar open time, columns = coins),
`univ` a boolean DataFrame of the same shape (True = coin in the universe during the bar),
`mkt` the return of the equal-weight market of the universe, `ew` its weights.
"""
import numpy as np
import pandas as pd
import cvxpy as cvx
from sklearn.covariance import LedoitWolf

DAYS_PER_YEAR = 252     # course convention for annualizing daily statistics
COST = 0.0020           # market orders: 7 bps commission + 13 bps slippage
COST_LIMIT = 0.0007     # limit orders: commission only

EVAL_START = '2021-04-01'          # after the one-year warm-up
TRAIN_END = '2025-08-31'
TEST_START = '2025-09-01'

SETS = {'4 strategies': ['XS', 'TS', 'FS', 'BAB'],
        '3 (no TS)':    ['XS', 'FS', 'BAB'],
        '3 (no BAB)':   ['XS', 'TS', 'FS']}


# ---------- statistics ----------
def to_daily(x):
    # aggregate 4h returns to daily returns (summed, consistent with the cumulative-sum convention)
    return x.resample('1D').sum()


def sharpe(x):
    # course convention: Sharpe of daily-aggregated returns, annualized with sqrt(252)
    d = to_daily(x)
    return d.mean()/d.std()*np.sqrt(DAYS_PER_YEAR)


def max_drawdown(x):                       # on the cumulative (summed) return, from the starting value 0
    cum = x.cumsum()
    return (cum - cum.cummax().clip(lower=0)).min()


def perf(x, mkt):
    # Sharpe, return, vol, drawdown and beta to the market for one return series
    d, m = to_daily(x), to_daily(mkt.loc[x.index])
    return pd.Series({'Sharpe': sharpe(x), 'annual return': d.mean()*DAYS_PER_YEAR,
                      'annual vol': d.std()*np.sqrt(DAYS_PER_YEAR), 'max drawdown': max_drawdown(x),
                      'beta': d.cov(m)/m.var()})


def alpha_beta(x, f):
    # regression of daily strategy returns on daily factor returns
    x, f = to_daily(x), to_daily(f.loc[x.index])
    beta = x.cov(f)/f.var()
    resid = x - beta*f
    alpha = resid.mean()
    t_alpha = alpha / (resid.std()/np.sqrt(len(resid)))
    return pd.Series({'beta': beta, 'annual alpha': alpha*DAYS_PER_YEAR, 't-stat (alpha)': t_alpha})


def summary(x, a, b):
    # Sharpe, return, drawdown and worst year of x between a and b
    x = x.loc[a:b]
    return pd.Series({'Sharpe': sharpe(x), 'annual return': to_daily(x).mean()*DAYS_PER_YEAR, 'max drawdown': max_drawdown(x),
                      'worst year Sharpe': x.groupby(x.index.year).apply(sharpe).min()})


# ---------- portfolio construction (universe-aware) ----------
def norm(s, univ):
    # keep coins in the universe, demean across them (dollar neutral), scale so that sum|w| = 1, others -> 0
    s = s.where(univ)
    s = s.sub(s.mean(axis=1), axis=0)
    return s.div(s.abs().sum(axis=1), axis=0).fillna(0)


def equal_weight(univ):
    # long all coins in the universe, equal weight
    return univ.astype(float).div(univ.sum(axis=1), axis=0).fillna(0)


def hold_at(w, mask):
    # rebalance only where mask is True and hold in between
    return w[mask].reindex(w.index).ffill().fillna(0)


def only(coin, s, like):
    # position s in one coin, 0 elsewhere (same shape as `like`)
    w = pd.DataFrame(0.0, index=like.index, columns=like.columns)
    w[coin] = s
    return w


# ---------- backtest ----------
def drop_halted(held, r):
    # A coin that stops trading (no price) is closed at its last price: from the next bar its weight is 0,
    # and the other weights are rescaled, long and short legs separately, so each leg keeps its size.
    # (The bar in which it stops is only known at its end, so that bar still holds it, with return 0.)
    halted = r.isna().shift(1, fill_value=False)
    held = held.fillna(0)
    h = held.mask(halted, 0)
    long_old, long_new = held.clip(lower=0).sum(axis=1), h.clip(lower=0).sum(axis=1)
    short_old, short_new = held.clip(upper=0).sum(axis=1), h.clip(upper=0).sum(axis=1)
    s_long = (long_old / long_new).where(long_new > 0, 0)
    s_short = (short_old / short_new).where(short_new < 0, 0)
    return h.clip(lower=0).mul(s_long, axis=0) + h.clip(upper=0).mul(s_short, axis=0)


def backtest(held, r):
    # held = position during each bar (already lagged) -> gross return and turnover per bar.
    held = drop_halted(held, r)
    gross = (held*r).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    return gross, turnover


def cost_table(gross, to, eval_start=EVAL_START):
    # gross / to: DataFrames with one column per strategy, evaluated from eval_start
    gross, to = gross.loc[eval_start:], to.loc[eval_start:]
    return pd.DataFrame({
        'gross Sharpe':       sharpe(gross),
        'net Sharpe (20bps)': sharpe(gross - COST*to),
        'net Sharpe (7bps)':  sharpe(gross - COST_LIMIT*to),
        'avg turnover':       to.mean(),
        'breakeven (bps)':    gross.mean()/to.mean()*1e4,
    })


def replacement_turnover(held, lag, univ):
    # turnover on coins whose universe status changed at that bar (entering or leaving)
    u = univ.shift(lag).fillna(False)
    swap = u != u.shift(1).fillna(False)
    return (held.fillna(0).diff().abs()*swap).sum(axis=1)


def cost_table_u(gross, to, to_swap, eval_start=EVAL_START):
    t = cost_table(gross, to, eval_start)
    t['turnover from coin replacement'] = to_swap.loc[eval_start:].sum() / to.loc[eval_start:].sum()
    return t


# ---------- cross-sectional momentum ----------
def xs_momentum(r, h, univ):
    # cross-sectional momentum: rank only the coins in the universe (ranks must not depend on coins outside it)
    return norm(r.rolling(h//4, min_periods=1).mean().where(univ).rank(axis=1), univ)


def xs_momentum_mhva(r, univ, horizons=(168, 336, 720)):
    # multi-horizon + volatility-adjusted momentum, ranked among the coins in the universe
    vol30 = r.rolling(180, min_periods=90).std()
    ranks = [(r.rolling(h//4, min_periods=1).mean()/vol30).where(univ).rank(axis=1) for h in horizons]
    return norm(sum(ranks) / len(horizons), univ)


def tracking_error(w, ideal, sigma):
    # (w - ideal)' Sigma (w - ideal), written so that sigma enters as a PSD parameter
    P = cvx.Parameter(sigma.shape, value=sigma, PSD=True)
    return cvx.quad_form(w, P) - 2*ideal @ sigma @ w + ideal @ sigma @ ideal


def rolling_opt(ideal_w, r, univ, tc_penalty, N=180):
    # Every day at 00:00, over the coins in the universe:
    # min tracking_error(w, ideal) + tc_penalty * COST * |w - w_prev|  s.t. sum(w) = 0, sum|w| <= 1.
    # Coins outside the universe get weight 0. w_prev is the position actually held (previous solution).
    w_prev = pd.Series(0.0, index=r.columns)
    out = {}
    for t in r.index[r.index.hour == 0]:
        coins = r.columns[univ.loc[t].values]
        if len(coins) < 10:                       # before the universe starts
            out[t] = w_prev*0
            continue
        ideal = ideal_w.loc[t, coins].values
        sigma = LedoitWolf().fit(r.loc[:t, coins].iloc[-N:].fillna(0).values).covariance_   # past 30 days only
        w = cvx.Variable(len(coins))
        objective = tracking_error(w, ideal, sigma) + tc_penalty*COST*cvx.sum(cvx.abs(w - w_prev[coins].values))
        cvx.Problem(cvx.Minimize(objective), [cvx.sum(w) == 0, cvx.norm1(w) <= 1]).solve(solver=cvx.CLARABEL)
        new = pd.Series(0.0, index=r.columns)
        if w.value is not None:
            new[coins] = w.value
        else:                                     # solver failed: keep yesterday's position, inside the universe only
            new[coins] = w_prev[coins]
        w_prev = new
        out[t] = new
    return pd.DataFrame(out).T


# ---------- time-series momentum ----------
def ts_signal(m, lookbacks=(7, 14, 30, 60)):
    # sign of the past L-day return of m, for each L, and their average
    sigs = {f'{L}d': np.sign(m.rolling(L*6, min_periods=L*6).sum()) for L in lookbacks}
    sigs['average 7-60d'] = pd.concat(sigs, axis=1).mean(axis=1, skipna=False)
    return sigs


def ts_signal_current(r, univ, lookbacks=(7, 14, 30, 60)):
    # sign of the average past L-day return of the coins currently in the universe
    sigs = {f'{L}d': np.sign(r.rolling(L*6, min_periods=L*3).mean().where(univ).mean(axis=1)) for L in lookbacks}
    sigs['average 7-60d'] = pd.concat(sigs, axis=1).mean(axis=1, skipna=False)
    return sigs


# ---------- reversal ----------
def pairs_trading(r, px4, univ, entry=2.5, exit=0.5, W=180):
    # every pair of coins of the universe on its log-price spread; average over the eligible pairs
    lp = np.log(px4)
    U = univ.values
    co = U.T.astype(np.int32) @ U.astype(np.int32)                  # bars where both coins are in the universe
    cols = list(r.columns)
    G = pd.Series(0.0, index=r.index)
    T = pd.Series(0.0, index=r.index)
    for i, j in zip(*np.triu_indices(len(cols), 1)):
        if co[i, j] == 0:
            continue
        a, b = cols[i], cols[j]
        beta = lp[a].rolling(W).cov(lp[b]) / lp[b].rolling(W).var()
        spread = lp[a] - beta*lp[b]
        z = (spread - spread.rolling(W).mean()) / spread.rolling(W).std()
        both = univ[a] & univ[b]
        pos = pd.Series(np.nan, index=z.index)
        pos[z > entry], pos[z < -entry], pos[z.abs() < exit] = -1, 1, 0
        pos[~both] = 0                                                # close when a coin leaves the universe
        pos = pos.ffill().fillna(0)
        w = pd.DataFrame({a: pos/(1 + beta.abs()), b: -pos*beta/(1 + beta.abs())}).fillna(0)
        g, t = backtest(w.shift(1), r[[a, b]].fillna(0))
        G += g
        T += t
    n_eligible = (univ.sum(axis=1)*(univ.sum(axis=1) - 1)/2).shift(1)  # pairs with both coins in the universe
    n_eligible = n_eligible.where(n_eligible > 0)                       # no eligible pair: return and turnover 0
    return (G/n_eligible).fillna(0), (T/n_eligible).fillna(0)


def fire_sale(mkt, ew, k=2.0, hold=6):
    # buy the universe for `hold` bars after a 24h market drop below -k standard deviations
    z = mkt.rolling(6).sum() / (mkt.rolling(540, min_periods=270).std()*np.sqrt(6))
    s = pd.Series(np.where(z < -k, 1.0, np.nan), index=z.index).ffill(limit=hold-1).fillna(0)
    return ew.mul(s, axis=0), s


# ---------- betting against beta ----------
def betting_against_beta_to(r, f, univ):
    # long the low-beta half, short the high-beta half of the universe (betas to the factor f),
    # each leg scaled by 1/(its average beta)
    beta = pd.DataFrame({c: r[c].rolling(540, min_periods=270).cov(f) / f.rolling(540, min_periods=270).var()
                         for c in r.columns}).where(univ)
    lo = beta.rank(axis=1, pct=True) <= 0.5
    hi = ~lo & beta.notna()
    w_lo = lo.astype(float).div(lo.sum(axis=1), axis=0).div(beta.where(lo).mean(axis=1), axis=0)
    w_hi = hi.astype(float).div(hi.sum(axis=1), axis=0).div(beta.where(hi).mean(axis=1), axis=0)
    w = w_lo - w_hi
    return w.div(w.abs().sum(axis=1), axis=0).fillna(0), beta


def betting_against_beta(r, mkt, univ):
    # BAB with betas to the equal-weight market
    return betting_against_beta_to(r, mkt, univ)


# ---------- combination ----------
def risk_parity(R, cols, WIN=365*6):
    inv_vol = 1 / R[cols].rolling(WIN, min_periods=WIN//2).std()
    return inv_vol.div(inv_vol.sum(axis=1), axis=0).shift(1)


def rolling_max_sharpe(R, cols, min_hist=365*6):
    a, rp = {}, risk_parity(R, cols)
    for t in R.index[(R.index.hour == 0)][::7]:                 # update weekly
        past = R.loc[:t, cols]
        if len(past) < min_hist:
            continue
        w = np.clip(np.linalg.solve(past.cov().values, past.mean().values), 0, None)
        a[t] = w/w.sum() if w.sum() > 0 else rp.loc[t].values
    return pd.DataFrame(a, index=cols).T.reindex(R.index).ffill().shift(1)


def combinations(R, eval_start, sets=SETS):
    E = R.loc[eval_start:]
    out = {c: R[c] for c in ['XS', 'TS', 'FS', 'BAB']}
    for label, cols in sets.items():
        out[f'{label}: equal']               = R[cols].mean(axis=1)
        out[f'{label}: risk parity']         = (risk_parity(R, cols)*R[cols]).sum(axis=1)
        out[f'{label}: rolling max Sharpe']  = (rolling_max_sharpe(R, cols)*R[cols]).sum(axis=1)
        w = np.clip(np.linalg.solve(E[cols].cov().values, E[cols].mean().values), 0, None)
        out[f'{label}: full-sample max Sharpe *'] = R[cols] @ (w/w.sum())
    return pd.DataFrame(out).loc[eval_start:]


def run_shares(w, R, daily_t):
    # strategy shares updated daily at 00:00 and held; 20 bps on the change of each share
    w = w.loc[daily_t].reindex(R.index).ffill().fillna(0)
    return (w*R).sum(axis=1) - COST*w.diff().abs().sum(axis=1), w
