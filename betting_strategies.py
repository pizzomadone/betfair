"""
Betting strategies and financial analysis.

Strategies implemented:
1. Kelly Criterion (full and fractional)
2. Value betting detection
3. Closing Line Value (CLV) analysis
4. Market efficiency analysis (favorite-longshot bias, draw bias)
5. Overround / bookmaker margin analysis
6. Flat staking vs Kelly comparison
7. ROI tracking with bankroll simulation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# 1. KELLY CRITERION
# ─────────────────────────────────────────────

def kelly_stake(p_model: float, decimal_odds: float,
                fraction: float = 0.25) -> float:
    """
    Fractional Kelly criterion stake as fraction of bankroll.

    f = fraction * (b*p - q) / b
    where b = decimal_odds - 1, p = model probability, q = 1 - p.

    Parameters
    ----------
    p_model : Model's estimated probability of winning
    decimal_odds : Bookmaker's decimal odds
    fraction : Kelly fraction (1.0 = full Kelly, 0.25 = quarter Kelly)

    Returns
    -------
    Stake as fraction of bankroll (0 if no value)
    """
    if decimal_odds <= 1.0 or p_model <= 0.0 or p_model >= 1.0:
        return 0.0
    b = decimal_odds - 1.0
    q = 1.0 - p_model
    f = (b * p_model - q) / b
    return max(0.0, fraction * f)


def kelly_expected_log(p_model: float, decimal_odds: float) -> float:
    """Expected log-growth rate at full Kelly stake."""
    b = decimal_odds - 1.0
    f = kelly_stake(p_model, decimal_odds, fraction=1.0)
    if f <= 0:
        return 0.0
    return p_model * np.log(1 + b * f) + (1 - p_model) * np.log(1 - f)


# ─────────────────────────────────────────────
# 2. VALUE BET DETECTION
# ─────────────────────────────────────────────

def has_value(p_model: float, decimal_odds: float,
              min_edge: float = 0.02) -> bool:
    """
    Returns True if the model finds value: p_model > 1/decimal_odds + min_edge.
    min_edge accounts for model uncertainty and transaction costs.
    """
    if decimal_odds <= 1.0 or np.isnan(p_model) or np.isnan(decimal_odds):
        return False
    implied_prob = 1.0 / decimal_odds
    return p_model > implied_prob + min_edge


def value_ratio(p_model: float, decimal_odds: float) -> float:
    """
    Edge ratio: p_model / implied_probability.
    > 1.0 = positive value, < 1.0 = negative value.
    """
    if decimal_odds <= 1.0:
        return 0.0
    implied = 1.0 / decimal_odds
    return p_model / implied if implied > 0 else 0.0


# ─────────────────────────────────────────────
# 3. ROI SIMULATION
# ─────────────────────────────────────────────

class BettingSimulator:
    """
    Simulates a betting bankroll using a given staking strategy.
    Tracks ROI, drawdown, Sharpe ratio, and yield.
    """

    def __init__(self, initial_bankroll: float = 1000.0):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.bets: List[Dict] = []
        self.peak = initial_bankroll

    def reset(self):
        self.bankroll = self.initial_bankroll
        self.bets = []
        self.peak = self.initial_bankroll

    def place_bet(self, odds: float, stake: float, won: bool,
                  date=None, description: str = "") -> None:
        """Record a bet and update bankroll."""
        if stake <= 0 or np.isnan(odds) or np.isnan(stake):
            return
        stake = min(stake, self.bankroll * 0.5)  # safety cap: max 50% of bankroll
        profit = stake * (odds - 1) if won else -stake
        self.bankroll += profit
        self.peak = max(self.peak, self.bankroll)
        self.bets.append({
            "date": date,
            "odds": odds,
            "stake": stake,
            "won": won,
            "profit": profit,
            "bankroll": self.bankroll,
            "description": description,
        })

    def place_kelly_bet(self, p_model: float, odds: float, won: bool,
                        fraction: float = 0.25, date=None,
                        description: str = "") -> None:
        """Place a Kelly-sized bet."""
        f = kelly_stake(p_model, odds, fraction)
        stake = f * self.bankroll
        self.place_bet(odds, stake, won, date, description)

    def place_flat_bet(self, odds: float, won: bool,
                       unit: float = 10.0, date=None,
                       description: str = "") -> None:
        """Place a flat-stake bet."""
        self.place_bet(odds, unit, won, date, description)

    def summary(self) -> Dict:
        if not self.bets:
            return {"n_bets": 0}
        df = pd.DataFrame(self.bets)
        total_staked = df["stake"].sum()
        total_profit = df["profit"].sum()
        n_won = int(df["won"].sum())
        n_bets = len(df)

        # Max drawdown
        drawdowns = (self.peak - df["bankroll"]) / self.initial_bankroll
        max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

        # Sharpe ratio (daily returns, annualized)
        if len(df) > 1:
            returns = df["profit"] / df["stake"].shift(1).fillna(self.initial_bankroll / 10)
            sharpe = float(returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
        else:
            sharpe = 0.0

        roi = total_profit / total_staked if total_staked > 0 else 0.0
        return {
            "n_bets": n_bets,
            "n_won": n_won,
            "win_rate": n_won / n_bets,
            "total_staked": total_staked,
            "total_profit": total_profit,
            "roi": roi,
            "roi_pct": roi * 100,
            "final_bankroll": self.bankroll,
            "max_drawdown_pct": max_dd * 100,
            "avg_odds": float(df["odds"].mean()),
            "sharpe": sharpe,
        }

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.bets)


# ─────────────────────────────────────────────
# 4. CLOSING LINE VALUE (CLV)
# ─────────────────────────────────────────────

def compute_clv(
    df: pd.DataFrame,
    open_col: str,
    close_col: str,
    outcome_col: str,
    result_val,
) -> pd.DataFrame:
    """
    Closing Line Value: measures if the opening odds were better or worse
    than closing odds. Consistently positive CLV → long-term edge.

    CLV = log(close_odds / open_odds) in terms of probability:
    CLV = p_close - p_open  (closing prob - opening prob)
    If you bet Home at open_odds=3.0 and closing is 2.5 → CLV positive (price moved your way).

    Parameters
    ----------
    open_col, close_col : Column names for opening/closing decimal odds
    outcome_col : Column name for match result (e.g., "FTR")
    result_val : Value for the target outcome (e.g., "H" for home win)
    """
    df = df.copy()
    mask = df[[open_col, close_col]].notna().all(axis=1) & (df[open_col] > 1) & (df[close_col] > 1)
    sub = df[mask].copy()

    sub["open_prob"] = 1.0 / sub[open_col]
    sub["close_prob"] = 1.0 / sub[close_col]
    sub["clv_prob"] = sub["close_prob"] - sub["open_prob"]  # positive = moved your way
    sub["clv_odds"] = np.log(sub[open_col] / sub[close_col])   # positive = odds lengthened

    # When you bet, you want odds to LENGTHEN (positive clv_odds)
    # or equiv. close_prob < open_prob: market moved away from you (bad)
    # vs close_prob > open_prob: market moved toward you (good)

    # Actual performance
    sub["bet_won"] = (sub[outcome_col] == result_val).astype(float)
    sub["profit_flat"] = sub["bet_won"] * (sub[open_col] - 1) - (1 - sub["bet_won"])

    return sub[["Date", "HomeTeam", "AwayTeam", open_col, close_col,
                "open_prob", "close_prob", "clv_prob", "clv_odds",
                "bet_won", "profit_flat"]].reset_index(drop=True)


def analyze_clv(df: pd.DataFrame) -> Dict:
    """
    Full CLV analysis across all bookmakers for all three 1X2 outcomes.
    Returns summary statistics.
    """
    results = {}

    pairs = [
        ("B365", "H", "B365H", "B365CH", "FTR", "H"),
        ("B365", "D", "B365D", "B365CD", "FTR", "D"),
        ("B365", "A", "B365A", "B365CA", "FTR", "A"),
        ("PS", "H", "PSH", "PSCH", "FTR", "H"),
        ("PS", "D", "PSD", "PSCD", "FTR", "D"),
        ("PS", "A", "PSA", "PSCA", "FTR", "A"),
        ("BFE", "H", "BFEH", "BFECH", "FTR", "H"),
        ("BFE", "D", "BFED", "BFECD", "FTR", "D"),
        ("BFE", "A", "BFEA", "BFECA", "FTR", "A"),
    ]

    for book, outcome, open_col, close_col, res_col, res_val in pairs:
        if open_col not in df.columns or close_col not in df.columns:
            continue
        clv_df = compute_clv(df, open_col, close_col, res_col, res_val)
        if len(clv_df) < 10:
            continue
        results[f"{book}_{outcome}"] = {
            "n": len(clv_df),
            "avg_clv_prob": float(clv_df["clv_prob"].mean()),
            "avg_clv_odds": float(clv_df["clv_odds"].mean()),
            "pct_positive_clv": float((clv_df["clv_odds"] > 0).mean()),
            "avg_roi_flat": float(clv_df["profit_flat"].mean()),
            "avg_open_odds": float(clv_df[open_col].mean()),
            "avg_close_odds": float(clv_df[close_col].mean()),
        }

    return results


# ─────────────────────────────────────────────
# 5. MARKET EFFICIENCY ANALYSIS
# ─────────────────────────────────────────────

def analyze_market_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze bookmaker efficiency by computing:
    - Actual win rate vs implied probability by odds bucket
    - Favorite-longshot bias
    - Draw bias
    Returns calibration DataFrame.
    """
    rows = []
    for col, outcome in [("MaxH", "H"), ("MaxD", "D"), ("MaxA", "A")]:
        if col not in df.columns:
            continue
        mask = df[col].notna() & (df[col] > 1)
        sub = df[mask][[col, "FTR"]].copy()
        sub["implied"] = 1.0 / sub[col]
        sub["actual"] = (sub["FTR"] == outcome).astype(float)
        sub["bucket"] = pd.cut(sub[col], bins=[1, 1.3, 1.6, 2.0, 2.5, 3.5, 5.0, 100], right=True)
        for bucket, grp in sub.groupby("bucket", observed=True):
            rows.append({
                "market": col,
                "outcome": outcome,
                "odds_range": str(bucket),
                "n": len(grp),
                "avg_implied_prob": grp["implied"].mean(),
                "actual_win_rate": grp["actual"].mean(),
                "edge": grp["actual"].mean() - grp["implied"].mean(),
                "avg_odds": grp[col].mean(),
                "roi_flat": (grp["actual"] * (grp[col] - 1) - (1 - grp["actual"])).mean(),
            })
    return pd.DataFrame(rows)


def favorite_longshot_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Favorite-longshot bias: are favorites under-priced and longshots over-priced?
    Uses Pinnacle opening odds as reference.
    """
    rows = []
    for col, outcome in [("PSH", "H"), ("PSD", "D"), ("PSA", "A")]:
        if col not in df.columns:
            continue
        sub = df[df[col].notna() & (df[col] > 1)]
        sub = sub[[col, "FTR"]].copy()
        sub["implied"] = 1.0 / sub[col]
        sub["actual"] = (sub["FTR"] == outcome).astype(float)
        sub["bucket"] = pd.cut(sub[col], bins=[1, 1.5, 2.0, 3.0, 5.0, 100], right=True)
        for bucket, grp in sub.groupby("bucket", observed=True):
            if len(grp) < 10:
                continue
            rows.append({
                "odds_range": str(bucket),
                "outcome": outcome,
                "n": len(grp),
                "avg_odds": grp[col].mean(),
                "implied_prob": grp["implied"].mean(),
                "actual_prob": grp["actual"].mean(),
                "edge": grp["actual"].mean() - grp["implied"].mean(),
                "roi": (grp["actual"] * (grp[col] - 1) - (1 - grp["actual"])).mean(),
            })
    return pd.DataFrame(rows).sort_values("avg_odds")


def draw_bias_analysis(df: pd.DataFrame) -> Dict:
    """
    Draws are traditionally under-predicted by bettors.
    Analyze whether draws offer systematic value.
    """
    result = {}
    for col in ["B365D", "PSD", "BFED", "MaxD", "AvgD"]:
        if col not in df.columns:
            continue
        sub = df[df[col].notna() & (df[col] > 1)]
        actual_draw_rate = (sub["FTR"] == "D").mean()
        avg_implied = (1.0 / sub[col]).mean()
        roi = ((sub["FTR"] == "D").astype(float) * (sub[col] - 1)
               - (sub["FTR"] != "D").astype(float)).mean()
        result[col] = {
            "actual_draw_rate": actual_draw_rate,
            "avg_implied_prob": avg_implied,
            "bias": actual_draw_rate - avg_implied,
            "flat_roi": roi,
            "n": len(sub),
        }
    return result


# ─────────────────────────────────────────────
# 6. VALUE BETTING BACKTEST WITH MULTIPLE MODELS
# ─────────────────────────────────────────────

def backtest_value_strategy(
    df: pd.DataFrame,
    model_prob_cols: Dict[str, Tuple[str, str, str]],  # name -> (home_col, draw_col, away_col)
    odds_col_h: str = "B365H",
    odds_col_d: str = "B365D",
    odds_col_a: str = "B365A",
    min_edge: float = 0.03,
    kelly_fraction: float = 0.25,
    min_odds: float = 1.5,
    max_odds: float = 15.0,
    start_idx: int = 200,
) -> Dict[str, Dict]:
    """
    Backtest value betting for multiple models simultaneously.

    For each model, for each match from start_idx onwards:
    1. Get model probability for each outcome
    2. Compare to bookmaker odds
    3. If value exists (edge > min_edge), place Kelly bet
    4. Track ROI

    Returns dict: model_name → ROI summary
    """
    results = {}

    for model_name, (h_col, d_col, a_col) in model_prob_cols.items():
        sim = BettingSimulator()
        sim_flat = BettingSimulator()

        for i, row in df.iloc[start_idx:].iterrows():
            for outcome, p_col, odds_col in [
                ("H", h_col, odds_col_h),
                ("D", d_col, odds_col_d),
                ("A", a_col, odds_col_a),
            ]:
                p = row.get(p_col, np.nan)
                odds = row.get(odds_col, np.nan)
                if np.isnan(p) or np.isnan(odds):
                    continue
                if odds < min_odds or odds > max_odds:
                    continue
                if not has_value(p, odds, min_edge):
                    continue

                won = (row["FTR"] == outcome)
                sim.place_kelly_bet(p, odds, won, fraction=kelly_fraction,
                                    date=row.get("Date"),
                                    description=f"{row['HomeTeam']} vs {row['AwayTeam']} [{outcome}]")
                sim_flat.place_flat_bet(odds, won, unit=10.0,
                                        date=row.get("Date"),
                                        description=f"{row['HomeTeam']} vs {row['AwayTeam']} [{outcome}]")

        results[model_name] = {
            "kelly": sim.summary(),
            "flat": sim_flat.summary(),
            "bets_df": sim.as_dataframe(),
        }

    return results


def print_roi_report(results: Dict[str, Dict], title: str = "Value Betting ROI Report"):
    """Print a formatted ROI comparison table."""
    from tabulate import tabulate
    rows = []
    for model, res in results.items():
        for staking, data in [("Kelly", res.get("kelly", {})), ("Flat", res.get("flat", {}))]:
            if data.get("n_bets", 0) == 0:
                continue
            rows.append([
                model, staking,
                data.get("n_bets", 0),
                f"{data.get('win_rate', 0)*100:.1f}%",
                f"{data.get('avg_odds', 0):.2f}",
                f"{data.get('roi_pct', 0):.2f}%",
                f"€{data.get('total_profit', 0):.2f}",
                f"{data.get('max_drawdown_pct', 0):.1f}%",
                f"{data.get('sharpe', 0):.2f}",
            ])
    headers = ["Model", "Staking", "Bets", "Win%", "Avg Odds", "ROI%",
               "Profit", "MaxDD%", "Sharpe"]
    print(f"\n{'='*80}")
    print(f"  {title}")
    print('='*80)
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))


if __name__ == "__main__":
    from data_loader import load_data
    from statistical_models import add_pinnacle_probabilities

    df = load_data()
    df = add_pinnacle_probabilities(df)

    print("=== CLV Analysis ===")
    clv = analyze_clv(df)
    for k, v in list(clv.items())[:6]:
        print(f"  {k}: avg_clv={v['avg_clv_prob']:.4f}  pct_pos={v['pct_positive_clv']:.2%}  roi={v['avg_roi_flat']:.4f}")

    print("\n=== Draw Bias ===")
    draw = draw_bias_analysis(df)
    for book, stats in draw.items():
        print(f"  {book}: actual_draw={stats['actual_draw_rate']:.3f}  implied={stats['avg_implied_prob']:.3f}  roi={stats['flat_roi']:.4f}")

    print("\n=== Favorite-Longshot Bias (Pinnacle) ===")
    flb = favorite_longshot_bias(df)
    print(flb[flb["outcome"] == "H"][["odds_range", "n", "actual_prob", "implied_prob", "edge", "roi"]].to_string(index=False))
