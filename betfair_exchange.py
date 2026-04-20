"""
Betfair Exchange-specific analysis and strategies.

Betfair Exchange is fundamentally different from traditional bookmakers:
- Peer-to-peer: bettors back and lay against each other
- Commission: 2-5% on net winnings per market (not per bet)
- True market odds: no house margin, prices set by supply/demand
- In-play markets: odds update in real-time during the match

Strategies analyzed:
1. Exchange vs bookmaker arbitrage (back bookmaker, lay exchange)
2. Back-Lay spread analysis (Betfair's own margin)
3. O/U 2.5 market efficiency on Betfair
4. Asian Handicap exchange analysis
5. Lay-the-draw strategy analysis
6. Pre-match vs in-play value patterns
7. Dutch betting opportunities
8. Market liquidity and timing
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


BETFAIR_COMMISSION = 0.05   # Standard 5% commission (can be lower for high volume)
MIN_ARRATE_EDGE = 0.01      # Minimum arbitrage edge after commission


# ─────────────────────────────────────────────
# 1. BACK-LAY ARBITRAGE (Bookmaker vs Exchange)
# ─────────────────────────────────────────────

def detect_back_lay_arb(
    df: pd.DataFrame,
    bookmaker_h: str = "B365H",
    bookmaker_d: str = "B365D",
    bookmaker_a: str = "B365A",
    exchange_h: str = "BFEH",
    exchange_d: str = "BFED",
    exchange_a: str = "BFEA",
    commission: float = BETFAIR_COMMISSION,
) -> pd.DataFrame:
    """
    Detect matches where bookmaker back odds > Betfair exchange lay odds.

    Strategy: Back at bookmaker odds → guaranteed profit by laying on exchange.
    Profit = back_stake * (bookmaker_odds - 1) - lay_liability
    After commission, net profit on exchange win = lay_stake * (lay_odds - 1) * (1-comm)

    Arbitrage condition (for outcome X):
    bookmaker_odds_X > exchange_lay_odds_X (which is approx. the listed back price)

    Note: In practice exchange back price IS the lay price from the opposite side.
    If bookie offers 2.5 and exchange shows 2.3 back, you can:
    - Back at 2.5 with bookmaker (stake S)
    - Lay at 2.3 on exchange (lay stake = S * 2.5 / 2.3 to cover)
    Net profit ≈ S * (2.5/2.3 - 1) adjusted for commission
    """
    arbs = []

    for _, row in df.iterrows():
        for outcome, bk_col, ex_col in [
            ("H", bookmaker_h, exchange_h),
            ("D", bookmaker_d, exchange_d),
            ("A", bookmaker_a, exchange_a),
        ]:
            bk_odds = row.get(bk_col)
            ex_odds = row.get(ex_col)
            if pd.isna(bk_odds) or pd.isna(ex_odds):
                continue
            if bk_odds <= 1.01 or ex_odds <= 1.01:
                continue

            # Profit calculation for back-lay arb
            # Back 100 at bookmaker odds bk_odds
            # Lay on exchange: lay stake = 100 * bk_odds / ex_odds
            # Commission applies to exchange profits only
            back_stake = 100.0
            lay_stake = back_stake * bk_odds / ex_odds
            lay_liability = lay_stake * (ex_odds - 1)  # if lay loses

            # Scenario A: outcome wins → bookmaker pays, exchange loses
            profit_if_wins = back_stake * (bk_odds - 1) - lay_liability
            # Scenario B: outcome loses → bookmaker loses stake, exchange pays
            exchange_profit = lay_stake * (1 - commission)  # commission on winnings
            profit_if_loses = -back_stake + exchange_profit

            if profit_if_wins > 0 and profit_if_loses > 0:
                edge = min(profit_if_wins, profit_if_loses)
                arbs.append({
                    "Date": row.get("Date"),
                    "HomeTeam": row.get("HomeTeam"),
                    "AwayTeam": row.get("AwayTeam"),
                    "Outcome": outcome,
                    "Bookmaker": bk_col.replace("H", "").replace("D", "").replace("A", ""),
                    "bk_odds": bk_odds,
                    "ex_odds": ex_odds,
                    "profit_if_wins": profit_if_wins,
                    "profit_if_loses": profit_if_loses,
                    "min_profit": edge,
                    "roi_pct": edge / back_stake * 100,
                    "FTR": row.get("FTR"),
                })

    return pd.DataFrame(arbs) if arbs else pd.DataFrame()


# ─────────────────────────────────────────────
# 2. EXCHANGE BACK-LAY SPREAD ANALYSIS
# ─────────────────────────────────────────────

def analyze_exchange_spread(df: pd.DataFrame) -> Dict:
    """
    Analyze the implicit spread on Betfair Exchange for 1X2 markets.

    The exchange overround = 1/back_H + 1/back_D + 1/back_A
    For a perfectly efficient market this would be ≈ 1.0 (no juice).
    In practice slightly below 1.0 due to tick sizes.
    Compare exchange overround vs traditional bookmakers.
    """
    results = {}

    for prefix, h, d, a in [
        ("Betfair_Exchange", "BFEH", "BFED", "BFEA"),
        ("Betfair_Exchange_Close", "BFECH", "BFECD", "BFECA"),
        ("Pinnacle", "PSH", "PSD", "PSA"),
        ("Bet365", "B365H", "B365D", "B365A"),
        ("Max_Market", "MaxH", "MaxD", "MaxA"),
        ("Average_Market", "AvgH", "AvgD", "AvgA"),
    ]:
        if h not in df.columns:
            continue
        mask = df[[h, d, a]].notna().all(axis=1) & (df[h] > 1) & (df[d] > 1) & (df[a] > 1)
        sub = df[mask]
        if len(sub) < 10:
            continue
        overround = 1 / sub[h] + 1 / sub[d] + 1 / sub[a]
        results[prefix] = {
            "n": len(sub),
            "avg_overround": float(overround.mean()),
            "median_overround": float(overround.median()),
            "min_overround": float(overround.min()),
            "max_overround": float(overround.max()),
            "margin_pct": float((overround.mean() - 1) * 100),  # bookmaker margin %
        }

    return results


# ─────────────────────────────────────────────
# 3. OVER/UNDER 2.5 MARKET ANALYSIS
# ─────────────────────────────────────────────

def analyze_ou_market(df: pd.DataFrame) -> Dict:
    """
    Over/Under 2.5 markets are often less efficient than 1X2.
    Analyze efficiency, biases, and find edges.
    """
    result = {}

    # Check actual O/U distribution
    if "TotalGoals" in df.columns:
        total_over = (df["TotalGoals"] > 2.5).mean()
        result["actual_over25_rate"] = float(total_over)
        result["actual_under25_rate"] = float(1 - total_over)

    # Compare implied vs actual by bookmaker
    for bk, o_col, u_col in [
        ("BFE", "BFE>2.5", "BFE<2.5"),
        ("B365", "B365>2.5", "B365<2.5"),
        ("PS", "P>2.5", "P<2.5"),
        ("Avg", "Avg>2.5", "Avg<2.5"),
    ]:
        if o_col not in df.columns:
            continue
        sub = df[df[o_col].notna() & df[u_col].notna() & (df[o_col] > 1)].copy()
        if len(sub) < 20:
            continue

        actual_over = (sub["TotalGoals"] > 2.5).astype(float)
        implied_over = 1.0 / sub[o_col]
        vig = 1 / sub[o_col] + 1 / sub[u_col] - 1

        flat_roi_over = (actual_over * (sub[o_col] - 1) - (1 - actual_over)).mean()
        flat_roi_under = ((1 - actual_over) * (sub[u_col] - 1) - actual_over).mean()

        result[bk] = {
            "n": len(sub),
            "actual_over_rate": float(actual_over.mean()),
            "avg_implied_over": float(implied_over.mean()),
            "over_edge": float(actual_over.mean() - implied_over.mean()),
            "avg_vig_pct": float(vig.mean() * 100),
            "roi_over_flat": float(flat_roi_over),
            "roi_under_flat": float(flat_roi_under),
        }

    # Season-level O/U analysis
    if "Season" in df.columns:
        season_stats = df.groupby("Season").agg(
            n=("TotalGoals", "count"),
            avg_goals=("TotalGoals", "mean"),
            over25_rate=("TotalGoals", lambda x: (x > 2.5).mean()),
        ).reset_index()
        result["by_season"] = season_stats.to_dict("records")

    # Goal line: how often does total fall exactly on common lines
    if "TotalGoals" in df.columns:
        result["goals_distribution"] = {
            str(g): int((df["TotalGoals"] == g).sum())
            for g in range(0, 9)
        }

    return result


# ─────────────────────────────────────────────
# 4. LAY-THE-DRAW STRATEGY
# ─────────────────────────────────────────────

def analyze_lay_draw(df: pd.DataFrame, exchange_col: str = "BFED",
                     commission: float = BETFAIR_COMMISSION) -> Dict:
    """
    Lay-the-draw strategy:
    - Lay the draw before kick-off (bet that there WILL NOT be a draw)
    - If a goal is scored (usually within 60 min), draw odds lengthen
    - Trade out (back the draw) for profit, or let run if confident

    Historical analysis: if we always lay the draw and let it run:
    - Win if match doesn't end in draw
    - Lose if match ends in draw

    This is essentially betting against draws. The question is whether
    exchange draw odds offer value vs actual draw frequency.
    """
    if exchange_col not in df.columns:
        return {}

    sub = df[df[exchange_col].notna() & (df[exchange_col] > 1)].copy()
    sub["draw_occurred"] = (sub["FTR"] == "D").astype(float)
    sub["implied_draw"] = 1.0 / sub[exchange_col]
    actual_draw = sub["draw_occurred"].mean()

    # Flat lay: lay 100 at exchange draw odds
    # If no draw: profit = lay_stake * (1 - commission) = 100 * (1-0.05) = 95
    # If draw: loss = lay_stake * (odds - 1) = 100 * (lay_odds - 1)
    sub["lay_profit"] = np.where(
        sub["draw_occurred"] == 0,
        100 * (1 - commission),   # no draw: collect lay stake
        -100 * (sub[exchange_col] - 1),   # draw: pay out lay liability
    )

    # Filter by odds buckets
    results_by_odds = {}
    for lo, hi in [(1.0, 2.5), (2.5, 3.5), (3.5, 5.0), (5.0, 100)]:
        mask = (sub[exchange_col] >= lo) & (sub[exchange_col] < hi)
        grp = sub[mask]
        if len(grp) < 10:
            continue
        roi = grp["lay_profit"].sum() / (100 * len(grp))
        results_by_odds[f"{lo}-{hi}"] = {
            "n": len(grp),
            "actual_draw_rate": float(grp["draw_occurred"].mean()),
            "avg_implied": float(grp["implied_draw"].mean()),
            "roi_flat": float(roi),
            "avg_lay_odds": float(grp[exchange_col].mean()),
        }

    return {
        "total_matches": len(sub),
        "actual_draw_rate": float(actual_draw),
        "avg_implied_draw_prob": float(sub["implied_draw"].mean()),
        "overall_roi": float(sub["lay_profit"].sum() / (100 * len(sub))),
        "by_odds": results_by_odds,
    }


# ─────────────────────────────────────────────
# 5. ASIAN HANDICAP ANALYSIS
# ─────────────────────────────────────────────

def analyze_asian_handicap(df: pd.DataFrame) -> Dict:
    """
    Asian Handicap markets are often more efficient than 1X2 but can still
    have edges, especially on non-standard lines.

    Analyzes BFE AH market vs Pinnacle AH.
    """
    result = {}

    for bk, ahh_col, aha_col, ahch_col, ahca_col in [
        ("BFE", "BFEAHH", "BFEAHA", "BFECAHH", "BFECAHA"),
        ("B365", "B365AHH", "B365AHA", "B365CAHH", "B365CAHA"),
        ("PS", "PAHH", "PAHA", "PCAHH", "PCAHA"),
    ]:
        if ahh_col not in df.columns:
            continue
        sub = df[df[[ahh_col, aha_col]].notna().all(axis=1)].copy()
        if len(sub) < 20:
            continue

        sub["ah_line"] = sub.get("AHh", pd.Series(np.zeros(len(df))))[sub.index]
        sub["margin"] = 1 / sub[ahh_col] + 1 / sub[aha_col] - 1
        sub["implied_home"] = (1 / sub[ahh_col]) / (1 / sub[ahh_col] + 1 / sub[aha_col])

        # AH result: home wins with handicap
        def ah_result(row):
            line = row.get("AHh", 0) if "AHh" in row else 0
            gd = row["FTHG"] - row["FTAG"]
            adj = gd + line
            if adj > 0:
                return "home"
            elif adj < 0:
                return "away"
            return "push"

        sub["ah_result"] = sub.apply(ah_result, axis=1)
        sub["home_wins_ah"] = (sub["ah_result"] == "home").astype(float)
        sub["home_wins_ah"] = sub["home_wins_ah"].where(sub["ah_result"] != "push", 0.5)

        result[bk] = {
            "n": len(sub),
            "avg_margin_pct": float(sub["margin"].mean() * 100),
            "roi_home_ah": float((sub["home_wins_ah"] * (sub[ahh_col] - 1)
                                  - (1 - sub["home_wins_ah"])).mean()),
            "roi_away_ah": float(((1 - sub["home_wins_ah"]) * (sub[aha_col] - 1)
                                  - sub["home_wins_ah"]).mean()),
            "home_cover_rate": float(sub["home_wins_ah"].mean()),
        }

        # CLV for AH
        if ahch_col in df.columns:
            close_sub = sub[sub[ahch_col].notna()].copy()
            if len(close_sub) > 10:
                result[bk]["clv_home_ah"] = float(
                    (1 / close_sub[ahch_col] - 1 / close_sub[ahh_col]).mean()
                )

    return result


# ─────────────────────────────────────────────
# 6. DUTCH BETTING (COVERING MULTIPLE OUTCOMES)
# ─────────────────────────────────────────────

def dutch_bet_analysis(df: pd.DataFrame,
                       commission: float = BETFAIR_COMMISSION) -> pd.DataFrame:
    """
    Dutch betting: back multiple outcomes to guarantee a profit if any wins.
    On Betfair Exchange, you can lay to create risk-free positions.

    Find matches where backing 2+ outcomes across bookmakers guarantees profit.
    For example: Back Home at B365 + Back Draw at BFE → profit if H or D
    """
    rows = []
    for _, row in df.iterrows():
        # Dutch: home + away (exclude draw) across max odds
        odds_h = row.get("MaxH", np.nan)
        odds_a = row.get("MaxA", np.nan)

        if pd.isna(odds_h) or pd.isna(odds_a):
            continue
        if odds_h <= 1 or odds_a <= 1:
            continue

        # Total implied probability for H+A
        total_implied_ha = 1 / odds_h + 1 / odds_a
        if total_implied_ha < 1.0:  # combined odds < 1 → guaranteed profit (Dutch)
            profit_pct = (1 - total_implied_ha) / total_implied_ha * 100
            rows.append({
                "Date": row.get("Date"),
                "HomeTeam": row.get("HomeTeam"),
                "AwayTeam": row.get("AwayTeam"),
                "dutch_outcomes": "H+A",
                "odds_h": odds_h,
                "odds_a": odds_a,
                "total_implied": total_implied_ha,
                "profit_pct": profit_pct,
                "FTR": row.get("FTR"),
            })

        # Dutch: home + draw
        odds_d = row.get("MaxD", np.nan)
        if not pd.isna(odds_d) and odds_d > 1:
            total_hd = 1 / odds_h + 1 / odds_d
            if total_hd < 1.0:
                rows.append({
                    "Date": row.get("Date"),
                    "HomeTeam": row.get("HomeTeam"),
                    "AwayTeam": row.get("AwayTeam"),
                    "dutch_outcomes": "H+D",
                    "odds_h": odds_h,
                    "odds_a": odds_d,
                    "total_implied": total_hd,
                    "profit_pct": (1 - total_hd) / total_hd * 100,
                    "FTR": row.get("FTR"),
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────
# 7. BETFAIR EXCHANGE VS PINNACLE EFFICIENCY
# ─────────────────────────────────────────────

def compare_exchange_pinnacle(df: pd.DataFrame,
                               commission: float = BETFAIR_COMMISSION) -> pd.DataFrame:
    """
    Compare Betfair Exchange vs Pinnacle probability estimates.

    Which is more efficient?
    - Compute de-vigged probabilities for both
    - Compare to actual outcomes
    - Measure Brier score and log-loss for each

    Betfair Exchange odds must be adjusted for commission:
    effective_odds = raw_odds * (1 - commission) + commission
    (net payout = stake * raw_odds * (1-comm) when winning)
    """
    pairs = []

    for _, row in df.iterrows():
        bfe_h = row.get("BFEH", np.nan)
        bfe_d = row.get("BFED", np.nan)
        bfe_a = row.get("BFEA", np.nan)
        ps_h = row.get("PSH", np.nan)
        ps_d = row.get("PSD", np.nan)
        ps_a = row.get("PSA", np.nan)
        ftr = row.get("FTR")

        if any(pd.isna(x) for x in [bfe_h, bfe_d, bfe_a, ps_h, ps_d, ps_a, ftr]):
            continue

        # Betfair effective odds (after commission)
        eff_h = bfe_h * (1 - commission)
        eff_d = bfe_d * (1 - commission)
        eff_a = bfe_a * (1 - commission)

        # De-vigged probabilities
        bfe_tot = 1 / eff_h + 1 / eff_d + 1 / eff_a
        bfe_ph = (1 / eff_h) / bfe_tot
        bfe_pd = (1 / eff_d) / bfe_tot
        bfe_pa = (1 / eff_a) / bfe_tot

        ps_tot = 1 / ps_h + 1 / ps_d + 1 / ps_a
        ps_ph = (1 / ps_h) / ps_tot
        ps_pd = (1 / ps_d) / ps_tot
        ps_pa = (1 / ps_a) / ps_tot

        actual = {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}[ftr]

        # Brier score (lower = better)
        bfe_brier = sum((bfe_ph - actual[0]) ** 2 + (bfe_pd - actual[1]) ** 2
                        + (bfe_pa - actual[2]) ** 2 for _ in [1]) / 3
        ps_brier = sum((ps_ph - actual[0]) ** 2 + (ps_pd - actual[1]) ** 2
                       + (ps_pa - actual[2]) ** 2 for _ in [1]) / 3

        bfe_ll = -np.log([bfe_ph, bfe_pd, bfe_pa][["H", "D", "A"].index(ftr)] + 1e-9)
        ps_ll = -np.log([ps_ph, ps_pd, ps_pa][["H", "D", "A"].index(ftr)] + 1e-9)

        pairs.append({
            "Date": row.get("Date"),
            "bfe_ph": bfe_ph, "bfe_pd": bfe_pd, "bfe_pa": bfe_pa,
            "ps_ph": ps_ph, "ps_pd": ps_pd, "ps_pa": ps_pa,
            "diff_h": bfe_ph - ps_ph,
            "diff_d": bfe_pd - ps_pd,
            "diff_a": bfe_pa - ps_pa,
            "bfe_brier": bfe_brier,
            "ps_brier": ps_brier,
            "bfe_ll": bfe_ll,
            "ps_ll": ps_ll,
            "ftr": ftr,
        })

    return pd.DataFrame(pairs)


def find_bfe_vs_ps_edges(df: pd.DataFrame,
                          commission: float = BETFAIR_COMMISSION,
                          min_diff: float = 0.03) -> pd.DataFrame:
    """
    Find matches where BFE and Pinnacle significantly disagree.
    When Pinnacle (the sharper book) implies higher probability than BFE:
    → The BFE back price is too high → potential value bet on BFE.

    Strategy: Bet on Betfair Exchange when exchange odds > Pinnacle de-vigged fair odds.
    """
    rows = []
    for _, row in df.iterrows():
        for outcome, bfe_col, ps_col, bfe_close, ps_close in [
            ("H", "BFEH", "PSH", "BFECH", "PSCH"),
            ("D", "BFED", "PSD", "BFECD", "PSCD"),
            ("A", "BFEA", "PSA", "BFECA", "PSCA"),
        ]:
            bfe_odds = row.get(bfe_col)
            ps_h = row.get("PSH"); ps_d = row.get("PSD"); ps_a = row.get("PSA")
            if any(pd.isna(x) for x in [bfe_odds, ps_h, ps_d, ps_a]):
                continue
            if bfe_odds <= 1 or ps_h <= 1:
                continue

            # Pinnacle de-vigged probability
            ps_tot = 1 / ps_h + 1 / ps_d + 1 / ps_a
            ps_probs = {"H": 1/ps_h/ps_tot, "D": 1/ps_d/ps_tot, "A": 1/ps_a/ps_tot}
            ps_prob = ps_probs[outcome]

            # BFE effective probability (after commission)
            bfe_effective = bfe_odds * (1 - commission)
            bfe_implied = 1.0 / bfe_effective

            # Edge: if Pinnacle says higher prob than BFE implies → BFE is mispriced
            edge = ps_prob - bfe_implied
            if edge > min_diff:
                rows.append({
                    "Date": row.get("Date"),
                    "HomeTeam": row.get("HomeTeam"),
                    "AwayTeam": row.get("AwayTeam"),
                    "Outcome": outcome,
                    "BFE_odds": bfe_odds,
                    "BFE_effective_odds": bfe_effective,
                    "PS_fair_prob": ps_prob,
                    "BFE_implied_prob": bfe_implied,
                    "edge": edge,
                    "FTR": row.get("FTR"),
                    "Won": row.get("FTR") == outcome,
                })

    df_edges = pd.DataFrame(rows) if rows else pd.DataFrame()
    if len(df_edges) > 0:
        df_edges["profit_flat"] = (
            df_edges["Won"].astype(float) * (df_edges["BFE_effective_odds"] - 1)
            - (1 - df_edges["Won"].astype(float))
        )
    return df_edges


def full_exchange_report(df: pd.DataFrame) -> None:
    """Print comprehensive Betfair Exchange analysis report."""
    from tabulate import tabulate

    print("\n" + "=" * 70)
    print("  BETFAIR EXCHANGE ANALYSIS")
    print("=" * 70)

    # 1. Spread/overround comparison
    print("\n--- 1. MARKET SPREAD COMPARISON (lower = better for bettors) ---")
    spread = analyze_exchange_spread(df)
    rows = [[k, v["n"], f"{v['avg_overround']:.4f}", f"{v['margin_pct']:.2f}%"]
            for k, v in spread.items()]
    print(tabulate(rows, headers=["Market", "N", "Avg Overround", "Margin%"],
                   tablefmt="rounded_outline"))

    # 2. O/U analysis
    print("\n--- 2. OVER/UNDER 2.5 MARKET ---")
    ou = analyze_ou_market(df)
    print(f"  Actual Over 2.5 rate: {ou.get('actual_over25_rate', 0):.3f}")
    for bk in ["BFE", "B365", "PS"]:
        if bk in ou:
            d = ou[bk]
            print(f"  {bk}: implied_over={d['avg_implied_over']:.3f}  "
                  f"edge={d['over_edge']:+.4f}  ROI_over={d['roi_over_flat']:+.4f}  "
                  f"ROI_under={d['roi_under_flat']:+.4f}  margin={d['avg_vig_pct']:.2f}%")

    # 3. Lay the draw
    print("\n--- 3. LAY THE DRAW STRATEGY (Betfair Exchange) ---")
    ltd = analyze_lay_draw(df, "BFED")
    if ltd:
        print(f"  Total matches: {ltd['total_matches']}")
        print(f"  Actual draw rate: {ltd['actual_draw_rate']:.3f}")
        print(f"  Avg implied draw prob: {ltd['avg_implied_draw_prob']:.3f}")
        print(f"  Overall ROI (flat lay): {ltd['overall_roi']:.4f} ({ltd['overall_roi']*100:.2f}%)")
        for odds_range, stats in ltd.get("by_odds", {}).items():
            print(f"    Odds {odds_range}: draw_rate={stats['actual_draw_rate']:.3f}  "
                  f"roi={stats['roi_flat']:+.4f}")

    # 4. Asian Handicap
    print("\n--- 4. ASIAN HANDICAP MARKET ---")
    ah = analyze_asian_handicap(df)
    for bk, stats in ah.items():
        print(f"  {bk}: n={stats['n']}  margin={stats['avg_margin_pct']:.2f}%  "
              f"home_cover={stats['home_cover_rate']:.3f}  "
              f"ROI_home={stats['roi_home_ah']:+.4f}  ROI_away={stats['roi_away_ah']:+.4f}")

    # 5. Back-Lay arbitrage
    print("\n--- 5. BACK-LAY ARBITRAGE OPPORTUNITIES ---")
    arbs = detect_back_lay_arb(df)
    if len(arbs) > 0:
        print(f"  Found {len(arbs)} potential arbitrage opportunities")
        print(f"  Avg ROI: {arbs['roi_pct'].mean():.2f}%")
        print(f"  Max ROI: {arbs['roi_pct'].max():.2f}%")
        print(arbs.sort_values("roi_pct", ascending=False).head(10)[
            ["Date", "HomeTeam", "AwayTeam", "Outcome", "bk_odds", "ex_odds",
             "roi_pct"]].to_string(index=False))
    else:
        print("  No risk-free arbitrage found (market is well integrated)")

    # 6. BFE vs Pinnacle edges
    print("\n--- 6. BETFAIR EXCHANGE vs PINNACLE DISAGREEMENTS ---")
    edges = find_bfe_vs_ps_edges(df, min_diff=0.03)
    if len(edges) > 0:
        roi = edges["profit_flat"].mean() if "profit_flat" in edges.columns else np.nan
        print(f"  Matches where BFE overprices vs Pinnacle (edge>3%): {len(edges)}")
        print(f"  Win rate: {edges['Won'].mean():.3f}")
        print(f"  Avg edge: {edges['edge'].mean():.4f}")
        print(f"  Flat ROI on BFE: {roi:.4f} ({roi*100:.2f}%)")
    else:
        print("  No significant BFE vs Pinnacle edges found")


if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    full_exchange_report(df)
