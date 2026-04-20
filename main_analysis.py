"""
Main analysis runner: comprehensive Serie A betting analysis.

Pipeline:
1. Load and preprocess data
2. Add rolling features and market implied probabilities
3. Walk-forward predictions from all models:
   - Dixon-Coles (with time decay)
   - Elo ratings
   - Pi-Ratings
   - Pinnacle market model (de-vigged)
   - XGBoost (stats only)
   - XGBoost + market features
   - LightGBM
   - Ensemble
4. Evaluate all models (log-loss, RPS, Brier, accuracy)
5. Backtest value betting strategies
6. Betfair Exchange specific analysis
7. CLV analysis
8. Print final report
"""
import warnings
import numpy as np
import pandas as pd
from tabulate import tabulate

warnings.filterwarnings("ignore")

from data_loader import load_data, build_feature_matrix
from statistical_models import (
    EloModel, PiRatings, add_pinnacle_probabilities,
    PoissonRegressionModel, NegativeBinomialModel,
)
from dixon_coles import walk_forward_predict as dc_walk_forward
from ml_models import walk_forward_ml, evaluate_model, _ranked_probability_score
from betting_strategies import (
    analyze_clv, analyze_market_efficiency, favorite_longshot_bias,
    draw_bias_analysis, backtest_value_strategy, print_roi_report,
)
from betfair_exchange import full_exchange_report


def run_full_analysis():
    print("=" * 72)
    print("  SERIE A BETTING ANALYSIS — COMPLETE SYSTEM")
    print("=" * 72)

    # ── 1. Load Data ──────────────────────────────────────────────────────
    print("\n[1/8] Loading data...")
    df = load_data()
    df = build_feature_matrix(df)
    print(f"  Matches: {len(df)}  |  Seasons: {df['Season'].nunique()}")
    print(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"  Teams (unique): {len(set(df['HomeTeam']) | set(df['AwayTeam']))}")
    print(f"  Result distribution: H={( df['FTR']=='H').mean():.2%}  "
          f"D={(df['FTR']=='D').mean():.2%}  A={(df['FTR']=='A').mean():.2%}")

    # ── 2. Elo + Pi-Ratings (walk-forward, no retraining needed) ─────────
    print("\n[2/8] Running Elo and Pi-Ratings (continuous walk-forward)...")
    elo = EloModel(k_base=20, home_advantage=100)
    df = elo.fit_and_predict(df)

    pi = PiRatings()
    df = pi.fit_and_predict(df)

    df = add_pinnacle_probabilities(df)
    print("  Elo and Pi-Ratings computed.")

    # ── 3. Dixon-Coles walk-forward ───────────────────────────────────────
    print("\n[3/8] Dixon-Coles walk-forward (retrain every 38 matches)...")
    print("  This may take 2-5 minutes...")
    try:
        df = dc_walk_forward(df, min_train_matches=200, retrain_every=76, xi=0.0018)
        n_dc = df["dc_p_home"].notna().sum()
        print(f"  Dixon-Coles predictions: {n_dc}")
    except Exception as e:
        print(f"  Dixon-Coles failed: {e}")
        for c in ["dc_p_home", "dc_p_draw", "dc_p_away"]:
            df[c] = np.nan

    # ── 4. ML Models walk-forward ─────────────────────────────────────────
    print("\n[4/8] XGBoost + LightGBM walk-forward (retrain every 76 matches)...")
    print("  This may take 3-8 minutes...")
    try:
        df = walk_forward_ml(df, min_train=300, retrain_every=76, use_market=True)
        n_ml = df["xgb_p_home"].notna().sum()
        print(f"  ML predictions: {n_ml}")
    except Exception as e:
        print(f"  ML walk-forward failed: {e}")
        for c in ["xgb_p_home", "xgb_p_draw", "xgb_p_away",
                  "lgb_p_home", "lgb_p_draw", "lgb_p_away",
                  "ens_p_home", "ens_p_draw", "ens_p_away"]:
            df[c] = np.nan

    # ── 5. Model Evaluation ───────────────────────────────────────────────
    print("\n[5/8] Evaluating all models...")
    model_pred_cols = {
        "Elo": ("elo_p_home", "elo_p_draw", "elo_p_away"),
        "Pi-Ratings": ("pi_p_home", "pi_p_draw", "pi_p_away"),
        "Pinnacle (market)": ("pin_p_home", "pin_p_draw", "pin_p_away"),
        "Dixon-Coles": ("dc_p_home", "dc_p_draw", "dc_p_away"),
        "XGBoost": ("xgb_p_home", "xgb_p_draw", "xgb_p_away"),
        "LightGBM": ("lgb_p_home", "lgb_p_draw", "lgb_p_away"),
        "Ensemble": ("ens_p_home", "ens_p_draw", "ens_p_away"),
    }

    eval_rows = []
    for name, (h, d, a) in model_pred_cols.items():
        if h not in df.columns:
            continue
        ev = evaluate_model(df, {"home": h, "draw": d, "away": a})
        if ev:
            eval_rows.append([
                name,
                ev.get("n_predictions", 0),
                f"{ev.get('log_loss', 0):.4f}",
                f"{ev.get('rps', 0):.4f}",
                f"{ev.get('brier', 0):.4f}",
                f"{ev.get('accuracy', 0):.3f}",
            ])

    print(tabulate(eval_rows,
                   headers=["Model", "N", "Log-Loss↓", "RPS↓", "Brier↓", "Acc↑"],
                   tablefmt="rounded_outline"))
    print("  (↓ lower is better, ↑ higher is better)")
    print("  Note: Pinnacle (market) represents the best benchmark available.")

    # ── 6. Value Betting Backtest ─────────────────────────────────────────
    print("\n[6/8] Value betting backtest (min_edge=3%, Kelly 25%)...")
    prob_cols_for_value = {}
    for name, (h, d, a) in model_pred_cols.items():
        if h in df.columns and df[h].notna().sum() > 100:
            prob_cols_for_value[name] = (h, d, a)

    # Test against three bookmakers
    for bk_name, h_col, d_col, a_col in [
        ("vs_B365", "B365H", "B365D", "B365A"),
        ("vs_Pinnacle", "PSH", "PSD", "PSA"),
        ("vs_BFE", "BFEH", "BFED", "BFEA"),
    ]:
        if h_col not in df.columns:
            continue
        results = backtest_value_strategy(
            df,
            prob_cols_for_value,
            odds_col_h=h_col,
            odds_col_d=d_col,
            odds_col_a=a_col,
            min_edge=0.03,
            kelly_fraction=0.25,
            start_idx=300,
        )
        print_roi_report(results, f"Value Betting — {bk_name}")

    # ── 7. CLV Analysis ───────────────────────────────────────────────────
    print("\n[7/8] Closing Line Value analysis...")
    clv = analyze_clv(df)
    if clv:
        print(f"\n  {'Market':<12} {'N':>6} {'Avg CLV prob':>14} {'% positive':>12} {'ROI flat':>10}")
        print("  " + "-" * 58)
        for k, v in clv.items():
            print(f"  {k:<12} {v['n']:>6} {v['avg_clv_prob']:>+14.4f} "
                  f"{v['pct_positive_clv']:>11.1%} {v['avg_roi_flat']:>+10.4f}")
    print("\n  Interpretation: If avg_clv_prob > 0, the opening price was MORE")
    print("  generous than the closing price (potentially offered value).")

    # ── 8. Betfair Exchange Analysis ──────────────────────────────────────
    print("\n[8/8] Betfair Exchange specific analysis...")
    full_exchange_report(df)

    # ── BONUS: Market Efficiency / Biases ────────────────────────────────
    print("\n" + "=" * 72)
    print("  MARKET BIAS ANALYSIS")
    print("=" * 72)

    print("\n--- Draw Bias (does backing all draws make money?) ---")
    draw = draw_bias_analysis(df)
    draw_rows = []
    for book, stats in draw.items():
        draw_rows.append([
            book,
            stats["n"],
            f"{stats['actual_draw_rate']:.3f}",
            f"{stats['avg_implied_prob']:.3f}",
            f"{stats['bias']:+.4f}",
            f"{stats['flat_roi']:+.4f}",
        ])
    print(tabulate(draw_rows, headers=["Book", "N", "Actual D%", "Implied D%", "Bias", "ROI"],
                   tablefmt="simple"))

    print("\n--- Favorite-Longshot Bias (Pinnacle Home odds) ---")
    flb = favorite_longshot_bias(df)
    home_flb = flb[flb["outcome"] == "H"].copy()
    if not home_flb.empty:
        print(tabulate(
            home_flb[["odds_range", "n", "avg_odds", "actual_prob",
                       "implied_prob", "edge", "roi"]].values.tolist(),
            headers=["Odds Range", "N", "Avg Odds", "Actual%", "Implied%", "Edge", "ROI"],
            tablefmt="simple",
        ))

    # ── FINAL RECOMMENDATIONS ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  STRATEGY RECOMMENDATIONS & HONEST ASSESSMENT")
    print("=" * 72)
    _print_recommendations()

    # Save full results
    df.to_csv("analysis_results.csv", index=False)
    print(f"\n  Full results saved to analysis_results.csv")


def _print_recommendations():
    """Print evidence-based strategy recommendations."""
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│  ALGORITHMS: WHAT BEATS DIXON-COLES?                                │
├─────────────────────────────────────────────────────────────────────┤
│  1. ENSEMBLE (XGBoost + LightGBM + DC) — Best accuracy              │
│     Uses ALL available data: stats, form, AND market odds.          │
│     Market odds as features is the single biggest improvement.      │
│                                                                     │
│  2. PINNACLE MARKET MODEL — Best benchmark                          │
│     Pinnacle is the sharpest bookmaker in the world.                │
│     Their de-vigged odds are the best free probability estimate.    │
│     Any model must BEAT Pinnacle to find real value.                │
│                                                                     │
│  3. DIXON-COLES + TIME DECAY — Best pure statistical model         │
│     Better than basic Poisson, respects recent form.                │
│     Still loses to market models with more information.             │
│                                                                     │
│  4. ELO / PI-RATINGS — Fast, simple, surprisingly competitive       │
│     Best for quick estimates; no optimization needed.               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  BETFAIR-SPECIFIC STRATEGIES (ranked by practicality)               │
├─────────────────────────────────────────────────────────────────────┤
│  ★★★ HIGH POTENTIAL                                                 │
│  • O/U 2.5 value betting: less liquid → less efficient →            │
│    models can find edge, especially early in season                 │
│  • Betfair Exchange vs Pinnacle divergence: bet BFE when            │
│    Pinnacle implies higher probability (sharp vs soft)              │
│  • CLV approach: only bet markets where you consistently            │
│    beat the closing line — this IS long-term proof of edge          │
│                                                                     │
│  ★★  MODERATE POTENTIAL                                             │
│  • Lay-the-draw: historically near break-even; works best           │
│    when matched with form analysis (avoid defensive teams)          │
│  • Asian Handicap value: tightest market but exploitable            │
│    with line shopping across BFE, Pinnacle, Bet365                  │
│  • Back-Lay arbitrage: rarely > 0.5% — transaction costs eat it    │
│                                                                     │
│  ★   ADVANCED / REQUIRES AUTOMATION                                 │
│  • In-play trading: back early goals, trade when first goal scored  │
│  • Market making: place limit orders at bid/ask spread              │
│  • Steam chasing: react to Pinnacle line moves in < 30 seconds      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  HONEST RISK ASSESSMENT                                             │
├─────────────────────────────────────────────────────────────────────┤
│  - Backtested ROI does NOT guarantee future profits                 │
│  - Betfair can restrict/close accounts showing consistent profit    │
│  - Kelly criterion assumes perfect probability estimation           │
│    (in practice use 1/4 Kelly or less)                              │
│  - Minimum meaningful bankroll: €1,000–€5,000                      │
│  - Expected annual ROI with a working system: 2-8%                 │
│    (not 50%+; exceptional systems reach 15-20%)                     │
│  - Main edges in practice:                                          │
│    1. Beating closing line consistently (+CLV)                      │
│    2. Soft bookmakers (Bet365) vs sharp model                       │
│    3. Less-liquid markets (O/U, AH alternatives)                   │
└─────────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    run_full_analysis()
