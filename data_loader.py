"""
Data loading, cleaning, and feature engineering for Serie A betting analysis.
"""
import subprocess
import pandas as pd
import numpy as np
from io import StringIO


def load_data(filepath="serie-A.csv"):
    """Load Serie A data, restoring from git history if file is empty."""
    try:
        df = pd.read_csv(filepath)
        if len(df) < 10:
            raise ValueError("File too small, restoring from git")
    except Exception:
        result = subprocess.run(
            ["git", "show", "bbee22f:I1_combined.csv"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise FileNotFoundError("Cannot load data from CSV or git history")
        df = pd.read_csv(StringIO(result.stdout))
        df.to_csv(filepath, index=False)

    return _preprocess(df)


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["Date", "FTHG", "FTAG", "HomeTeam", "AwayTeam"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["TotalGoals"] = df["FTHG"] + df["FTAG"]
    df["GoalDiff"] = df["FTHG"] - df["FTAG"]

    # Numeric outcome: 1=Home, 0=Draw, -1=Away
    df["Result"] = df["FTR"].map({"H": 1, "D": 0, "A": -1})

    # Convert all odds/numeric columns
    skip = {"Div", "Date", "HomeTeam", "AwayTeam", "FTR", "HTR"}
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Season label (Italian football: season starts Aug, ends May)
    df["Season"] = df["Date"].apply(
        lambda d: f"{d.year}/{d.year+1}" if d.month >= 8 else f"{d.year-1}/{d.year}"
    )

    # Remove rows where result is unknown (NaN)
    df = df.dropna(subset=["Result"]).reset_index(drop=True)
    return df


def add_rolling_features(df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """
    Add rolling team-level features computed only from past matches
    (no data leakage).
    """
    df = df.copy().reset_index(drop=True)
    teams = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())

    # Accumulators keyed by team
    stats = {t: {
        "goals_scored": [], "goals_conceded": [],
        "shots": [], "shots_against": [],
        "points": [], "form": [],  # last window results
    } for t in teams}

    feats = {
        "home_atk": np.full(len(df), np.nan),
        "home_def": np.full(len(df), np.nan),
        "away_atk": np.full(len(df), np.nan),
        "away_def": np.full(len(df), np.nan),
        "home_form": np.full(len(df), np.nan),
        "away_form": np.full(len(df), np.nan),
        "home_ppg": np.full(len(df), np.nan),
        "away_ppg": np.full(len(df), np.nan),
    }

    def tail_mean(lst, w):
        s = lst[-w:] if len(lst) >= w else lst
        return float(np.mean(s)) if s else np.nan

    for i, row in df.iterrows():
        ht, at = row["HomeTeam"], row["AwayTeam"]
        h, a = stats[ht], stats[at]

        feats["home_atk"][i] = tail_mean(h["goals_scored"], window)
        feats["home_def"][i] = tail_mean(h["goals_conceded"], window)
        feats["away_atk"][i] = tail_mean(a["goals_scored"], window)
        feats["away_def"][i] = tail_mean(a["goals_conceded"], window)
        feats["home_form"][i] = tail_mean(h["form"], window)
        feats["away_form"][i] = tail_mean(a["form"], window)
        feats["home_ppg"][i] = tail_mean(h["points"], window)
        feats["away_ppg"][i] = tail_mean(a["points"], window)

        # Update after storing features (no leakage)
        hg, ag = int(row["FTHG"]), int(row["FTAG"])
        r = row["FTR"]

        h["goals_scored"].append(hg)
        h["goals_conceded"].append(ag)
        a["goals_scored"].append(ag)
        a["goals_conceded"].append(hg)

        h_pts = 3 if r == "H" else (1 if r == "D" else 0)
        a_pts = 3 if r == "A" else (1 if r == "D" else 0)
        h["points"].append(h_pts)
        a["points"].append(a_pts)

        h["form"].append(1 if r == "H" else (0.5 if r == "D" else 0))
        a["form"].append(1 if r == "A" else (0.5 if r == "D" else 0))

        hs = row.get("HS", np.nan)
        as_ = row.get("AS", np.nan)
        if not np.isnan(hs):
            h["shots"].append(hs); a["shots_against"].append(hs)
        if not np.isnan(as_):
            a["shots"].append(as_); h["shots_against"].append(as_)

    for k, v in feats.items():
        df[k] = v

    df["form_diff"] = df["home_form"] - df["away_form"]
    df["atk_diff"] = df["home_atk"] - df["away_atk"]
    df["def_diff"] = df["home_def"] - df["away_def"]
    return df


def get_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract implied probabilities from market odds (no leakage risk — odds are pre-match)."""
    df = df.copy()

    def implied_prob(odds_h, odds_d, odds_a):
        """Convert decimal odds to implied probabilities (Pinnacle no-vig method)."""
        raw_h = 1 / odds_h
        raw_d = 1 / odds_d
        raw_a = 1 / odds_a
        total = raw_h + raw_d + raw_a
        return raw_h / total, raw_d / total, raw_a / total

    for prefix, h_col, d_col, a_col in [
        ("ps", "PSH", "PSD", "PSA"),
        ("b365", "B365H", "B365D", "B365A"),
        ("bfe", "BFEH", "BFED", "BFEA"),
        ("avg", "AvgH", "AvgD", "AvgA"),
        ("max", "MaxH", "MaxD", "MaxA"),
    ]:
        if h_col in df.columns:
            mask = df[[h_col, d_col, a_col]].notna().all(axis=1)
            df.loc[mask, f"ip_{prefix}_h"] = df.loc[mask].apply(
                lambda r: implied_prob(r[h_col], r[d_col], r[a_col])[0], axis=1
            )
            df.loc[mask, f"ip_{prefix}_d"] = df.loc[mask].apply(
                lambda r: implied_prob(r[h_col], r[d_col], r[a_col])[1], axis=1
            )
            df.loc[mask, f"ip_{prefix}_a"] = df.loc[mask].apply(
                lambda r: implied_prob(r[h_col], r[d_col], r[a_col])[2], axis=1
            )

    # Overround (bookmaker margin)
    for prefix, h_col, d_col, a_col in [
        ("b365", "B365H", "B365D", "B365A"),
        ("ps", "PSH", "PSD", "PSA"),
        ("bfe", "BFEH", "BFED", "BFEA"),
    ]:
        if h_col in df.columns:
            mask = df[[h_col, d_col, a_col]].notna().all(axis=1)
            df.loc[mask, f"overround_{prefix}"] = (
                1 / df.loc[mask, h_col]
                + 1 / df.loc[mask, d_col]
                + 1 / df.loc[mask, a_col]
            )

    return df


def get_ou_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Over/Under 2.5 market features."""
    df = df.copy()
    for prefix, o_col, u_col in [
        ("bfe", "BFE>2.5", "BFE<2.5"),
        ("b365", "B365>2.5", "B365<2.5"),
        ("avg", "Avg>2.5", "Avg<2.5"),
        ("ps", "P>2.5", "P<2.5"),
    ]:
        if o_col in df.columns and u_col in df.columns:
            mask = df[[o_col, u_col]].notna().all(axis=1)
            total = 1 / df.loc[mask, o_col] + 1 / df.loc[mask, u_col]
            df.loc[mask, f"ip_{prefix}_over25"] = (1 / df.loc[mask, o_col]) / total
            df.loc[mask, f"ip_{prefix}_under25"] = (1 / df.loc[mask, u_col]) / total
            df.loc[mask, f"ou_overround_{prefix}"] = total

    df["OverResult"] = (df["TotalGoals"] > 2.5).astype(float)
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Combine rolling stats + market implied probabilities into one feature matrix."""
    df = add_rolling_features(df)
    df = get_market_features(df)
    df = get_ou_features(df)
    return df


if __name__ == "__main__":
    df = load_data()
    df = build_feature_matrix(df)
    print(f"Loaded {len(df)} matches ({df['Season'].nunique()} seasons)")
    print(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"Teams: {sorted(df['HomeTeam'].unique())}")
    print(df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
              "home_atk", "away_atk", "ip_ps_h", "ip_ps_d", "ip_ps_a"]].tail(10).to_string())
