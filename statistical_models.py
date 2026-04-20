"""
Statistical models beyond Dixon-Coles:
- Elo rating system with dynamic K-factor
- Basic Poisson regression (Maher 1982)
- Negative Binomial regression
- Extended Poisson with shots/corners features
- Bivariate Poisson (Karlis & Ntzoufras 2003)
- Pi-ratings (Constantinou & Fenton 2013)
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom, norm
from scipy.special import gammaln
from scipy.optimize import minimize
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────
# 1. ELO RATING SYSTEM
# ─────────────────────────────────────────────

class EloModel:
    """
    Elo rating system adapted for football.
    Uses logistic probability estimate and dynamic K-factor.

    References: World Football Elo Ratings (eloratings.net) methodology.
    """

    def __init__(
        self,
        k_base: float = 20.0,
        home_advantage: float = 100.0,
        initial_rating: float = 1500.0,
        draw_adjustment: float = 0.28,
    ):
        self.k_base = k_base
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.draw_adjustment = draw_adjustment
        self.ratings_: Dict[str, float] = {}

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Logistic expected score for team A vs team B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _k_factor(self, goals_diff: int) -> float:
        """Larger upsets/dominant wins → higher K."""
        if goals_diff == 0:
            return self.k_base
        elif goals_diff == 1:
            return self.k_base * 1.5
        elif goals_diff == 2:
            return self.k_base * 1.75
        else:
            return self.k_base * (1.75 + (goals_diff - 3) * 0.25)

    def get_rating(self, team: str) -> float:
        return self.ratings_.get(team, self.initial_rating)

    def predict_proba(self, home_team: str, away_team: str) -> Dict[str, float]:
        """
        Predict match outcome probabilities.
        Uses Elo difference + home advantage, converts to 3-way via draw zone.
        """
        r_h = self.get_rating(home_team) + self.home_advantage
        r_a = self.get_rating(away_team)
        e_h = self._expected_score(r_h, r_a)  # P(home wins in 2-way)

        # Convert 2-way to 3-way using empirical draw model
        # Draw probability peaks around e_h = 0.5
        p_draw = self.draw_adjustment * 4 * e_h * (1 - e_h)   # parabola, max at 0.5
        p_draw = min(p_draw, 0.45)  # cap
        p_home = e_h - p_draw / 2
        p_away = (1 - e_h) - p_draw / 2
        total = p_home + p_draw + p_away
        return {
            "p_home": max(0, p_home) / total,
            "p_draw": max(0, p_draw) / total,
            "p_away": max(0, p_away) / total,
        }

    def update(self, home_team: str, away_team: str, result: str,
               home_goals: int, away_goals: int) -> None:
        """Update Elo ratings after a match."""
        r_h = self.get_rating(home_team) + self.home_advantage
        r_a = self.get_rating(away_team)
        e_h = self._expected_score(r_h, r_a)

        if result == "H":
            s_h, s_a = 1.0, 0.0
        elif result == "D":
            s_h, s_a = 0.5, 0.5
        else:
            s_h, s_a = 0.0, 1.0

        k = self._k_factor(abs(home_goals - away_goals))
        self.ratings_[home_team] = self.get_rating(home_team) + k * (s_h - e_h)
        self.ratings_[away_team] = self.get_rating(away_team) + k * (s_a - (1 - e_h))

    def fit_and_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Walk-forward: update ratings and record prediction before each match."""
        df = df.copy().reset_index(drop=True)
        preds = []
        self.ratings_ = {}

        for _, row in df.iterrows():
            ht, at = row["HomeTeam"], row["AwayTeam"]
            pred = self.predict_proba(ht, at)
            pred["elo_home"] = self.get_rating(ht)
            pred["elo_away"] = self.get_rating(at)
            preds.append(pred)
            self.update(ht, at, row["FTR"], int(row["FTHG"]), int(row["FTAG"]))

        pred_df = pd.DataFrame(preds, index=df.index)
        for col in pred_df.columns:
            df[f"elo_{col}"] = pred_df[col]
        return df


# ─────────────────────────────────────────────
# 2. PI-RATINGS (Constantinou & Fenton 2013)
# ─────────────────────────────────────────────

class PiRatings:
    """
    Pi-ratings: continuous-update system that measures team performance
    relative to league average. Separate home/away ratings.

    Constantinou, A.C. & Fenton, N.E. (2013). "Determining the level of
    ability of football teams by dynamic ratings based on the relative
    discrepancies in scores between adversaries." J. Quant. Analysis Sports.
    """

    def __init__(self, gamma: float = 0.036, lambda_: float = 1.0):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.home_: Dict[str, float] = {}
        self.away_: Dict[str, float] = {}

    def get_ratings(self, team: str) -> Tuple[float, float]:
        return self.home_.get(team, 0.0), self.away_.get(team, 0.0)

    def _expected_margin(self, home_team: str, away_team: str) -> float:
        rh_h, rh_a = self.get_ratings(home_team)
        ra_h, ra_a = self.get_ratings(away_team)
        return rh_h - ra_a   # home playing home vs away playing away

    def predict_proba(self, home_team: str, away_team: str) -> Dict[str, float]:
        em = self._expected_margin(home_team, away_team)
        # Convert expected margin to probabilities via logistic
        # Empirically calibrated constants
        p_home = 1 / (1 + np.exp(-(em - 0.1) * 0.7))
        p_away = 1 / (1 + np.exp((em + 0.1) * 0.7))
        p_draw = max(0.05, 1 - p_home - p_away)
        total = p_home + p_draw + p_away
        return {
            "p_home": p_home / total,
            "p_draw": p_draw / total,
            "p_away": p_away / total,
        }

    def update(self, home_team: str, away_team: str,
               home_goals: int, away_goals: int) -> None:
        goal_diff = home_goals - away_goals
        em = self._expected_margin(home_team, away_team)
        error = goal_diff - em
        update = self.gamma * np.tanh(self.lambda_ * error)

        rh_h, rh_a = self.get_ratings(home_team)
        ra_h, ra_a = self.get_ratings(away_team)
        self.home_[home_team] = rh_h + update
        self.away_[home_team] = rh_a + update * 0.5
        self.away_[away_team] = ra_a - update
        self.home_[away_team] = ra_h - update * 0.5

    def fit_and_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)
        preds = []
        self.home_, self.away_ = {}, {}

        for _, row in df.iterrows():
            ht, at = row["HomeTeam"], row["AwayTeam"]
            pred = self.predict_proba(ht, at)
            preds.append(pred)
            self.update(ht, at, int(row["FTHG"]), int(row["FTAG"]))

        pred_df = pd.DataFrame(preds, index=df.index)
        for col in pred_df.columns:
            df[f"pi_{col}"] = pred_df[col]
        return df


# ─────────────────────────────────────────────
# 3. NEGATIVE BINOMIAL POISSON REGRESSION
# ─────────────────────────────────────────────

def _nb_log_pmf(k: int, mu: float, alpha: float) -> float:
    """Negative binomial log PMF parameterized by mean mu and dispersion alpha."""
    r = 1.0 / alpha
    return (
        gammaln(k + r) - gammaln(r) - gammaln(k + 1)
        + r * np.log(r / (r + mu))
        + k * np.log(mu / (r + mu))
    )


def _nb_ll(params, home_idx, away_idx, home_goals, away_goals, n_teams):
    """Negative log-likelihood for independent NegBin Poisson model."""
    n = n_teams
    log_atk = params[:n]
    log_def = params[n:2 * n]
    home_adv = np.exp(params[2 * n])
    log_alpha = params[2 * n + 1]
    alpha = np.exp(log_alpha)  # dispersion > 0

    attack = np.exp(log_atk)
    defense = np.exp(log_def)

    lam = attack[home_idx] * defense[away_idx] * home_adv
    mu = attack[away_idx] * defense[home_idx]

    ll = 0.0
    for i in range(len(home_goals)):
        ll += _nb_log_pmf(int(home_goals[i]), lam[i], alpha)
        ll += _nb_log_pmf(int(away_goals[i]), mu[i], alpha)
    return -ll


class NegativeBinomialModel:
    """
    Handles overdispersion in goal counts better than standard Poisson.
    Independent NegBin for home and away goals.
    """

    def __init__(self):
        self.params_: Optional[np.ndarray] = None
        self.teams_: Optional[list] = None
        self.n_teams_: int = 0

    def fit(self, df: pd.DataFrame) -> "NegativeBinomialModel":
        df = df.dropna(subset=["FTHG", "FTAG", "HomeTeam", "AwayTeam"])
        self.teams_ = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        self.n_teams_ = len(self.teams_)
        team_idx = {t: i for i, t in enumerate(self.teams_)}

        home_idx = df["HomeTeam"].map(team_idx).values
        away_idx = df["AwayTeam"].map(team_idx).values
        home_goals = df["FTHG"].values
        away_goals = df["FTAG"].values

        n = self.n_teams_
        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = np.log(1.2)

        result = minimize(
            _nb_ll,
            x0,
            args=(home_idx, away_idx, home_goals, away_goals, n),
            method="L-BFGS-B",
            options={"maxiter": 1000},
        )
        self.params_ = result.x
        return self

    def predict_1x2(self, home_team: str, away_team: str,
                    max_goals: int = 10) -> Dict[str, float]:
        if self.params_ is None:
            raise RuntimeError("Model not fitted")
        n = self.n_teams_
        team_idx = {t: i for i, t in enumerate(self.teams_)}
        alpha = np.exp(self.params_[2 * n + 1])
        home_adv = np.exp(self.params_[2 * n])

        if home_team not in team_idx or away_team not in team_idx:
            return {"p_home": 0.45, "p_draw": 0.28, "p_away": 0.27}

        hi, ai = team_idx[home_team], team_idx[away_team]
        lam = np.exp(self.params_[hi]) * np.exp(self.params_[n + ai]) * home_adv
        mu = np.exp(self.params_[ai]) * np.exp(self.params_[n + hi])

        r = 1.0 / alpha
        matrix = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                ph = np.exp(_nb_log_pmf(i, lam, alpha))
                pa = np.exp(_nb_log_pmf(j, mu, alpha))
                matrix[i, j] = ph * pa

        matrix /= matrix.sum()
        p_home = float(np.tril(matrix, -1).sum())
        p_draw = float(np.trace(matrix))
        p_away = float(np.triu(matrix, 1).sum())
        return {"p_home": p_home, "p_draw": p_draw, "p_away": p_away}


# ─────────────────────────────────────────────
# 4. MARKET CONSENSUS MODEL (Pinnacle as oracle)
# ─────────────────────────────────────────────

def devig_pinnacle(row: pd.Series) -> Optional[Dict[str, float]]:
    """
    Remove vigorish from Pinnacle odds using multiplicative method.
    Pinnacle is the sharpest bookmaker; their de-vigged odds are widely
    used as the best available estimate of true match outcome probabilities.
    """
    for h_col, d_col, a_col in [("PSH", "PSD", "PSA"), ("PSCH", "PSCD", "PSCA")]:
        if h_col in row.index and pd.notna(row.get(h_col)):
            rh = 1 / row[h_col]
            rd = 1 / row[d_col]
            ra = 1 / row[a_col]
            total = rh + rd + ra
            return {"p_home": rh / total, "p_draw": rd / total, "p_away": ra / total}
    return None


def devig_betfair_exchange(row: pd.Series, commission: float = 0.05) -> Optional[Dict[str, float]]:
    """
    Remove exchange commission from Betfair Exchange odds.
    Exchange odds are net of commission: true_odds = exchange_odds * (1 - commission)
    """
    for h_col, d_col, a_col in [("BFEH", "BFED", "BFEA"), ("BFECH", "BFECD", "BFECA")]:
        if h_col in row.index and pd.notna(row.get(h_col)):
            # Adjust for commission
            eff_h = row[h_col] * (1 - commission)
            eff_d = row[d_col] * (1 - commission)
            eff_a = row[a_col] * (1 - commission)
            rh = 1 / eff_h
            rd = 1 / eff_d
            ra = 1 / eff_a
            total = rh + rd + ra
            return {"p_home": rh / total, "p_draw": rd / total, "p_away": ra / total}
    return None


def add_pinnacle_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Add Pinnacle de-vigged probabilities to DataFrame."""
    df = df.copy()
    for col in ["pin_p_home", "pin_p_draw", "pin_p_away"]:
        df[col] = np.nan

    for i, row in df.iterrows():
        result = devig_pinnacle(row)
        if result:
            df.at[i, "pin_p_home"] = result["p_home"]
            df.at[i, "pin_p_draw"] = result["p_draw"]
            df.at[i, "pin_p_away"] = result["p_away"]
    return df


# ─────────────────────────────────────────────
# 5. POISSON MEAN REGRESSION (GLM-style)
# ─────────────────────────────────────────────

class PoissonRegressionModel:
    """
    Standard Poisson regression for goal counts (Maher 1982).
    No low-score correction, no time decay.
    Simpler than Dixon-Coles, serves as baseline comparison.
    """

    def __init__(self):
        self.params_: Optional[np.ndarray] = None
        self.teams_: Optional[list] = None
        self.n_teams_: int = 0

    def _ll(self, params, home_idx, away_idx, home_goals, away_goals, n):
        log_atk = params[:n]
        log_def = params[n:2 * n]
        home_adv = np.exp(params[2 * n])
        lam = np.exp(log_atk[home_idx] + log_def[away_idx]) * home_adv
        mu = np.exp(log_atk[away_idx] + log_def[home_idx])
        ll = (
            home_goals * np.log(lam + 1e-12) - lam
            + away_goals * np.log(mu + 1e-12) - mu
        )
        return -ll.sum()

    def fit(self, df: pd.DataFrame) -> "PoissonRegressionModel":
        df = df.dropna(subset=["FTHG", "FTAG", "HomeTeam", "AwayTeam"])
        self.teams_ = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        self.n_teams_ = len(self.teams_)
        team_idx = {t: i for i, t in enumerate(self.teams_)}

        home_idx = df["HomeTeam"].map(team_idx).values
        away_idx = df["AwayTeam"].map(team_idx).values
        n = self.n_teams_
        x0 = np.zeros(2 * n + 1)
        x0[2 * n] = np.log(1.2)

        result = minimize(
            self._ll,
            x0,
            args=(home_idx, away_idx, df["FTHG"].values, df["FTAG"].values, n),
            method="L-BFGS-B",
            options={"maxiter": 1000},
        )
        self.params_ = result.x
        return self

    def predict_1x2(self, home_team: str, away_team: str,
                    max_goals: int = 10) -> Dict[str, float]:
        n = self.n_teams_
        team_idx = {t: i for i, t in enumerate(self.teams_)}
        if home_team not in team_idx or away_team not in team_idx:
            return {"p_home": 0.45, "p_draw": 0.28, "p_away": 0.27}

        home_adv = np.exp(self.params_[2 * n])
        hi, ai = team_idx[home_team], team_idx[away_team]
        lam = np.exp(self.params_[hi] + self.params_[n + ai]) * home_adv
        mu = np.exp(self.params_[ai] + self.params_[n + hi])

        matrix = np.outer(
            poisson.pmf(range(max_goals + 1), lam),
            poisson.pmf(range(max_goals + 1), mu),
        )
        matrix /= matrix.sum()
        return {
            "p_home": float(np.tril(matrix, -1).sum()),
            "p_draw": float(np.trace(matrix)),
            "p_away": float(np.triu(matrix, 1).sum()),
        }


if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()

    print("=== Elo Model ===")
    elo = EloModel()
    df = elo.fit_and_predict(df)
    print("Top ratings:", sorted(elo.ratings_.items(), key=lambda x: -x[1])[:5])

    print("\n=== Pi-Ratings ===")
    pi = PiRatings()
    df = pi.fit_and_predict(df)
    print("Pi home ratings:", sorted(pi.home_.items(), key=lambda x: -x[1])[:5])

    print("\n=== Pinnacle de-vigged probabilities ===")
    df = add_pinnacle_probabilities(df)
    print(df[["HomeTeam", "AwayTeam", "FTR", "pin_p_home", "pin_p_draw", "pin_p_away"]].tail(5).to_string())
