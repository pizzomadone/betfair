"""
Dixon-Coles model (1997) with time-decay extension.

Dixon & Coles (1997) "Modelling Association Football Scores and Inefficiencies
in the Football Betting Market", Applied Statistics 46(2), 265-280.

Extensions implemented:
- Time decay weighting (Rue & Salvesen 2000)
- Home advantage parameter
- Low-score correction factor ρ
- Prediction: P(H), P(D), P(A) and expected goals
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from typing import Dict, Tuple, Optional


def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles low-score correction factor τ."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    elif x == 1 and y == 0:
        return 1.0 + mu * rho
    elif x == 0 and y == 1:
        return 1.0 + lam * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _log_likelihood(
    params: np.ndarray,
    teams: list,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
) -> float:
    """Vectorized negative log-likelihood for Dixon-Coles model."""
    n = n_teams
    home_adv = np.exp(params[2 * n])
    rho = -0.9 / (1.0 + np.exp(-params[2 * n + 1]))

    attack = np.exp(params[:n])
    defense = np.exp(params[n:2 * n])

    lam = attack[home_idx] * defense[away_idx] * home_adv
    mu = attack[away_idx] * defense[home_idx]

    if np.any(lam <= 0) or np.any(mu <= 0):
        return 1e12

    # Vectorized tau correction (only affects scores 0-0, 1-0, 0-1, 1-1)
    tau = np.ones(len(home_goals))
    m00 = (home_goals == 0) & (away_goals == 0)
    m10 = (home_goals == 1) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m11 = (home_goals == 1) & (away_goals == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m11] = 1.0 - rho

    if np.any(tau <= 0):
        return 1e12

    ll = weights * (
        np.log(tau)
        + poisson.logpmf(home_goals, lam)
        + poisson.logpmf(away_goals, mu)
    )
    return -ll.sum()


class DixonColesModel:
    """
    Dixon-Coles Poisson model with optional time-decay weighting.

    Parameters
    ----------
    xi : float
        Time-decay constant. 0 = no decay. Typical value: 0.0018 (half-life ~385 days).
    max_goals : int
        Maximum goals to consider when computing match outcome probabilities.
    """

    def __init__(self, xi: float = 0.0018, max_goals: int = 10):
        self.xi = xi
        self.max_goals = max_goals
        self.params_: Optional[np.ndarray] = None
        self.teams_: Optional[list] = None
        self.n_teams_: int = 0
        self.fit_date_: Optional[pd.Timestamp] = None

    def fit(
        self,
        df: pd.DataFrame,
        reference_date: Optional[pd.Timestamp] = None,
    ) -> "DixonColesModel":
        """
        Fit model on historical matches.

        Parameters
        ----------
        df : DataFrame with columns HomeTeam, AwayTeam, FTHG, FTAG, Date
        reference_date : date to compute time-decay weights from. Default: max(Date).
        """
        df = df.dropna(subset=["FTHG", "FTAG", "HomeTeam", "AwayTeam", "Date"])
        if reference_date is None:
            reference_date = df["Date"].max()
        self.fit_date_ = reference_date

        self.teams_ = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        self.n_teams_ = len(self.teams_)
        team_idx = {t: i for i, t in enumerate(self.teams_)}

        home_idx = df["HomeTeam"].map(team_idx).values
        away_idx = df["AwayTeam"].map(team_idx).values
        home_goals = df["FTHG"].values.astype(int)
        away_goals = df["FTAG"].values.astype(int)

        days_elapsed = (reference_date - df["Date"]).dt.days.values.astype(float)
        weights = np.exp(-self.xi * days_elapsed)

        n = self.n_teams_
        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = np.log(1.2)

        result = minimize(
            _log_likelihood,
            x0,
            args=(self.teams_, home_idx, away_idx, home_goals, away_goals, weights, n),
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-7},
        )
        self.params_ = result.x
        return self

    def _get_lambdas(self, home_team: str, away_team: str) -> Tuple[float, float, float]:
        """Return (lambda_home, lambda_away, rho) for a given matchup."""
        if self.params_ is None:
            raise RuntimeError("Model not fitted")
        n = self.n_teams_
        team_idx = {t: i for i, t in enumerate(self.teams_)}

        if home_team not in team_idx or away_team not in team_idx:
            # Unknown team: use average parameters
            atk = np.exp(np.mean(self.params_[:n]))
            def_ = np.exp(np.mean(self.params_[n:2 * n]))
            home_adv = np.exp(self.params_[2 * n])
            rho = -0.9 / (1.0 + np.exp(-self.params_[2 * n + 1]))
            lam = atk * def_ * home_adv
            mu = atk * def_
            return lam, mu, rho

        hi = team_idx[home_team]
        ai = team_idx[away_team]
        home_adv = np.exp(self.params_[2 * n])
        rho = -0.9 / (1.0 + np.exp(-self.params_[2 * n + 1]))

        lam = np.exp(self.params_[hi]) * np.exp(self.params_[n + ai]) * home_adv
        mu = np.exp(self.params_[ai]) * np.exp(self.params_[n + hi])
        return lam, mu, rho

    def predict_score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """Return probability matrix P[i,j] = P(home=i, away=j goals)."""
        lam, mu, rho = self._get_lambdas(home_team, away_team)
        mg = self.max_goals
        h_pmf = poisson.pmf(np.arange(mg + 1), lam)
        a_pmf = poisson.pmf(np.arange(mg + 1), mu)
        matrix = np.outer(h_pmf, a_pmf)
        # Apply tau corrections only for low scores
        for i in range(min(2, mg + 1)):
            for j in range(min(2, mg + 1)):
                matrix[i, j] *= _tau(i, j, lam, mu, rho)
        matrix /= matrix.sum()
        return matrix

    def predict_1x2(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Return dict with P(H), P(D), P(A) and expected goals."""
        matrix = self.predict_score_matrix(home_team, away_team)
        p_home = float(np.tril(matrix, -1).sum())    # home > away
        p_draw = float(np.trace(matrix))              # home == away
        p_away = float(np.triu(matrix, 1).sum())      # away > home
        lam, mu, _ = self._get_lambdas(home_team, away_team)
        return {
            "p_home": p_home,
            "p_draw": p_draw,
            "p_away": p_away,
            "exp_home_goals": lam,
            "exp_away_goals": mu,
        }

    def predict_over_under(
        self, home_team: str, away_team: str, line: float = 2.5
    ) -> Dict[str, float]:
        """Return P(Over line) and P(Under line)."""
        matrix = self.predict_score_matrix(home_team, away_team)
        mg = self.max_goals
        p_over = 0.0
        for i in range(mg + 1):
            for j in range(mg + 1):
                if i + j > line:
                    p_over += matrix[i, j]
        return {"p_over": float(p_over), "p_under": float(1 - p_over)}

    def predict_asian_handicap(
        self, home_team: str, away_team: str, line: float
    ) -> Dict[str, float]:
        """
        Asian Handicap: home team starts with `line` goals adjustment.
        line < 0: home team disadvantaged (must win by more than |line| to cover)
        line > 0: home team advantaged
        Returns P(home covers), P(away covers).
        """
        matrix = self.predict_score_matrix(home_team, away_team)
        mg = self.max_goals
        p_home_cover = p_away_cover = p_push = 0.0
        for i in range(mg + 1):
            for j in range(mg + 1):
                diff = (i - j) + line   # adjusted goal difference
                if diff > 0:
                    p_home_cover += matrix[i, j]
                elif diff < 0:
                    p_away_cover += matrix[i, j]
                else:
                    p_push += matrix[i, j]
        return {
            "p_home_cover": float(p_home_cover),
            "p_away_cover": float(p_away_cover),
            "p_push": float(p_push),
        }

    def get_team_ratings(self) -> pd.DataFrame:
        """Return a DataFrame with attack/defense ratings for each team."""
        if self.params_ is None:
            raise RuntimeError("Model not fitted")
        n = self.n_teams_
        home_adv = np.exp(self.params_[2 * n])
        return pd.DataFrame({
            "Team": self.teams_,
            "Attack": np.exp(self.params_[:n]),
            "Defense": np.exp(self.params_[n:2 * n]),
            "AttackRank": pd.Series(np.exp(self.params_[:n])).rank(ascending=False).values,
            "DefenseRank": pd.Series(np.exp(self.params_[n:2 * n])).rank(ascending=True).values,
        }).sort_values("Attack", ascending=False).reset_index(drop=True), home_adv


def walk_forward_predict(
    df: pd.DataFrame,
    min_train_matches: int = 200,
    retrain_every: int = 76,
    xi: float = 0.0018,
    max_train_days: int = 730,
) -> pd.DataFrame:
    """
    Walk-forward validation of Dixon-Coles model.

    Uses a rolling window of max_train_days (default 2 years) to keep
    training set size manageable. Retrains every retrain_every matches.
    Returns DataFrame with predictions appended.
    """
    df = df.copy().reset_index(drop=True)
    preds = []

    model = DixonColesModel(xi=xi)
    n = len(df)

    for start in range(min_train_matches, n, retrain_every):
        ref_date = df.iloc[start - 1]["Date"]
        cutoff = ref_date - pd.Timedelta(days=max_train_days)
        train = df.iloc[:start]
        train = train[train["Date"] >= cutoff]
        if len(train) < 80:
            train = df.iloc[max(0, start - 400):start]

        test = df.iloc[start:start + retrain_every]

        try:
            model.fit(train, reference_date=ref_date)
        except Exception:
            continue

        for _, row in test.iterrows():
            ht, at = row["HomeTeam"], row["AwayTeam"]
            try:
                pred = model.predict_1x2(ht, at)
                ou = model.predict_over_under(ht, at, 2.5)
                pred.update(ou)
            except Exception:
                pred = {
                    "p_home": np.nan, "p_draw": np.nan, "p_away": np.nan,
                    "exp_home_goals": np.nan, "exp_away_goals": np.nan,
                    "p_over": np.nan, "p_under": np.nan,
                }
            pred["match_id"] = row.name
            preds.append(pred)

    pred_df = pd.DataFrame(preds).set_index("match_id")
    for col in pred_df.columns:
        df.loc[pred_df.index, f"dc_{col}"] = pred_df[col]

    return df


if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    print(f"Fitting Dixon-Coles on {len(df)} matches...")
    model = DixonColesModel(xi=0.0018)
    model.fit(df)
    ratings, home_adv = model.get_team_ratings()
    print(f"\nHome advantage factor: {home_adv:.3f}")
    print("\nTop 5 Attack ratings:")
    print(ratings.head(5)[["Team", "Attack", "Defense"]].to_string(index=False))
    print("\nSample prediction — Juventus vs Inter:")
    try:
        p = model.predict_1x2("Juventus", "Inter")
        print(f"  P(H)={p['p_home']:.3f}  P(D)={p['p_draw']:.3f}  P(A)={p['p_away']:.3f}")
        print(f"  E[goals]: {p['exp_home_goals']:.2f} - {p['exp_away_goals']:.2f}")
    except Exception as e:
        print(f"  Error: {e}")
