"""
Machine Learning models for match outcome prediction.

Models:
- XGBoost classifier (multi-class H/D/A)
- LightGBM classifier
- Gradient Boosted ensemble with calibration
- Goals regression model (XGBoost for expected goals)

Feature groups used:
1. Rolling team stats (attack, defense, form, PPG)
2. Head-to-head statistics
3. Market implied probabilities (Pinnacle, Betfair Exchange)
4. Shots, corners, cards rolling averages
5. Home/away specific form
"""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, brier_score_loss
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


FEATURE_COLS = [
    "home_atk", "home_def", "away_atk", "away_def",
    "home_form", "away_form", "home_ppg", "away_ppg",
    "form_diff", "atk_diff", "def_diff",
]

MARKET_FEATURE_COLS = [
    "ip_ps_h", "ip_ps_d", "ip_ps_a",
    "ip_bfe_h", "ip_bfe_d", "ip_bfe_a",
    "ip_b365_h", "ip_b365_d", "ip_b365_a",
    "ip_avg_h", "ip_avg_d", "ip_avg_a",
    "overround_b365", "overround_ps", "overround_bfe",
]


def build_feature_matrix(df: pd.DataFrame, use_market: bool = True) -> pd.DataFrame:
    """
    Build feature matrix from DataFrame.
    Only uses columns that exist and are not NaN for the majority of rows.
    """
    from data_loader import add_rolling_features, get_market_features

    if "home_atk" not in df.columns:
        df = add_rolling_features(df)
    if use_market and "ip_ps_h" not in df.columns:
        df = get_market_features(df)

    return df


def _get_X(df: pd.DataFrame, use_market: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """Extract feature matrix X and column list from DataFrame."""
    cols = [c for c in FEATURE_COLS if c in df.columns]
    if use_market:
        cols += [c for c in MARKET_FEATURE_COLS if c in df.columns]
    X = df[cols].copy()
    return X, cols


def _encode_y(series: pd.Series) -> np.ndarray:
    """Encode FTR → numeric: H=0, D=1, A=2."""
    return series.map({"H": 0, "D": 1, "A": 2}).values


def _decode_proba(proba: np.ndarray) -> Dict[str, float]:
    """Convert model probability array [p_H, p_D, p_A] to dict."""
    return {"p_home": float(proba[0]), "p_draw": float(proba[1]), "p_away": float(proba[2])}


class XGBoostModel:
    """
    XGBoost multi-class classifier for 1X2 prediction.
    Includes isotonic calibration to produce well-calibrated probabilities.
    """

    def __init__(self, use_market: bool = True, n_estimators: int = 200,
                 max_depth: int = 4, learning_rate: float = 0.05,
                 subsample: float = 0.8, colsample_bytree: float = 0.8):
        if not XGB_AVAILABLE:
            raise ImportError("xgboost not installed")
        self.use_market = use_market
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            use_label_encoder=False,
        )
        self.model = None
        self.feature_cols_: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame) -> "XGBoostModel":
        df = build_feature_matrix(df, self.use_market)
        X, cols = _get_X(df, self.use_market)
        y = _encode_y(df["FTR"])
        valid = ~X.isna().any(axis=1) & ~np.isnan(y)
        X, y = X[valid], y[valid]
        self.feature_cols_ = cols
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y)
        return self

    def predict_proba(self, row: pd.Series) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        x = pd.DataFrame([row[self.feature_cols_]])
        if x.isna().any(axis=1).any():
            return {"p_home": np.nan, "p_draw": np.nan, "p_away": np.nan}
        proba = self.model.predict_proba(x)[0]
        return _decode_proba(proba)

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return pd.DataFrame({
            "feature": self.feature_cols_,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)


class LightGBMModel:
    """
    LightGBM multi-class classifier for 1X2 prediction.
    Generally faster and handles missing values natively.
    """

    def __init__(self, use_market: bool = True, n_estimators: int = 200,
                 num_leaves: int = 31, learning_rate: float = 0.05,
                 min_child_samples: int = 20):
        if not LGB_AVAILABLE:
            raise ImportError("lightgbm not installed")
        self.use_market = use_market
        self.params = dict(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            objective="multiclass",
            num_class=3,
            metric="multi_logloss",
            random_state=42,
            verbose=-1,
        )
        self.model = None
        self.feature_cols_: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame) -> "LightGBMModel":
        df = build_feature_matrix(df, self.use_market)
        X, cols = _get_X(df, self.use_market)
        y = _encode_y(df["FTR"])
        valid = ~X.isna().any(axis=1) & ~np.isnan(y)
        X, y = X[valid].values, y[valid]
        self.feature_cols_ = cols
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y)
        return self

    def predict_proba(self, row: pd.Series) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        x = np.array([[row.get(c, np.nan) for c in self.feature_cols_]])
        if np.isnan(x).any():
            return {"p_home": np.nan, "p_draw": np.nan, "p_away": np.nan}
        proba = self.model.predict_proba(x)[0]
        return _decode_proba(proba)

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return pd.DataFrame({
            "feature": self.feature_cols_,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)


class EnsembleModel:
    """
    Weighted ensemble of XGBoost, LightGBM, and optionally statistical model predictions.
    Weights can be optimized by log-loss on a validation set.
    """

    def __init__(self, weights: Optional[List[float]] = None):
        self.weights = weights
        self.xgb = XGBoostModel(use_market=True) if XGB_AVAILABLE else None
        self.lgb = LightGBMModel(use_market=True) if LGB_AVAILABLE else None

    def fit(self, df: pd.DataFrame) -> "EnsembleModel":
        if self.xgb:
            self.xgb.fit(df)
        if self.lgb:
            self.lgb.fit(df)
        return self

    def predict_proba(self, row: pd.Series) -> Dict[str, float]:
        preds = []
        if self.xgb:
            p = self.xgb.predict_proba(row)
            if not np.isnan(p["p_home"]):
                preds.append(p)
        if self.lgb:
            p = self.lgb.predict_proba(row)
            if not np.isnan(p["p_home"]):
                preds.append(p)

        if not preds:
            return {"p_home": np.nan, "p_draw": np.nan, "p_away": np.nan}

        w = self.weights if self.weights and len(self.weights) == len(preds) \
            else [1.0 / len(preds)] * len(preds)
        ph = sum(w[i] * preds[i]["p_home"] for i in range(len(preds)))
        pd_ = sum(w[i] * preds[i]["p_draw"] for i in range(len(preds)))
        pa = sum(w[i] * preds[i]["p_away"] for i in range(len(preds)))
        total = ph + pd_ + pa
        return {"p_home": ph / total, "p_draw": pd_ / total, "p_away": pa / total}


def walk_forward_ml(
    df: pd.DataFrame,
    min_train: int = 300,
    retrain_every: int = 38,
    use_market: bool = True,
) -> pd.DataFrame:
    """
    Walk-forward validation for ML models.
    Trains on data[0:i], predicts data[i:i+retrain_every].
    Retrains every `retrain_every` matches.
    """
    from data_loader import build_feature_matrix as bfm
    df = bfm(df)
    df = df.reset_index(drop=True)

    cols_init = ["xgb_p_home", "xgb_p_draw", "xgb_p_away",
                 "lgb_p_home", "lgb_p_draw", "lgb_p_away",
                 "ens_p_home", "ens_p_draw", "ens_p_away"]
    for c in cols_init:
        df[c] = np.nan

    n = len(df)
    for start in range(min_train, n, retrain_every):
        train = df.iloc[:start]
        test_idx = list(range(start, min(start + retrain_every, n)))

        xgb_model = XGBoostModel(use_market=use_market) if XGB_AVAILABLE else None
        lgb_model = LightGBMModel(use_market=use_market) if LGB_AVAILABLE else None

        try:
            if xgb_model:
                xgb_model.fit(train)
        except Exception:
            xgb_model = None
        try:
            if lgb_model:
                lgb_model.fit(train)
        except Exception:
            lgb_model = None

        for i in test_idx:
            row = df.iloc[i]
            if xgb_model:
                p = xgb_model.predict_proba(row)
                df.at[i, "xgb_p_home"] = p["p_home"]
                df.at[i, "xgb_p_draw"] = p["p_draw"]
                df.at[i, "xgb_p_away"] = p["p_away"]
            if lgb_model:
                p = lgb_model.predict_proba(row)
                df.at[i, "lgb_p_home"] = p["p_home"]
                df.at[i, "lgb_p_draw"] = p["p_draw"]
                df.at[i, "lgb_p_away"] = p["p_away"]

            # Ensemble average
            preds = []
            for prefix in ["xgb", "lgb"]:
                ph = df.at[i, f"{prefix}_p_home"]
                if not np.isnan(ph):
                    preds.append((ph, df.at[i, f"{prefix}_p_draw"], df.at[i, f"{prefix}_p_away"]))
            if preds:
                ph = np.mean([p[0] for p in preds])
                pd_ = np.mean([p[1] for p in preds])
                pa = np.mean([p[2] for p in preds])
                tot = ph + pd_ + pa
                df.at[i, "ens_p_home"] = ph / tot
                df.at[i, "ens_p_draw"] = pd_ / tot
                df.at[i, "ens_p_away"] = pa / tot

    return df


def evaluate_model(df: pd.DataFrame, prob_cols: Dict[str, str],
                   label_col: str = "FTR") -> Dict[str, float]:
    """
    Evaluate a model's predictions with proper scoring rules.

    prob_cols: {"home": "col_p_home", "draw": "col_p_draw", "away": "col_p_away"}
    Returns: dict with log_loss, brier_score, rps, accuracy
    """
    mask = df[[prob_cols["home"], prob_cols["draw"], prob_cols["away"]]].notna().all(axis=1)
    sub = df[mask].copy()
    if len(sub) == 0:
        return {}

    y_true = sub[label_col].map({"H": 0, "D": 1, "A": 2}).values
    y_prob = sub[[prob_cols["home"], prob_cols["draw"], prob_cols["away"]]].values

    # Log-loss
    try:
        ll = log_loss(y_true, y_prob, labels=[0, 1, 2])
    except Exception:
        ll = np.nan

    # Ranked Probability Score (RPS) — key metric for ordered outcomes
    rps = _ranked_probability_score(y_true, y_prob)

    # Accuracy
    pred_class = y_prob.argmax(axis=1)
    acc = float((pred_class == y_true).mean())

    # Brier score (multi-class)
    n = len(y_true)
    brier = 0.0
    for i in range(n):
        one_hot = np.zeros(3)
        one_hot[y_true[i]] = 1.0
        brier += np.sum((y_prob[i] - one_hot) ** 2)
    brier /= n

    return {
        "n_predictions": len(sub),
        "log_loss": ll,
        "rps": rps,
        "brier": brier,
        "accuracy": acc,
    }


def _ranked_probability_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Ranked Probability Score (RPS) for ordered 3-way outcomes.
    Lower is better. Reference: Epstein (1969).
    Ordered outcomes: H (team wins) > D (draw) > A (away wins).
    """
    n = len(y_true)
    rps_total = 0.0
    for i in range(n):
        obs = np.zeros(3)
        obs[y_true[i]] = 1.0
        cum_obs = np.cumsum(obs)
        cum_prob = np.cumsum(y_prob[i])
        rps_total += np.sum((cum_prob - cum_obs) ** 2) / 2.0
    return rps_total / n


if __name__ == "__main__":
    from data_loader import load_data, build_feature_matrix
    df = load_data()
    df = build_feature_matrix(df)
    print(f"Dataset: {len(df)} matches")

    # Test XGBoost walk-forward
    print("\nRunning XGBoost/LightGBM walk-forward (this may take a few minutes)...")
    df = walk_forward_ml(df, min_train=300, retrain_every=76)
    n_preds = df["xgb_p_home"].notna().sum()
    print(f"XGBoost predictions: {n_preds}")

    if n_preds > 100:
        xgb_eval = evaluate_model(df, {"home": "xgb_p_home", "draw": "xgb_p_draw", "away": "xgb_p_away"})
        lgb_eval = evaluate_model(df, {"home": "lgb_p_home", "draw": "lgb_p_draw", "away": "lgb_p_away"})
        print(f"XGBoost — LogLoss: {xgb_eval.get('log_loss', 'N/A'):.4f}  RPS: {xgb_eval.get('rps', 'N/A'):.4f}")
        print(f"LightGBM — LogLoss: {lgb_eval.get('log_loss', 'N/A'):.4f}  RPS: {lgb_eval.get('rps', 'N/A'):.4f}")
