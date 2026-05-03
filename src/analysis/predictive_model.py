"""
Predictive Model
Predicts fund volatility based on market signal patterns.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import json
import os


class VolatilityPredictor:
    """Predicts future fund volatility using market signals."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )
        self.scaler = StandardScaler()
        self.feature_importance = None

    def prepare_features(
        self, signal_matrix: pd.DataFrame, target_col: str = "is_anomaly"
    ) -> tuple:
        """
        Prepare feature matrix and target variable.

        Args:
            signal_matrix: Output from MarketCorrelator.build_signal_matrix()
            target_col: Column to predict

        Returns:
            Tuple of (X, y, feature_names)
        """
        print("Preparing features...")

        # Select feature columns (exclude fund identifiers and target-related columns)
        exclude_cols = [
            "fund_cnpj", "date", "fund_type", "nav", "aum", "net_aum", "inflows", "outflows",
            "shareholders", "daily_return", "net_flow",
            "is_anomaly", "is_large_outflow", "vol_regime_change",
            "z_score", "flow_z_score",
            "rolling_mean", "rolling_std", "flow_rolling_mean", "flow_rolling_std",
        ]

        feature_cols = [c for c in signal_matrix.columns if c not in exclude_cols]
        feature_cols = [c for c in feature_cols if signal_matrix[c].dtype in ["float64", "int64"]]

        X = signal_matrix[feature_cols].copy()

        # One-hot encode fund_type if present
        if "fund_type" in signal_matrix.columns:
            dummies = pd.get_dummies(signal_matrix["fund_type"], prefix="type", dtype=float)
            dummies = dummies.loc[X.index]
            X = pd.concat([X, dummies], axis=1)
            print(f"  → Fund type dummies added: {list(dummies.columns)}")
        y = signal_matrix[target_col].astype(int).copy()

        # Remove any remaining NaN
        mask = X.notna().all(axis=1)
        X = X[mask]
        y = y[mask]

        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y.loc[X.index]

        print(f"  → Features: {len(feature_cols)}")
        print(f"  → Samples: {len(X):,}")
        print(f"  → Anomaly rate: {y.mean()*100:.2f}%")

        return X, y, feature_cols

    def train_and_evaluate(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
    ) -> dict:
        """
        Train the model using time-series cross-validation.

        Args:
            X: Feature matrix
            y: Target variable
            n_splits: Number of CV splits

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nTraining with {n_splits}-fold time series CV...")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics = {"precision": [], "recall": [], "f1": []}

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)

            # Metrics
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["f1"].append(f1)

            print(f"  Fold {fold+1}: Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")

        # Final training on all data
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        # Feature importance
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=X.columns,
        ).sort_values(ascending=False)

        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        print(f"\n  Average: Precision={avg_metrics['precision']:.3f} "
              f"Recall={avg_metrics['recall']:.3f} F1={avg_metrics['f1']:.3f}")

        return avg_metrics

    def get_top_features(self, n: int = 15) -> pd.Series:
        """Return top N most important features."""
        if self.feature_importance is None:
            raise ValueError("Model not trained yet.")
        return self.feature_importance.head(n)

    def predict_current_risk(self, current_signals: pd.DataFrame) -> dict:
        """
        Predict risk level based on current market signals.

        Args:
            current_signals: Current market data in same format as training features

        Returns:
            Dictionary with prediction and probability
        """
        X_scaled = self.scaler.transform(current_signals)
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]

        risk_level = "Low"
        if probability[1] > 0.3:
            risk_level = "Moderate"
        if probability[1] > 0.5:
            risk_level = "Elevated"
        if probability[1] > 0.7:
            risk_level = "High"

        return {
            "prediction": int(prediction),
            "anomaly_probability": float(probability[1]),
            "risk_level": risk_level,
            "top_signals": self.get_top_features(5).to_dict(),
        }

    def save_model_results(self, metrics: dict, output_dir: str = "reports/generated"):
        """Save model performance results."""
        os.makedirs(output_dir, exist_ok=True)

        results = {
            "metrics": metrics,
            "top_features": self.get_top_features(15).to_dict() if self.feature_importance is not None else {},
        }

        filepath = os.path.join(output_dir, "model_results.json")
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Model results saved to: {filepath}")


if __name__ == "__main__":
    try:
        signal_matrix = pd.read_parquet("data/processed/signal_matrix.parquet")

        predictor = VolatilityPredictor()

        print("=" * 60)
        print("Volatility Prediction Model")
        print("=" * 60)

        X, y, features = predictor.prepare_features(signal_matrix)
        metrics = predictor.train_and_evaluate(X, y)

        print("\nTop 10 Most Important Features:")
        print(predictor.get_top_features(10).to_string())

        predictor.save_model_results(metrics)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run the correlation analysis step first.")
