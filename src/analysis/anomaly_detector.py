"""
Anomaly Detection Engine
Identifies unusual behavior in fund returns using statistical methods.
"""

import pandas as pd
import numpy as np
from typing import Tuple


class AnomalyDetector:
    """Detects anomalous behavior in investment fund returns."""

    def __init__(self, window: int = 63, z_threshold: float = 2.5):
        """
        Args:
            window: Rolling window size in trading days (63 ≈ 3 months)
            z_threshold: Z-score threshold to flag anomalies
        """
        self.window = window
        self.z_threshold = z_threshold

    def calculate_rolling_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling Z-score for each fund's daily returns.

        Args:
            df: DataFrame with columns ['fund_cnpj', 'date', 'daily_return']

        Returns:
            DataFrame with additional z_score and is_anomaly columns
        """
        print("Calculating rolling Z-scores...")

        df = df.sort_values(["fund_cnpj", "date"]).copy()

        # Rolling mean and std per fund
        grouped = df.groupby("fund_cnpj")["daily_return"]
        df["rolling_mean"] = grouped.transform(
            lambda x: x.rolling(self.window, min_periods=21).mean()
        )
        df["rolling_std"] = grouped.transform(
            lambda x: x.rolling(self.window, min_periods=21).std()
        )

        # Z-score
        df["z_score"] = (df["daily_return"] - df["rolling_mean"]) / df["rolling_std"]

        # Flag anomalies
        df["is_anomaly"] = df["z_score"].abs() > self.z_threshold

        anomaly_count = df["is_anomaly"].sum()
        total = len(df.dropna(subset=["z_score"]))
        print(f"  → {anomaly_count:,} anomalies detected ({anomaly_count/total*100:.2f}%)")

        return df

    def calculate_rolling_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling volatility and detect volatility regime changes.

        Args:
            df: DataFrame with columns ['fund_cnpj', 'date', 'daily_return']

        Returns:
            DataFrame with volatility metrics
        """
        print("Calculating rolling volatility...")

        df = df.sort_values(["fund_cnpj", "date"]).copy()

        grouped = df.groupby("fund_cnpj")["daily_return"]

        # Short-term vs long-term volatility
        df["vol_short"] = grouped.transform(
            lambda x: x.rolling(21, min_periods=10).std() * np.sqrt(252)
        )
        df["vol_long"] = grouped.transform(
            lambda x: x.rolling(63, min_periods=21).std() * np.sqrt(252)
        )

        # Volatility ratio — values > 1.5 suggest regime change
        df["vol_ratio"] = df["vol_short"] / df["vol_long"]
        df["vol_regime_change"] = df["vol_ratio"] > 1.5

        regime_changes = df["vol_regime_change"].sum()
        print(f"  → {regime_changes:,} volatility regime changes detected")

        return df

    def detect_flow_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect unusual inflow/outflow patterns.

        Args:
            df: DataFrame with columns ['fund_cnpj', 'date', 'net_flow', 'net_aum']

        Returns:
            DataFrame with flow anomaly flags
        """
        print("Detecting flow anomalies...")

        df = df.sort_values(["fund_cnpj", "date"]).copy()

        # Net flow as percentage of AUM
        df["flow_pct"] = df["net_flow"] / df["net_aum"]

        # Rolling stats for flow
        grouped = df.groupby("fund_cnpj")["flow_pct"]
        df["flow_rolling_mean"] = grouped.transform(
            lambda x: x.rolling(self.window, min_periods=21).mean()
        )
        df["flow_rolling_std"] = grouped.transform(
            lambda x: x.rolling(self.window, min_periods=21).std()
        )

        # Flow Z-score
        df["flow_z_score"] = (
            (df["flow_pct"] - df["flow_rolling_mean"]) / df["flow_rolling_std"]
        )

        # Large outflows are particularly important
        df["is_large_outflow"] = df["flow_z_score"] < -self.z_threshold

        outflow_count = df["is_large_outflow"].sum()
        print(f"  → {outflow_count:,} large outflow events detected")

        return df

    def get_anomaly_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a summary of all detected anomalies grouped by date.

        Args:
            df: DataFrame with anomaly flags

        Returns:
            Daily summary of anomalies across all funds
        """
        anomaly_cols = ["is_anomaly", "vol_regime_change", "is_large_outflow"]
        available_cols = [col for col in anomaly_cols if col in df.columns]

        summary = df.groupby("date").agg(
            total_funds=("fund_cnpj", "nunique"),
            **{
                f"n_{col}": (col, "sum")
                for col in available_cols
            },
        ).reset_index()

        # Calculate anomaly density (% of funds with anomalies per day)
        for col in available_cols:
            summary[f"pct_{col}"] = summary[f"n_{col}"] / summary["total_funds"] * 100

        return summary

    def run_full_detection(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete anomaly detection pipeline.

        Args:
            df: Processed fund data

        Returns:
            Tuple of (detailed results, daily summary)
        """
        print("=" * 60)
        print("Running Anomaly Detection Pipeline")
        print("=" * 60)

        df = self.calculate_rolling_zscore(df)
        df = self.calculate_rolling_volatility(df)

        if "net_flow" in df.columns and "net_aum" in df.columns:
            df = self.detect_flow_anomalies(df)

        summary = self.get_anomaly_summary(df)

        print(f"\nPipeline complete.")
        print(f"  → Period: {df['date'].min()} to {df['date'].max()}")
        print(f"  → Funds analyzed: {df['fund_cnpj'].nunique():,}")

        return df, summary


if __name__ == "__main__":
    # Load processed fund data
    try:
        df = pd.read_parquet("data/processed/processed_funds.parquet")
        detector = AnomalyDetector()
        results, summary = detector.run_full_detection(df)

        # Save results
        results.to_parquet("data/processed/anomaly_results.parquet", index=False)
        summary.to_parquet("data/processed/anomaly_summary.parquet", index=False)
        print("\nResults saved to data/processed/")

    except FileNotFoundError:
        print("Error: Run cvm_collector.py first to generate processed fund data.")
