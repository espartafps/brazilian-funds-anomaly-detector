"""
Market Correlator
Correlates detected fund anomalies with external market events.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class MarketCorrelator:
    """Analyzes correlation between fund anomalies and market events."""

    def __init__(self):
        self.correlation_results = {}

    def merge_fund_and_market(
        self, fund_data: pd.DataFrame, market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge fund anomaly data with market indicators.

        Args:
            fund_data: Fund data with anomaly flags (from AnomalyDetector)
            market_data: Market indicators (from MarketCollector)

        Returns:
            Merged DataFrame
        """
        print("Merging fund and market data...")

        fund_data = fund_data.copy()
        fund_data["date"] = pd.to_datetime(fund_data["date"])

        market_data = market_data.copy()
        market_data.index = pd.to_datetime(market_data.index)
        market_data = market_data.reset_index().rename(columns={"index": "date"})

        merged = fund_data.merge(market_data, on="date", how="left")
        print(f"  → {len(merged):,} records after merge")

        return merged

    def calculate_correlations(
        self, merged_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate correlation between fund returns and market indicators.

        Returns:
            Correlation matrix
        """
        print("Calculating correlations...")

        return_cols = [c for c in merged_df.columns if "return" in c and c != "daily_return"]
        close_cols = [c for c in merged_df.columns if "close" in c]

        analysis_cols = ["daily_return", "z_score"] + return_cols
        available = [c for c in analysis_cols if c in merged_df.columns]

        corr_matrix = merged_df[available].corr()
        self.correlation_results["full_period"] = corr_matrix

        print(f"  → Correlation matrix: {corr_matrix.shape}")
        return corr_matrix

    def analyze_anomaly_market_conditions(
        self, merged_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compare market conditions during anomaly days vs normal days.

        Returns:
            DataFrame comparing market stats for anomaly vs normal periods
        """
        print("Analyzing market conditions during anomalies...")

        if "is_anomaly" not in merged_df.columns:
            print("  → No anomaly flags found. Run AnomalyDetector first.")
            return pd.DataFrame()

        market_cols = [c for c in merged_df.columns if any(
            x in c for x in ["close", "return", "cdi", "selic", "vol"]
        ) and c not in ["daily_return"]]

        available = [c for c in market_cols if c in merged_df.columns]

        if not available:
            print("  → No market columns found.")
            return pd.DataFrame()

        comparison = merged_df.groupby("is_anomaly")[available].agg(["mean", "std"])
        print(f"  → Comparison generated for {len(available)} market indicators")

        return comparison

    def find_leading_indicators(
        self, merged_df: pd.DataFrame, lead_days: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, pd.DataFrame]:
        """
        Check if market indicators lead fund anomalies.
        Tests whether market movements N days before predict anomalies.

        Args:
            merged_df: Merged fund + market data
            lead_days: List of lead periods to test

        Returns:
            Dict with correlation results for each lead period
        """
        print("Searching for leading indicators...")

        return_cols = [c for c in merged_df.columns if "return" in c and c != "daily_return"]
        results = {}

        for days in lead_days:
            lead_corrs = {}
            for col in return_cols:
                if col in merged_df.columns:
                    # Shift market data forward (today's market → future anomaly)
                    merged_df[f"{col}_lag{days}"] = merged_df.groupby("fund_cnpj")[col].shift(days)

                    if "is_anomaly" in merged_df.columns:
                        corr = merged_df[f"{col}_lag{days}"].corr(
                            merged_df["is_anomaly"].astype(float)
                        )
                        lead_corrs[col] = corr

            results[f"lead_{days}d"] = pd.Series(lead_corrs)
            print(f"  → Lead {days}d: strongest signal = {max(lead_corrs.values(), default=0):.4f}")

        return results

    def build_signal_matrix(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a matrix of market signals for the predictive model.

        Returns:
            Feature matrix ready for modeling
        """
        print("Building signal matrix for predictive model...")

        features = merged_df.copy()

        # Add lagged features
        market_cols = [c for c in features.columns if "return" in c and c != "daily_return"]

        for col in market_cols:
            for lag in [1, 3, 5, 10]:
                features[f"{col}_lag{lag}"] = features.groupby("fund_cnpj")[col].shift(lag)

        # Add rolling features
        for col in market_cols:
            features[f"{col}_roll5_mean"] = features.groupby("fund_cnpj")[col].transform(
                lambda x: x.rolling(5).mean()
            )

        # Drop NaN rows
        initial_len = len(features)
        features = features.dropna()
        print(f"  → {len(features):,} records ({initial_len - len(features):,} dropped due to NaN)")

        return features


if __name__ == "__main__":
    try:
        # Load data
        fund_data = pd.read_parquet("data/processed/anomaly_results.parquet")
        market_data = pd.read_parquet("data/processed/market_data.parquet")

        correlator = MarketCorrelator()

        print("=" * 60)
        print("Market Correlation Analysis")
        print("=" * 60)

        merged = correlator.merge_fund_and_market(fund_data, market_data)
        corr_matrix = correlator.calculate_correlations(merged)
        comparison = correlator.analyze_anomaly_market_conditions(merged)
        leading = correlator.find_leading_indicators(merged)
        signal_matrix = correlator.build_signal_matrix(merged)

        # Save
        signal_matrix.to_parquet("data/processed/signal_matrix.parquet", index=False)
        print("\nSignal matrix saved. Ready for predictive modeling.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run previous pipeline steps first.")
