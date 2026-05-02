"""
Market Data Collector
Fetches external market data (Ibovespa, USD/BRL, VIX, CDI) for correlation analysis.
Sources: Yahoo Finance, Central Bank of Brazil (BCB)
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


class MarketCollector:
    """Collects external market data for correlation with fund anomalies."""

    # Yahoo Finance tickers for Brazilian market indicators
    TICKERS = {
        "ibovespa": "^BVSP",
        "usd_brl": "BRL=X",
        "vix": "^VIX",
        "sp500": "^GSPC",
        "us_10y": "^TNX",
    }

    # BCB series codes for Brazilian economic indicators
    BCB_SERIES = {
        "cdi": 12,
        "selic": 432,
        "ipca": 433,
    }

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def fetch_yahoo_data(
        self, start_date: str, end_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch market data from Yahoo Finance.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (defaults to today)

        Returns:
            DataFrame with daily market indicators
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print("Fetching market data from Yahoo Finance...")
        all_data = {}

        for name, ticker in self.TICKERS.items():
            try:
                print(f"  → {name} ({ticker})...")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)

                if not data.empty:
                    all_data[f"{name}_close"] = data["Close"].squeeze()
                    all_data[f"{name}_return"] = data["Close"].squeeze().pct_change()
                    all_data[f"{name}_volatility"] = (
                        data["Close"].squeeze().pct_change().rolling(21).std()
                    )

            except Exception as e:
                print(f"  → Error fetching {name}: {e}")

        if all_data:
            df = pd.DataFrame(all_data)
            df.index.name = "date"
            print(f"\nMarket data: {len(df):,} trading days loaded")
            return df

        return pd.DataFrame()

    def fetch_bcb_data(self, start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        Fetch economic indicators from Central Bank of Brazil.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            DataFrame with BCB indicators
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print("\nFetching data from Central Bank of Brazil...")

        # Format dates for BCB API
        start_fmt = datetime.strptime(start_date, "%Y-%m-%d").strftime("%d/%m/%Y")
        end_fmt = datetime.strptime(end_date, "%Y-%m-%d").strftime("%d/%m/%Y")

        all_data = {}

        for name, series_code in self.BCB_SERIES.items():
            try:
                print(f"  → {name} (series {series_code})...")
                url = (
                    f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}"
                    f"/dados?formato=json&dataInicial={start_fmt}&dataFinal={end_fmt}"
                )
                df_temp = pd.read_json(url)

                if not df_temp.empty:
                    df_temp["data"] = pd.to_datetime(df_temp["data"], format="%d/%m/%Y")
                    df_temp = df_temp.set_index("data")
                    all_data[name] = df_temp["valor"].astype(float)

            except Exception as e:
                print(f"  → Error fetching {name}: {e}")

        if all_data:
            df = pd.DataFrame(all_data)
            df.index.name = "date"
            print(f"\nBCB data: {len(df):,} records loaded")
            return df

        return pd.DataFrame()

    def merge_market_data(
        self, yahoo_df: pd.DataFrame, bcb_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge Yahoo Finance and BCB data into a single dataset."""
        if yahoo_df.empty and bcb_df.empty:
            return pd.DataFrame()

        if yahoo_df.empty:
            return bcb_df
        if bcb_df.empty:
            return yahoo_df

        merged = yahoo_df.join(bcb_df, how="outer")
        merged = merged.sort_index()
        merged = merged.ffill()  # Forward fill for non-trading days

        print(f"\nMerged market data: {len(merged):,} records | {len(merged.columns)} indicators")
        return merged

    def save_market_data(self, df: pd.DataFrame, filename: str = "market_data.parquet"):
        """Save market data to parquet format."""
        output_dir = self.data_dir.replace("raw", "processed")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        df.to_parquet(filepath)
        print(f"Saved to: {filepath}")


if __name__ == "__main__":
    collector = MarketCollector()

    # Fetch last 12 months
    start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    print("=" * 60)
    print("Market Data Collector")
    print("=" * 60)

    yahoo_data = collector.fetch_yahoo_data(start_date=start)
    bcb_data = collector.fetch_bcb_data(start_date=start)

    merged = collector.merge_market_data(yahoo_data, bcb_data)

    if not merged.empty:
        collector.save_market_data(merged)
        print("\nDone! Market data ready for correlation analysis.")
