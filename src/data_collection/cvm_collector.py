"""
CVM Fund Data Collector
Fetches daily fund data (returns, AUM, inflows/outflows) from CVM open data portal.
Source: https://dados.cvm.gov.br/
"""

import os
import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta


class CVMCollector:
    """Collects and processes Brazilian investment fund data from CVM."""

    BASE_URL = "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS"
    HIST_URL = "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/HIST"

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def _build_url(self, year: int, month: int) -> str:
        """Build the download URL for a given year/month."""
        current_year = datetime.now().year
        if year < current_year:
            return f"{self.HIST_URL}/inf_diario_fi_{year}.zip"
        else:
            return f"{self.BASE_URL}/inf_diario_fi_{year}{month:02d}.zip"

    def fetch_monthly_data(self, year: int, month: int) -> pd.DataFrame:
        """Fetch fund data for a specific month."""
        url = self._build_url(year, month)
        print(f"Fetching data from: {url}")

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            filepath = os.path.join(self.data_dir, f"inf_diario_fi_{year}{month:02d}.zip")
            with open(filepath, "wb") as f:
                f.write(response.content)

            df = pd.read_csv(
                filepath,
                sep=";",
                encoding="latin-1",
                compression="zip",
                parse_dates=["DT_COMPTC"],
            )
            print(f"  → {len(df):,} records loaded")
            return df

        except requests.exceptions.RequestException as e:
            print(f"  → Error fetching data: {e}")
            return pd.DataFrame()

    def fetch_date_range(
        self, start_year: int, start_month: int, end_year: int, end_month: int
    ) -> pd.DataFrame:
        """Fetch fund data for a range of months."""
        all_data = []

        current = datetime(start_year, start_month, 1)
        end = datetime(end_year, end_month, 1)

        while current <= end:
            df = self.fetch_monthly_data(current.year, current.month)
            if not df.empty:
                all_data.append(df)
            current += timedelta(days=32)
            current = current.replace(day=1)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            print(f"\nTotal records: {len(combined):,}")
            return combined
        return pd.DataFrame()

    def process_fund_data(self, df: pd.DataFrame, min_aum: float = 1_000_000) -> pd.DataFrame:
        """
        Clean and process raw CVM data.

        Args:
            df: Raw dataframe from CVM
            min_aum: Minimum AUM (patrimônio líquido) to filter relevant funds

        Returns:
            Processed dataframe with calculated fields
        """
        if df.empty:
            return df

        # Rename columns for clarity
        column_map = {
            "CNPJ_FUNDO_CLASSE": "fund_cnpj",
            "TP_FUNDO_CLASSE":   "fund_type",
            "DT_COMPTC": "date",
            "VL_TOTAL": "aum",
            "VL_QUOTA": "nav",
            "VL_PATRIM_LIQ": "net_aum",
            "CAPTC_DIA": "inflows",
            "RESG_DIA": "outflows",
            "NR_COTST": "shareholders",
        }

        df = df.rename(columns=column_map)

        # Keep only relevant columns
        cols_to_keep = [col for col in column_map.values() if col in df.columns]
        df = df[cols_to_keep].copy()

        # Filter by minimum AUM
        df = df[df["net_aum"] >= min_aum].copy()

        # Sort by fund and date
        df = df.sort_values(["fund_cnpj", "date"]).reset_index(drop=True)

        # Calculate daily returns
        df["daily_return"] = df.groupby("fund_cnpj")["nav"].pct_change()

        # Calculate net flow
        df["net_flow"] = df["inflows"] - df["outflows"]

        print(f"Processed data: {len(df):,} records | {df['fund_cnpj'].nunique():,} funds")
        return df

    def save_processed(self, df: pd.DataFrame, filename: str = "processed_funds.parquet"):
        """Save processed data to parquet format."""
        output_dir = self.data_dir.replace("raw", "processed")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        df.to_parquet(filepath, index=False)
        print(f"Saved to: {filepath}")


if __name__ == "__main__":
    collector = CVMCollector()

    # Fetch last 12 months of data
    today = datetime.now()
    start = today - timedelta(days=365)

    print("=" * 60)
    print("CVM Fund Data Collector")
    print("=" * 60)

    df_raw = collector.fetch_date_range(
        start_year=start.year,
        start_month=start.month,
        end_year=today.year,
        end_month=today.month,
    )

    if not df_raw.empty:
        df_processed = collector.process_fund_data(df_raw)
        collector.save_processed(df_processed)
        print("\nDone! Data ready for analysis.")
