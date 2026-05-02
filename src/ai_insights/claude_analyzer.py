"""
Claude AI Analyzer
Uses the Anthropic API to generate natural language insights from detected anomalies.
"""

import os
import json
import pandas as pd
from datetime import datetime
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class ClaudeAnalyzer:
    """Generates AI-powered insights from fund anomaly data using Claude."""

    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-sonnet-4-20250514"

    def analyze_anomaly_period(
        self,
        fund_data: pd.DataFrame,
        market_data: pd.DataFrame,
        anomaly_date: str,
        fund_cnpj: str,
        window_days: int = 10,
    ) -> str:
        """
        Generate an AI-powered analysis of a specific anomaly.

        Args:
            fund_data: Fund-level data with anomaly flags
            market_data: Market indicators data
            anomaly_date: Date of the detected anomaly
            fund_cnpj: CNPJ of the fund to analyze
            window_days: Days before/after anomaly to include as context

        Returns:
            Natural language analysis from Claude
        """
        anomaly_dt = pd.to_datetime(anomaly_date)
        start = anomaly_dt - pd.Timedelta(days=window_days)
        end = anomaly_dt + pd.Timedelta(days=window_days)

        # Extract fund context
        fund_slice = fund_data[
            (fund_data["fund_cnpj"] == fund_cnpj)
            & (fund_data["date"] >= start)
            & (fund_data["date"] <= end)
        ].copy()

        # Extract market context
        market_slice = market_data[
            (market_data.index >= start) & (market_data.index <= end)
        ].copy()

        # Build context for Claude
        fund_context = self._build_fund_context(fund_slice, anomaly_date)
        market_context = self._build_market_context(market_slice)

        prompt = f"""You are a senior financial analyst specializing in Brazilian investment funds.

Analyze the following anomaly detected in a fund and provide insights:

## Fund Data (around anomaly date: {anomaly_date})
{fund_context}

## Market Context (same period)
{market_context}

Provide a concise analysis covering:
1. What happened: Describe the anomaly in clear terms
2. Probable causes: Based on the market data, what likely triggered this behavior
3. Historical pattern: Does this combination of signals typically precede further volatility
4. Risk assessment: How concerned should an investor be

Keep the analysis to 3-4 paragraphs, professional but accessible.
Use specific numbers from the data provided."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    def generate_executive_report(
        self,
        anomaly_summary: pd.DataFrame,
        market_data: pd.DataFrame,
        top_n: int = 5,
    ) -> str:
        """
        Generate an executive summary report of all detected anomalies.

        Args:
            anomaly_summary: Daily anomaly summary
            market_data: Market indicators
            top_n: Number of top anomaly days to highlight

        Returns:
            Executive report in natural language
        """
        # Find days with highest anomaly concentration
        if "n_is_anomaly" in anomaly_summary.columns:
            top_days = anomaly_summary.nlargest(top_n, "n_is_anomaly")
        else:
            top_days = anomaly_summary.head(top_n)

        summary_stats = {
            "period": f"{anomaly_summary['date'].min()} to {anomaly_summary['date'].max()}",
            "total_trading_days": len(anomaly_summary),
            "top_anomaly_days": top_days[["date", "total_funds", "n_is_anomaly"]].to_dict("records")
            if "n_is_anomaly" in top_days.columns
            else [],
        }

        prompt = f"""You are a senior consultant preparing an executive briefing on Brazilian investment fund behavior.

Based on the following anomaly detection results, generate a professional executive report:

## Analysis Summary
{json.dumps(summary_stats, indent=2, default=str)}

## Market Indicators (key statistics)
{market_data.describe().to_string() if not market_data.empty else "Not available"}

Write an executive report covering:
1. Executive Summary: Key findings in 2-3 sentences
2. Critical Periods: What happened on the days with highest anomaly concentration
3. Market Correlation: Which external factors showed strongest relationship with anomalies
4. Forward-Looking Signals: What combination of market indicators should be monitored
5. Recommendations: Actionable next steps for portfolio management

Format as a professional consulting report. Be specific with data points.
Keep it under 500 words."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    def generate_predictive_alert(
        self, current_market: dict, historical_patterns: dict
    ) -> str:
        """
        Generate a forward-looking alert based on current market conditions.

        Args:
            current_market: Current market indicators
            historical_patterns: Historical pattern matches

        Returns:
            Alert message with risk assessment
        """
        prompt = f"""You are a risk monitoring system for Brazilian investment funds.

Current market conditions:
{json.dumps(current_market, indent=2, default=str)}

Historical pattern matches (similar conditions in the past):
{json.dumps(historical_patterns, indent=2, default=str)}

Based on these conditions, generate a concise alert that:
1. States the current risk level (Low / Moderate / Elevated / High)
2. Identifies which signals are triggering concern
3. References what happened historically when similar conditions occurred
4. Suggests specific monitoring actions

Keep it to 2-3 paragraphs. Be direct and actionable."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    def _build_fund_context(self, fund_slice: pd.DataFrame, anomaly_date: str) -> str:
        """Build a text summary of fund data around the anomaly."""
        if fund_slice.empty:
            return "No fund data available for this period."

        cols_to_show = ["date", "daily_return", "z_score", "net_aum", "net_flow"]
        available = [c for c in cols_to_show if c in fund_slice.columns]

        return fund_slice[available].to_string(index=False)

    def _build_market_context(self, market_slice: pd.DataFrame) -> str:
        """Build a text summary of market conditions."""
        if market_slice.empty:
            return "No market data available for this period."

        # Select key columns
        key_cols = [c for c in market_slice.columns if "close" in c or "return" in c]
        if key_cols:
            return market_slice[key_cols].to_string()
        return market_slice.to_string()

    def save_report(self, report: str, filename: str = None):
        """Save generated report to file."""
        os.makedirs("reports/generated", exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.md"

        filepath = os.path.join("reports/generated", filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"Report saved to: {filepath}")


if __name__ == "__main__":
    try:
        # Load data
        anomaly_summary = pd.read_parquet("data/processed/anomaly_summary.parquet")
        market_data = pd.read_parquet("data/processed/market_data.parquet")

        analyzer = ClaudeAnalyzer()

        print("=" * 60)
        print("Generating Executive Report with AI Insights")
        print("=" * 60)

        report = analyzer.generate_executive_report(anomaly_summary, market_data)
        print("\n" + report)

        analyzer.save_report(report, "executive_report.md")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run the data collection and anomaly detection steps first.")
