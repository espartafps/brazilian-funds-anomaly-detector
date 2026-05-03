"""
DeepSeek AI Analyzer
Uses the DeepSeek API to generate natural language insights from detected anomalies.
"""

import os
import json
import time
import pandas as pd
from datetime import datetime
import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class DeepSeekAnalyzer:
    """Generates AI-powered insights from fund anomaly data using DeepSeek."""

    def __init__(self):
        self.model = "deepseek-chat"
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

    def _generate(self, prompt: str, max_tokens: int) -> str:
        """Call DeepSeek with one automatic retry on 429 rate-limit errors."""
        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except openai.RateLimitError:
                if attempt == 0:
                    print("Rate limit hit — waiting 30 s before retry...")
                    time.sleep(30)
                else:
                    raise

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
            Natural language analysis from DeepSeek
        """
        anomaly_dt = pd.to_datetime(anomaly_date)
        start = anomaly_dt - pd.Timedelta(days=window_days)
        end = anomaly_dt + pd.Timedelta(days=window_days)

        fund_slice = fund_data[
            (fund_data["fund_cnpj"] == fund_cnpj)
            & (fund_data["date"] >= start)
            & (fund_data["date"] <= end)
        ].copy()

        market_slice = market_data[
            (market_data.index >= start) & (market_data.index <= end)
        ].copy()

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

        return self._generate(prompt, max_tokens=1000)

    def generate_executive_report(
        self,
        anomaly_summary: pd.DataFrame,
        market_data: pd.DataFrame,
        top_n: int = 5,
        anomaly_by_type: pd.DataFrame = None,
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
        recent_summary = anomaly_summary.tail(30)

        if "n_is_anomaly" in recent_summary.columns:
            top_days = recent_summary.nlargest(top_n, "n_is_anomaly")
        else:
            top_days = recent_summary.head(top_n)

        summary_stats = {
            "period": f"{anomaly_summary['date'].min()} to {anomaly_summary['date'].max()}",
            "total_trading_days": len(anomaly_summary),
            "top_anomaly_days": top_days[["date", "total_funds", "n_is_anomaly"]].to_dict("records")
            if "n_is_anomaly" in top_days.columns
            else [],
        }

        close_return_cols = [c for c in market_data.columns if "close" in c or "return" in c]
        market_stats = (
            market_data[close_return_cols].describe().to_string()
            if not market_data.empty and close_return_cols
            else "Not available"
        )

        type_section = ""
        if anomaly_by_type is not None and not anomaly_by_type.empty:
            type_section = f"""
## Fund Type Breakdown
{self._build_fund_type_context(anomaly_by_type)}
"""

        prompt = f"""You are a senior consultant preparing an executive briefing on Brazilian investment fund behavior.

Based on the following anomaly detection results, generate a professional executive report:

## Analysis Summary
{json.dumps(summary_stats, indent=2, default=str)}

## Market Indicators (key statistics)
{market_stats}
{type_section}
Write an executive report covering:
1. Executive Summary: Key findings in 2-3 sentences
2. Critical Periods: What happened on the days with highest anomaly concentration
3. Market Correlation: Which external factors showed strongest relationship with anomalies
4. Fund Type Exposure: Which categories (FIA, FIM, FIDC, etc.) showed the highest anomaly rates and why{' — use the Fund Type Breakdown data above' if type_section else ' — note: fund type data not available yet'}
5. Forward-Looking Signals: What combination of market indicators should be monitored
6. Recommendations: Actionable next steps for portfolio management

Format as a professional consulting report. Be specific with data points.
Keep it under 600 words."""

        return self._generate(prompt, max_tokens=1800)

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

        return self._generate(prompt, max_tokens=800)

    def generate_geopolitical_report(
        self,
        anomaly_summary: pd.DataFrame,
        market_data: pd.DataFrame,
        anomaly_by_type: pd.DataFrame = None,
    ) -> str:
        """
        Generate a geopolitical analysis crossing detected anomalies with the
        2026 Iran-US conflict and its impact on Brazilian investment funds.

        Args:
            anomaly_summary: Daily anomaly summary
            market_data: Market indicators

        Returns:
            Geopolitical impact report in natural language
        """
        recent_summary = anomaly_summary.tail(30)

        if "n_is_anomaly" in recent_summary.columns:
            top_days = recent_summary.nlargest(10, "n_is_anomaly")
        else:
            top_days = recent_summary.head(10)

        anomaly_dates = (
            top_days[["date", "n_is_anomaly", "total_funds"]].to_dict("records")
            if "n_is_anomaly" in top_days.columns
            else top_days[["date"]].to_dict("records")
        )

        close_return_cols = [c for c in market_data.columns if "close" in c or "return" in c]
        market_stats = (
            market_data[close_return_cols].describe().to_string()
            if not market_data.empty and close_return_cols
            else "Not available"
        )

        type_section = ""
        if anomaly_by_type is not None and not anomaly_by_type.empty:
            type_section = f"""
## Fund Type Breakdown During Peak Anomaly Days
{self._build_fund_type_context(anomaly_by_type, top_n=10)}
"""

        prompt = f"""You are a senior geopolitical risk analyst specializing in the impact of international conflicts on emerging market investment funds.

## Detected Anomaly Peaks (top 10 days by number of anomalous funds)
{json.dumps(anomaly_dates, indent=2, default=str)}

## Market Indicators Summary
{market_stats}
{type_section}
## Geopolitical Context — Iran-US Conflict (2026)
- Late February 2026: Iran closed the Strait of Hormuz following US and Israeli strikes
- March 2: IRGC officially confirmed the closure of the strait to hostile nations
- March 17: US bombed Iranian missile silos on the Iranian coast
- March 18: UAE joined the US military effort
- March 19–20: US deployed A-10 jets and Apache helicopters; 16 Iranian commercial vessels destroyed
- Late March / early April: Brent crude surpassed US$126/barrel, the highest price in 4 years
- April 8: Ceasefire announced between the US and Iran
- Brazil's Central Bank cited the conflict in its Copom statement as a factor of caution for the Selic rate

## Your Task
Cross-reference the anomaly peak dates with the geopolitical timeline above and write a professional report covering:

1. **Timeline Correlation**: Map each major anomaly peak to the closest geopolitical event and explain why that event would affect Brazilian funds within 1–3 trading days
2. **Causality Chain**: Explain the full transmission mechanism — Strait of Hormuz closure → oil price spike → global VIX surge → USD/BRL appreciation → Ibovespa selloff → Brazilian fund anomalies
3. **Sector Impact**: Which types of Brazilian funds were most affected at each stage of the conflict{' — use the Fund Type Breakdown data above with real numbers' if type_section else ' (equity, fixed income, FX-hedged, commodities)'}, and why
4. **Ceasefire Effect**: Analyze whether the April 8 ceasefire announcement is visible in the anomaly data as a relief event
5. **BCB / Copom Dimension**: Explain how the Central Bank's mention of the conflict as a Selic risk factor adds a second-order effect on fixed-income funds
6. **Conclusion**: Summarize what this episode reveals about Brazilian funds' sensitivity to Middle East geopolitical shocks

Write in a professional consulting tone. Be specific with dates and numbers from both the anomaly data and the geopolitical timeline. Aim for 600–800 words."""

        return self._generate(prompt, max_tokens=2000)

    def generate_fund_type_report(
        self,
        anomaly_by_type: pd.DataFrame,
        market_data: pd.DataFrame,
        correlations_by_type: pd.DataFrame = None,
    ) -> str:
        """
        Generate a dedicated report on anomaly behavior segmented by fund type.

        Args:
            anomaly_by_type: Daily anomaly summary broken down by fund_type
            market_data: Market indicators
            correlations_by_type: Per-type correlation table (from correlations_by_type.csv)

        Returns:
            Fund type analysis report in natural language
        """
        type_context = self._build_fund_type_context(anomaly_by_type, top_n=10)

        # Aggregate stats per fund type for the full period
        agg = (
            anomaly_by_type.groupby("fund_type")
            .agg(
                avg_anomaly_pct=("pct_is_anomaly", "mean"),
                max_anomaly_pct=("pct_is_anomaly", "max"),
                total_funds=("total_funds", "mean"),
            )
            .round(2)
            .sort_values("avg_anomaly_pct", ascending=False)
            .reset_index()
            .to_dict("records")
        )

        corr_section = ""
        if correlations_by_type is not None and not correlations_by_type.empty:
            corr_section = f"""
## Market Correlations by Fund Type (correlation with daily_return)
{correlations_by_type.to_string(index=False)}
"""

        close_return_cols = [c for c in market_data.columns if "close" in c or "return" in c]
        market_stats = (
            market_data[close_return_cols].describe().to_string()
            if not market_data.empty and close_return_cols
            else "Not available"
        )

        prompt = f"""You are a senior portfolio analyst specializing in Brazilian investment fund classification and risk.

## Period Covered
{anomaly_by_type['date'].min()} to {anomaly_by_type['date'].max()}

## Anomaly Rate by Fund Type (full period average and peak)
{json.dumps(agg, indent=2, default=str)}

## Top Anomaly Days by Fund Type
{type_context}

## Market Indicators Summary
{market_stats}
{corr_section}
Write a professional analytical report covering:

1. **Ranking de Exposição**: Which fund types showed the highest and lowest anomaly rates? Quantify with the data above.
2. **Perfil de Risco por Categoria**:
   - **FIA (Ações)**: How equity funds behaved vs Ibovespa movements
   - **FIM (Multimercado)**: How multi-strategy funds reacted to VIX and USD/BRL
   - **FIDC**: How credit funds showed lagged or distinct reactions compared to market funds
   - **Other types present in the data**: Describe their behavior patterns
3. **Sincronização vs Defasagem**: Did all fund types peak on the same days, or did some lag others? What does that reveal about transmission mechanisms?
4. **Correlação com Mercado**: Based on the correlation data, which market variable (Ibovespa, USD/BRL, VIX, CDI) best explains each fund type's anomaly rate?
5. **Implicações para Diversificação**: What do these results mean for a multi-fund portfolio? Did holding multiple fund types reduce or amplify systemic risk during peak stress?
6. **Conclusão e Recomendações**: Concrete takeaways for fund selection and risk monitoring by category.

Write in Portuguese, professional consulting tone. Use specific numbers. Aim for 700–900 words."""

        return self._generate(prompt, max_tokens=2500)

    def _build_fund_type_context(
        self, anomaly_by_type: pd.DataFrame, top_n: int = 5
    ) -> str:
        """Build a compact text summary of the by-type anomaly data for prompt injection."""
        if anomaly_by_type.empty or "fund_type" not in anomaly_by_type.columns:
            return "Fund type data not available."

        if "pct_is_anomaly" not in anomaly_by_type.columns:
            return "pct_is_anomaly column missing from by-type summary."

        lines = []
        for ft, group in anomaly_by_type.groupby("fund_type"):
            top_days = group.nlargest(top_n, "pct_is_anomaly")[
                ["date", "pct_is_anomaly", "total_funds"]
            ].copy()
            top_days["date"] = top_days["date"].astype(str)
            avg = group["pct_is_anomaly"].mean()
            peak = group["pct_is_anomaly"].max()
            lines.append(
                f"\n### {ft}  |  avg anomaly: {avg:.1f}%  |  peak: {peak:.1f}%"
            )
            lines.append(top_days.to_string(index=False))

        return "\n".join(lines)

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
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    try:
        anomaly_summary = pd.read_parquet("data/processed/anomaly_summary.parquet")
        market_data     = pd.read_parquet("data/processed/market_data.parquet")

        # Optional — only available after re-running pipeline with fund_type
        try:
            anomaly_by_type = pd.read_parquet("data/processed/anomaly_summary_by_type.parquet")
            print("✓ Fund type data loaded")
        except FileNotFoundError:
            anomaly_by_type = None
            print("⚠ anomaly_summary_by_type.parquet not found — reports will run without fund type breakdown")

        try:
            import os as _os
            corr_path = _os.path.join("reports", "generated", "correlations_by_type.csv")
            correlations_by_type = pd.read_csv(corr_path)
            print("✓ Correlations by type loaded")
        except FileNotFoundError:
            correlations_by_type = None
            print("⚠ correlations_by_type.csv not found")

        analyzer = DeepSeekAnalyzer()

        print("\n" + "=" * 60)
        print("1/3 — Generating Executive Report")
        print("=" * 60)
        report = analyzer.generate_executive_report(
            anomaly_summary, market_data, anomaly_by_type=anomaly_by_type
        )
        analyzer.save_report(report, "executive_report.md")
        print("\n" + report)

        print("\n" + "=" * 60)
        print("2/3 — Generating Geopolitical Impact Report")
        print("=" * 60)
        geo_report = analyzer.generate_geopolitical_report(
            anomaly_summary, market_data, anomaly_by_type=anomaly_by_type
        )
        analyzer.save_report(geo_report, "geopolitical_report.md")
        print("\n" + geo_report)

        if anomaly_by_type is not None:
            print("\n" + "=" * 60)
            print("3/3 — Generating Fund Type Report")
            print("=" * 60)
            type_report = analyzer.generate_fund_type_report(
                anomaly_by_type, market_data, correlations_by_type=correlations_by_type
            )
            analyzer.save_report(type_report, "fund_type_report.md")
            print("\n" + type_report)
        else:
            print("\n⚠ Skipping Fund Type Report — re-run pipeline with fund_type data first.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run the data collection and anomaly detection steps first.")
