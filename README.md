# Brazilian Investment Funds — Anomaly Detection & Market Correlation

An analytical tool that detects anomalous behavior in Brazilian investment funds, correlates anomalies with external market events, and uses AI-powered insights to anticipate future volatility.

## Overview

This project combines financial data analysis, statistical modeling, and AI to create an end-to-end analytical pipeline for Brazilian investment funds. It goes beyond descriptive analysis by identifying **why** anomalies happen and **what signals** may predict future volatility.

## Architecture

```
CVM Data → Data Processing → Anomaly Detection → Market Correlation → AI Insights → Report
                                                        ↑
                                              Yahoo Finance / BCB API
```

## Features

- **Automated Data Collection**: Fetches fund data from CVM and market data from Yahoo Finance / Central Bank of Brazil
- **Anomaly Detection**: Identifies unusual fund behavior using rolling Z-scores and statistical models
- **Market Correlation**: Maps detected anomalies against macroeconomic events (USD/BRL, interest rates, Ibovespa, VIX)
- **AI-Powered Insights**: Uses Claude API to generate natural language explanations of detected anomalies and their probable causes
- **Predictive Alerts**: Flags combinations of market signals that historically preceded fund volatility
- **Executive Reports**: Auto-generates business-ready summaries from technical findings

## Tech Stack

- **Python** — Core language
- **Pandas / NumPy** — Data processing and analysis
- **Scikit-learn** — Predictive modeling
- **Plotly** — Interactive visualizations
- **Anthropic API (Claude)** — AI-powered insights and report generation
- **yfinance** — Market data collection
- **requests** — CVM and BCB API integration

## Project Structure

```
brazilian-funds-anomaly-detector/
│
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── data_collection/
│   │   ├── __init__.py
│   │   ├── cvm_collector.py          # Fetches fund data from CVM
│   │   ├── market_collector.py       # Fetches market data (Yahoo Finance, BCB)
│   │   └── data_merger.py            # Merges fund + market datasets
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── anomaly_detector.py       # Statistical anomaly detection
│   │   ├── market_correlator.py      # Correlates anomalies with market events
│   │   └── predictive_model.py       # Volatility prediction model
│   │
│   ├── ai_insights/
│   │   ├── __init__.py
│   │   └── claude_analyzer.py        # Claude API integration for insights
│   │
│   └── visualization/
│       ├── __init__.py
│       └── dashboard.py              # Plotly interactive dashboard
│
├── notebooks/
│   └── exploratory_analysis.ipynb    # Jupyter notebook for exploration
│
├── data/
│   ├── raw/                          # Raw downloaded data
│   └── processed/                    # Cleaned and processed data
│
├── reports/
│   └── generated/                    # AI-generated reports
│
└── tests/
    └── test_anomaly_detector.py
```

## Getting Started

### Prerequisites

- Python 3.10+
- Anthropic API key ([get one here](https://console.anthropic.com/))

### Installation

```bash
# Clone the repository
git clone https://github.com/espartafps/brazilian-funds-anomaly-detector.git
cd brazilian-funds-anomaly-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### Usage

```bash
# Step 1: Collect data
python -m src.data_collection.cvm_collector

# Step 2: Collect market data
python -m src.data_collection.market_collector

# Step 3: Run anomaly detection
python -m src.analysis.anomaly_detector

# Step 4: Correlate with market events
python -m src.analysis.market_correlator

# Step 5: Generate AI insights
python -m src.ai_insights.claude_analyzer

# Step 6: Launch dashboard
python -m src.visualization.dashboard
```

## Data Sources

| Source | Data | Access |
|--------|------|--------|
| [CVM Open Data](https://dados.cvm.gov.br/) | Daily fund returns, AUM, inflows/outflows | Public CSV files |
| [Yahoo Finance](https://finance.yahoo.com/) | Ibovespa, USD/BRL, VIX | yfinance library |
| [Central Bank of Brazil](https://dadosabertos.bcb.gov.br/) | CDI, interest rates, yield curve | Public API |

## Example Output

When the model detects an anomaly, Claude generates insights like:

> *"Fund XYZ showed atypical volatility between March 10-15, 2023, with returns 3.2 standard deviations below its rolling mean. During this period, USD/BRL spiked 4.2% and the VIX rose above 25. Historical pattern analysis suggests that similar combinations of currency stress and global risk aversion preceded fund volatility in 78% of observed cases."*

## Roadmap

- [x] Project structure and documentation
- [ ] CVM data collection pipeline
- [ ] Market data collection pipeline
- [ ] Anomaly detection engine
- [ ] Market correlation analysis
- [ ] Claude API integration
- [ ] Predictive model
- [ ] Interactive dashboard
- [ ] Executive report generation

## Author

**Felipe Pereira da Silva**
Capital Markets Analyst | Analytics & Strategy

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/felipesilvanuts/)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
