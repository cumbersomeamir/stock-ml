# Trading Lab

**⚠️ RESEARCH TOOL ONLY - NOT FINANCIAL ADVICE**

A complete, modular ML-driven trading research system for equities analysis. Supports Indian equities (NSE/BSE) and US equities via yfinance, with optional extensions for news, social media, and fundamental data.

## Features

- **Modular Data Ingestion**: Support for prices (required), news, social media, fundamentals, and macro data
- **Feature Engineering**: Time-series features, event-based features, and fundamentals features
- **ML Approaches**:
  - **Supervised**: Classification (direction) and regression (volatility) with walk-forward validation
  - **Unsupervised**: Market regime detection
  - **RL**: Optional reinforcement learning agent (disabled by default)
- **Backtesting**: Walk-forward validation with transaction costs, slippage, position limits, and risk controls
- **CLI Interface**: Simple commands for the entire pipeline

## Installation

```bash
# Clone or navigate to the project directory
cd stock-ml

# Install in development mode
pip install -e .

# Optional: Install with LightGBM support
pip install -e ".[lightgbm]"

# Optional: Install with RL support
pip install -e ".[rl]"

# Install development dependencies
pip install -e ".[dev]"
```

## Quick Start

1. **Set up environment** (optional - works without API keys):
   ```bash
   cp .env.example .env
   # Edit .env to add API keys if you want to use optional data sources
   ```

2. **Download price data**:
   ```bash
   python -m trading_lab.cli download-prices --tickers "AAPL,MSFT" --start 2018-01-01 --end 2024-12-31
   ```

3. **Build features**:
   ```bash
   python -m trading_lab.cli build-features
   ```

4. **Train supervised models**:
   ```bash
   python -m trading_lab.cli train-supervised
   ```

5. **Run backtest**:
   ```bash
   python -m trading_lab.cli backtest --strategy supervised_prob_threshold
   ```

6. **Generate report**:
   ```bash
   python -m trading_lab.cli report
   ```

## Data Sources

### Required (No API Key)
- **Prices**: yfinance (NSE/BSE tickers: `.NS`, `.BO`; US: standard symbols)

### Optional (Require API Keys)
- **News**: NewsAPI (set `NEWSAPI_KEY` in `.env`)
- **Social**: Reddit (set `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`)
- **Fundamentals**: Financial Modeling Prep, Alpha Vantage (stubs provided)
- **Macro**: FRED API, RBI (stubs provided)

## CLI Commands

### Download Data
```bash
python -m trading_lab.cli download-prices \
  --tickers "RELIANCE.NS,TCS.NS,AAPL,MSFT" \
  --start 2018-01-01 \
  --end 2025-12-31
```

### Build Features
```bash
python -m trading_lab.cli build-features
```

### Train Models
```bash
# Supervised models (classification + regression)
python -m trading_lab.cli train-supervised

# Regime detection (unsupervised)
python -m trading_lab.cli train-regime-detection
```

### Backtest
```bash
python -m trading_lab.cli backtest \
  --strategy supervised_prob_threshold \
  --train-window-years 2 \
  --test-window-months 3
```

### Generate Report
```bash
python -m trading_lab.cli report
```

## Project Structure

```
trading-lab/
├── data/
│   ├── raw/           # Raw downloaded data
│   ├── processed/     # Unified and processed data
│   └── artifacts/     # Models, backtests, reports
├── src/
│   └── trading_lab/
│       ├── config/    # Configuration and settings
│       ├── common/    # Utilities (logging, io, schemas)
│       ├── data_sources/  # Data fetchers
│       ├── unify/     # Data unification
│       ├── features/  # Feature engineering
│       ├── labeling/  # Target generation
│       ├── models/    # ML models (supervised, unsupervised, RL)
│       ├── backtest/  # Backtesting engine
│       └── reports/   # Reporting and visualization
├── tests/             # Unit tests
└── CLI (cli.py)       # Command-line interface
```

## Models

### Supervised Learning

- **Classification**: Predict next-day return direction (up/down)
  - Models: Logistic Regression, Random Forest, Gradient Boosting, LightGBM (if installed)
  - Label: `y_class` (1 if return > threshold, 0 if < -threshold, else neutral/drop)

- **Regression**: Predict next-day volatility
  - Models: Ridge, Lasso, Gradient Boosting, LightGBM (if installed)
  - Label: `y_reg` (realized volatility proxy)

- **Validation**: Time-series split with purged cross-validation to avoid leakage

### Unsupervised Learning

- **Regime Detection**: KMeans/Gaussian Mixture on rolling features
  - Detects market states (bull, bear, high volatility, low volatility)
  - Regime labels appended to features for supervised models

### Reinforcement Learning (Optional)

- **Environment**: Gymnasium-based trading environment
  - Observation: Recent features window
  - Action: Position {-1, 0, +1}
  - Reward: PnL - costs - risk penalty
- **Status**: Disabled by default (requires `gymnasium` and `stable-baselines3`)
- **Usage**: Only use after supervised baselines work well

## Features

### Price-Based Features
- Returns (1d, 5d, 20d), log returns
- Rolling volatility (5d, 20d)
- Momentum indicators (RSI, MACD, ATR proxy)
- Moving average gaps
- Volume features (z-score, ratios)

### Event-Based Features (Optional)
- Daily sentiment scores (news, social)
- News count per day/ticker
- Reddit mention counts

### Fundamentals Features (Optional)
- PE, PB, dividend yield (if available)

## Backtesting

- **Walk-Forward Validation**: Train on rolling windows, test on out-of-sample periods
- **Position Sizing**: Configurable limits per asset and gross exposure
- **Costs**: Transaction costs (bps) and slippage (bps)
- **Risk Controls**:
  - Circuit breaker: Stop trading if drawdown > threshold
  - Position limits
  - Volatility targeting (optional)

### Metrics
- CAGR (Compound Annual Growth Rate)
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Turnover

## Best Practices

1. **No Data Leakage**: All features use proper time-shifting
2. **Out-of-Sample Testing**: Always validate on unseen data
3. **Walk-Forward**: Use time-series cross-validation, not random splits
4. **Transaction Costs**: Always include realistic costs in backtests
5. **Risk Management**: Respect position limits and drawdown thresholds
6. **Overfitting**: Monitor train vs test performance carefully

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=trading_lab

# Run specific test
pytest tests/test_features.py
```

## Development

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Install pre-commit hooks
pre-commit install
```

## Warnings and Disclaimers

⚠️ **THIS IS A RESEARCH TOOL, NOT FINANCIAL ADVICE**

- This software is for research and educational purposes only
- Past performance does not guarantee future results
- Always validate models on out-of-sample data
- Real trading involves additional risks not captured in backtests
- Do not use this software for actual trading without proper risk management and regulatory compliance
- The authors are not responsible for any financial losses

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please ensure:
- Code follows black/ruff formatting
- Tests pass
- Documentation is updated
- No data leakage in features

## Support

For issues and questions, please open an issue on the project repository.

# stock-ml
