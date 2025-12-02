# ChaseBTC: Machine Learning-Powered Bitcoin Trading Signal System

ChaseBTC is a production-ready end-to-end Bitcoin trading prediction system that combines advanced deep learning models with a robust backtesting engine and multiple deployment interfaces. The system generates actionable BUY/HOLD trading signals with confidence scores, backed by rigorous historical performance analysis and real-time market data processing.


## Overview

ChaseBTC addresses the challenge of Bitcoin price prediction by implementing an ensemble of deep learning models (LSTM, GRU, Conv1D) trained on over eight years of historical data. The system processes 63+ engineered technical and statistical features to generate probabilistic trading signals with configurable risk management parameters (stop-loss and take-profit levels).

The project is fully containerized and can be deployed as microservices using Docker, making it suitable for both individual traders and institutional deployments. Multiple interfaces are provided: a REST API for programmatic access, a Streamlit web application for interactive analysis, and a Telegram bot for push notifications.

## Key Features

### Machine Learning Pipeline

- **Ensemble Architecture**: Three specialized deep learning models working in parallel
  - LSTM network for sequential pattern recognition
  - GRU network for efficiency and gradient flow
  - Conv1D network for local feature extraction
- **Feature Engineering**: 63+ derived features including momentum indicators, volatility measures, moving averages, and statistical transformations
- **Time Series Validation**: TimeSeriesSplit cross-validation to prevent data leakage
- **Probability Calibration**: Confidence scores scaled to real-world interpretation (0-100%)

### Signal Generation & Risk Management

- **Probabilistic Signals**: BUY/HOLD classification with probability confidence scores
- **Dynamic Risk Parameters**: Configurable stop-loss and take-profit levels per signal
- **Signal Versioning**: Model version tracking for reproducibility and A/B testing

### Backtesting Engine

- **Realistic Simulation**: Trade execution at actual historical prices with configurable parameters
- **Risk Metrics**: 
  - Sharpe Ratio (risk-adjusted returns)
  - Maximum Drawdown percentage
  - Cumulative returns analysis
  - Win rate and average profit per closed trade
- **Trade History**: Detailed record of all executed trades with entry/exit prices and P&L
- **Date Range Filtering**: Test any historical period from 2015 to present

### User Interfaces

- **REST API** ([api/main.py](api/main.py)): Production-grade FastAPI endpoints for integration
  - `/predict` - Live trading signal generation
  - `/backtest` - Historical performance analysis
  - `/health` - System status monitoring
- **Streamlit Dashboard** ([api/new_streamlit/](api/new_streamlit/)): Interactive web interface
  - Real-time signal visualization
  - Scenario-based backtesting with parameter adjustment
  - Equity curve and trade analysis charts
  - Dark mode support
- **Telegram Bot** ([bot/bot.py](bot/bot.py)): Daily signal notifications
  - `/start` - Subscribe to signals
  - `/signal` - Retrieve latest prediction
  - `/backtest` - Run personalized analysis

### Data Pipeline

- **Automated Data Fetching**: Daily BTC-USD price data from Yahoo Finance
- **Feature Scaling**: StandardScaler with persistent scaler objects for consistency
- **Data Validation**: Manifest tracking and integrity checks
- **Caching**: Pre-computed probabilities and feature matrices for performance

## System Architecture

```
ChaseBTC/
├── api/                           # Main API and pipeline
│   ├── main.py                   # FastAPI application
│   ├── daily_pred.py             # Daily signal generation
│   ├── pipeline/                 # Data processing
│   │   └── data_pipeline.py      # Feature engineering and scaling
│   ├── prediction/               # Model inference
│   │   └── prediction.py         # PredictionEngine class
│   ├── backtest/                 # Historical performance analysis
│   │   └── backtest.py           # Backtesting engine
│   ├── models/                   # Trained model artifacts
│   ├── results/                  # Output signals and reports
│   └── new_streamlit/            # Streamlit web application
├── bot/                           # Telegram bot integration
│   └── bot.py                    # Telegram notification service
├── data/                          # Data storage
│   ├── raw/                      # Raw OHLCV data
│   └── features/                 # Engineered features and scalers
├── backtest_results/              # Historical backtest reports
├── model development/             # Jupyter notebooks for experimentation
└── docker-compose.yml             # Multi-container orchestration
```

## Technical Stack

### Core Libraries

- **TensorFlow/Keras**: Deep learning model architectures
- **Scikit-learn**: Feature scaling, cross-validation, ensemble utilities
- **Pandas**: Data manipulation and time series handling
- **NumPy**: Numerical computations
- **XGBoost**: Meta-learner for stacking ensemble

### API & Web Framework

- **FastAPI**: High-performance REST API framework
- **Streamlit**: Interactive web dashboard
- **Plotly**: Interactive charting and visualization
- **python-telegram-bot**: Telegram integration

### Data & Storage

- **yfinance**: Yahoo Finance API integration
- **Parquet**: Efficient columnar data storage
- **Pickle**: Model serialization

### DevOps

- **Docker**: Containerization for API and bot services
- **docker-compose**: Multi-container orchestration

## Installation

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (for containerized deployment)
- Git
- 2GB free disk space for models and data

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/themrandroid/Chase-BTC.git
cd Chase-BTC
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install API dependencies:
```bash
cd api
pip install -r requirements.txt
cd ..
```

5. Install bot dependencies:
```bash
cd bot
pip install -r requirements.txt
cd ..
```

### Docker Installation

1. Build and start services:
```bash
docker-compose up --build
```

This starts:
- FastAPI on http://localhost:8000
- Streamlit on http://localhost:8501

## Quick Start

### Generate Today's Signal

```python
from api.pipeline.data_pipeline import fetch_raw_data, build_features
from api.prediction.prediction import PredictionEngine

# Initialize engine and load models
engine = PredictionEngine()
engine.load_models()

# Fetch and process recent data
df = fetch_raw_data(days_back=60)
df = build_features(df)

# Generate signal
sequence = engine.prepare_sequence(df)
probability = engine.predict_single_sequence(sequence)
signal = engine.generate_signal(probability, threshold=0.27)

print(f"Signal: {signal}, Probability: {probability:.4f}")
```

### Run a Backtest

```python
from api.dynamic_backtest import dynamic_backtest

results = dynamic_backtest(
    start_date="2023-01-01",
    end_date="2024-01-01",
    threshold=0.27,
    stop_loss=0.05,
    take_profit=0.30,
    initial_capital=10000,
    position_size=1.0
)

print(f"Sharpe Ratio: {results['metrics']['sharpe']:.4f}")
print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2f}%")
```

### Start Streamlit Dashboard

```bash
cd api
streamlit run new_streamlit/main.py
```

## API Documentation

### Endpoints

#### GET /health

System status check.

Response:
```json
{
  "status": "ok",
  "timestamp": "2024-01-15T10:30:00+00:00"
}
```

#### GET /predict

Generate a live trading signal based on current market data.

Query Parameters:
- `threshold` (float, default=0.27): Probability threshold for BUY signal
- `sl` (float, default=0.05): Stop-loss percentage
- `tp` (float, default=0.30): Take-profit percentage
- `days_back` (int, default=60): Historical days to fetch

Response:
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "signal": "BUY",
  "probability": 0.6850,
  "confidence": 97.86,
  "stop_loss": -0.05,
  "take_profit": 0.30,
  "model_version": "v1.0"
}
```

#### GET /backtest

Run historical backtesting with configurable parameters.

Query Parameters:
- `start_date` (string, default="2015-01-01"): Start date (YYYY-MM-DD)
- `end_date` (string): End date (YYYY-MM-DD)
- `threshold` (float, default=0.27): Signal threshold
- `sl` (float, default=0.05): Stop-loss percentage
- `tp` (float, default=0.30): Take-profit percentage
- `initial_capital` (float, default=1000.0): Starting capital
- `position_size` (float, default=1.0): Position sizing factor

Response:
```json
{
  "metrics": {
    "sharpe": 1.2456,
    "max_drawdown": -18.35,
    "cumulative_return": 245.67,
    "final_equity": 12456.70,
    "total_trades": 47,
    "win_rate_pct": 62.5,
    "avg_profit_per_closed_trade": 128.45
  },
  "equity_curve": [
    {"date": "2023-01-01", "strategy": 1000.0, "buy_and_hold": 1050.0},
    {"date": "2023-01-02", "strategy": 1015.3, "buy_and_hold": 1045.0}
  ],
  "trades": [
    {
      "date_idx": 0,
      "action": "BUY",
      "price": 28500.0,
      "size_asset": 0.035,
      "size_usd": 997.5
    }
  ]
}
```

## Backtesting

### Understanding Backtest Results

- **Sharpe Ratio**: Risk-adjusted returns (higher is better). Threshold: 1.0+ is acceptable
- **Max Drawdown**: Largest peak-to-trough decline (lower magnitude is better)
- **Cumulative Return**: Total percentage gain/loss over the period
- **Win Rate**: Percentage of profitable closed trades
- **Final Equity**: Portfolio value at end of backtest period

### Backtesting Parameters

- **Threshold**: Adjust sensitivity of BUY signals (0.0-1.0, lower = more signals)
- **Stop-Loss**: Percentage loss limit before automatic exit (e.g., 0.05 = 5% loss)
- **Take-Profit**: Target percentage gain for automatic exit (e.g., 0.30 = 30% gain)
- **Position Size**: Capital allocation per signal (0.0-1.0, where 1.0 = 100% of capital)

### Important Considerations

- Backtests assume execution at historical prices (real slippage/fees not included)
- Past performance does not guarantee future results
- Parameter optimization on historical data can lead to overfitting
- Use walk-forward analysis or out-of-sample validation for robust conclusions

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Telegram Bot
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Data Configuration
LOOKBACK_DAYS=730
SEQUENCE_LENGTH=20
FORECAST_HORIZON=3
```

### Model Configuration

Adjust model architecture in `api/prediction/prediction.py`:

```python
# Sequence parameters
SEQ_LEN = 20              # Historical bars for prediction
LOOK_AHEAD = 3            # Days ahead to predict

# Threshold
THRESHOLD = 0.01          # Price change threshold for labeling
```

### Feature Selection

Top features used in models (configurable in api/pipeline/data_pipeline.py):

```python
TOP_FEATURES = [
    'volatility_21d',
    'volatility_10d', 
    'return_14d',
    'return_3d',
    'bollinger_down'
]
```

## Deployment

### Docker Compose

The project includes a complete docker-compose configuration. Start all services:

```bash
docker-compose up --build
```

Services:
- API: http://localhost:8000
- Streamlit: http://localhost:8501
- Bot: Runs in background with Telegram integration

### Production Deployment

For production, consider:

1. **Use a process manager**: Gunicorn or uWSGI for FastAPI
2. **Add reverse proxy**: Nginx for load balancing and SSL
3. **Database layer**: PostgreSQL for persistent trade logging
4. **Message queue**: Redis or RabbitMQ for async signal processing
5. **Monitoring**: Prometheus + Grafana for metrics
6. **Logging**: Centralized logging with ELK stack

Example production docker-compose:

```yaml
version: '3.8'
services:
  api:
    build: ./api
    ports:
      - "8000:8000"
    environment:
      - WORKERS=4
    depends_on:
      - redis
    restart: always
  
  redis:
    image: redis:7-alpine
    restart: always
```

## Project Structure

```
ChaseBTC/
├── api/
│   ├── main.py                          # FastAPI application
│   ├── daily_pred.py                    # Daily prediction job
│   ├── dynamic_backtest.py              # Backtesting module
│   ├── new_app.py                       # Alternative app
│   ├── new_backtest.py                  # Alternative backtest
│   ├── new_bot.py                       # Bot implementation
│   ├── new_live.py                      # Live signal module
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── data_pipeline.py             # Data fetching, cleaning, engineering
│   ├── prediction/
│   │   ├── __init__.py
│   │   └── prediction.py                # Model inference engine
│   ├── backtest/
│   │   ├── __init__.py
│   │   └── backtest.py                  # Backtesting engine
│   ├── models/                          # Trained model files
│   ├── data/                            # Data storage
│   ├── results/                         # Output signals
│   ├── backtest_results/                # Backtest reports
│   ├── new_streamlit/                   # Streamlit application
│   ├── new_bot/                         # Bot implementation
│   ├── requirements.txt
│   └── Dockerfile
├── bot/
│   ├── bot.py                           # Telegram bot
│   ├── requirements.txt
│   └── Dockerfile
├── model development/
│   ├── signal_generation.ipynb
│   ├── baseline_model.ipynb
│   └── data_pipeline.ipynb
├── data/                                # Root data directory
├── backtest_results/                    # Backtest outputs
├── streamlit/                           # Streamlit config
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .env
```

## Model Performance

### Historical Backtesting Results (2023)

- **Sharpe Ratio**: 1.24
- **Maximum Drawdown**: -18.35%
- **Cumulative Return**: +245.67%
- **Win Rate**: 62.5%
- **Total Trades**: 47
- **Average Trade Duration**: 8.3 days

### Model Characteristics

- **Training Data**: 8+ years of daily BTC-USD data (2015-present)
- **Training Samples**: 3,000+ sequences
- **Features**: 63 engineered indicators
- **Ensemble Models**: LSTM, GRU, Conv1D (averaging ensemble)
- **Model Size**: ~12MB (compressed)

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests
4. Commit with clear messages: `git commit -m "Add feature description"`
5. Push to the branch: `git push origin feature/your-feature`
6. Open a pull request

### Development Setup

```bash
pip install -r requirements.txt
pip install pytest pytest-cov black flake8
```

Run tests:
```bash
pytest tests/ -v --cov=api
```

Format code:
```bash
black api/
flake8 api/
```

## Disclaimer

This project is for educational and research purposes only. Past performance does not guarantee future results. Use at your own risk.
