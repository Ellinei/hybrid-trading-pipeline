# Hybrid Trading Pipeline

**Hybrid streaming + batch research pipeline for crypto strategy evaluation and paper trading on Binance market data.**

[![CI](https://github.com/Ellinei/hybrid-trading-pipeline/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Ellinei/hybrid-trading-pipeline/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.13-blue)
See the latest verification record in [the implementation log](log/phase13-implementation-progress.md).

## Project Status

The pipeline, evaluation harness, and paper-trading operations are complete. The trading strategy has **no proven edge**.

- **Historical result:** the corrected causal baseline (`exp008_corrected_20260925`) made 75 trades with profit factor 1.89. It fails the pre-registered gate of at least 100 trades, and its profit factor moves between 1.18 and 1.89 depending on one learning-rate setting (0.1 vs 0.05). See [the experiment ledger](docs/experiments.md).
- **Paper trial:** an operations-only trial logged 192 heartbeat-hours and 194 hourly cycles with 0 cycle errors, first on a Windows PC and then on an AWS VM. It opened no trades, so it gives no performance evidence. A deliberate stale-data test blocked new entries while SELL signals were still evaluated. See [the trial record](docs/forward-trial.md).

## Architecture Overview

See [docs/architecture.md](docs/architecture.md) for the full system diagram and design decisions.

The pipeline runs two layers in concert:

- **Streaming layer** — Binance WebSocket → Redpanda/Kafka → real-time feature processor writes VWAP, volatility, and trade-count snapshots to TimescaleDB every 60 seconds.
- **Batch layer** — Airflow runs an hourly chain of completed-candle backfill and dbt build/tests, producing clean staging views and a `features.feat_technical` table (including RSI, MACD, Bollinger Bands, ATR, stochastic, and candle structure).

The **multi-signal fusion engine** combines technical rules and an **XGBoost** model with 18 features into BUY / HOLD / SELL decisions. RSS sentiment is available but disabled in the adopted strategy to match historical evaluation. Streaming snapshots are monitored separately; the current strategy uses batch technical features. Its BTC macro regime filter permits entries only when the prior completed UTC day's classification is available and risk-on.

Execution supports Binance entries and protective **OCO (One-Cancels-Other) orders**, with exchange filter checks, and a separate persistent paper account. A historical **backtest harness** evaluates signal/risk logic against OHLCV history with walk-forward training and cached predictions. Paper fills, bar-based historical fills, and exchange execution have different limitations; see the operations guide.

## Tech Stack

| Layer | Technology |
|---|---|
| Market data | Binance API (WebSocket + REST, testnet-ready) |
| Message broker | **Redpanda** (Kafka-compatible, dual listeners) |
| Orchestration | **Apache Airflow 2.8** (custom Docker image) |
| Transformations | **dbt** — staging → intermediate → features; versions pinned in Poetry and the Airflow image |
| Storage | **PostgreSQL 15 + TimescaleDB** (hypertables) |
| ML model | **XGBoost** (8-bar return above a 0.3% hurdle, 18 features) |
| Sentiment | **VADER + public RSS** (CoinDesk, Cointelegraph, CryptoPanic); optional, disabled in the adopted strategy |
| Execution | **Binance OCO orders** — stop-loss + take-profit on entry |
| Backtesting | Custom engine + 192-combo parameter sweep |
| Monitoring | **Streamlit** dashboard |
| Containerisation | Docker Compose |

## Quick Start

```bash
git clone https://github.com/Ellinei/hybrid-trading-pipeline.git
cd hybrid-trading-pipeline

# 1. Configure environment
cp .env.example .env      # fill in your Binance testnet API keys

# 2. Start all services (Postgres, Redpanda, Airflow)
make up

# 3. Install Python dependencies
poetry install

# 4. Create Kafka topics
poetry run python -m streaming.setup_topics

# 5. Backfill 90 days of OHLCV data
poetry run python -m ingestion.rest_backfill

# 6. Validate connectivity + data
poetry run python -m ingestion.validate
```

## Service URLs

| Service | Host (external) | Docker network (internal) |
|---|---|---|
| Airflow UI | http://localhost:8080 | — |
| Redpanda Console | http://localhost:8085 | — |
| Redpanda (Kafka) | `localhost:19092` | `redpanda:9092` |
| PostgreSQL (TimescaleDB) | `localhost:5433` | `postgres:5432` |

> **Port notes:** Postgres is on `5433` to avoid colliding with a native Windows postgres install.
> Redpanda runs two listeners — host scripts use `localhost:19092`, Docker-internal services use
> `redpanda:9092`. See [docs/architecture.md](docs/architecture.md) for details.

## Running the Signal Engine

```bash
# Build and validate features, then publish their entry-readiness state
poetry run python -m cli features

# Train the XGBoost direction model (runs GridSearchCV hyperparameter tuning)
poetry run python -m signals.ml_model

# Run the full signal report
poetry run python -m signals.aggregator
```

## Running the Streaming Pipeline

```bash
# Terminal 1: WebSocket -> Kafka
poetry run python -m ingestion.ws_producer

# Terminal 2: Kafka -> real-time features -> Postgres
poetry run python -m streaming.feature_processor
```

## Backtesting

New operations features are available through `python -m cli trade --paper`
and the dashboard's **Operations** and **Experiments** pages. The bot checks
source-data freshness before new entries and journals all signal decisions.
See [Operations guide](docs/operations.md) for setup, thresholds, paper-account
cost assumptions, and experiment comparisons.

```bash
# Replay the full strategy against historical OHLCV data
python -m backtest.run_backtest

# Grid sweep RiskConfig parameters (192 combinations)
python -m backtest.sweep --top 15 --output sweep_results.csv

# Fine-grained sweep focused on stop-loss tuning (72 combinations)
python -m backtest.sweep --fine --top 15
```

## Project Structure

```
hybrid-trading-pipeline/
├── docker-compose.yml
├── pyproject.toml              # Poetry deps + ruff/pytest config
├── Makefile                    # Dev workflow shortcuts
├── .env.example
├── docs/
│   └── architecture.md
├── ingestion/                  # Binance REST backfill + WebSocket producer
│   ├── binance_client.py
│   ├── ws_producer.py
│   ├── rest_backfill.py
│   ├── models.py               # SQLAlchemy ORM + TimescaleDB hypertables
│   └── validate.py
├── streaming/                  # Kafka consumer + real-time feature processor
│   ├── feature_processor.py
│   └── setup_topics.py
├── airflow/
│   ├── Dockerfile              # Custom image: Airflow + dbt + project deps
│   ├── requirements.txt
│   └── dags/
│       └── ohlcv_backfill_dag.py # Hourly ingestion → validated features
├── dbt/                        # Staging / intermediate / features models
│   ├── models/
│   │   ├── staging/            # stg_ohlcv (view)
│   │   ├── intermediate/       # int_returns (view) — log returns, pct changes
│   │   └── features/           # feat_technical (table): 25 columns, 18 used by the model
│   ├── tests/
│   └── README.md
├── signals/                    # Multi-signal fusion engine
│   ├── base.py                 # Signal + AggregatedSignal dataclasses
│   ├── technical.py            # Rule-based: RSI, MACD, Bollinger Bands
│   ├── ml_model.py             # XGBoost: 8-bar return hurdle, 18 features
│   ├── sentiment.py            # VADER + RSS feeds
│   └── aggregator.py           # Weighted fusion -> BUY / HOLD / SELL
├── execution/                  # Live order execution
│   ├── risk_manager.py         # RiskConfig, position sizing, ATR stops
│   ├── trade_executor.py       # Binance OCO order placement
│   ├── exchange_filters.py     # Lot size, min notional, price precision
│   └── bot_runner.py           # Main trading loop
├── backtest/                   # Historical strategy evaluation
│   ├── engine.py               # Replay engine — identical logic to live
│   ├── sim_risk_manager.py     # SimulatedRiskManager (overrides I/O only)
│   ├── sweep.py                # 192-combo + 72-combo fine grid sweep
│   ├── metrics.py              # Sharpe, profit factor, max drawdown
│   ├── exit_simulation.py      # Stop-loss / take-profit bar simulation
│   └── run_backtest.py         # Entry point
├── dashboard/
│   └── app.py                  # Streamlit P&L and signal monitoring
├── models/                     # Saved ML artifacts (XGBoost + scaler .pkl)
├── tests/                      # Regression and integration checks across layers
└── docker/
    └── postgres/
        └── init.sql
```

## Roadmap

- [x] **Phase 1** — Project scaffold, Docker services, TimescaleDB schema init
- [x] **Phase 2** — Binance REST client, 90-day OHLCV backfill, validation script
- [x] **Phase 3** — Binance WebSocket producer, Kafka consumer, real-time features
- [x] **Phase 4** — Custom Airflow image, dbt staging → intermediate → features, 2 DAGs
- [x] **Phase 5** — Technical + XGBoost + RSS-sentiment signal engine, weighted fusion
- [x] **Phase 6** — Risk manager, position sizing, ATR-based stop/take-profit calculation
- [x] **Phase 7** — Live Binance OCO order execution, exchange filter compliance, bot runner, Streamlit dashboard
- [x] **Phase 8** — Historical backtest harness; SimulatedRiskManager runs identical approve/size/stop logic as live; chronological multi-symbol replay; Sharpe/PF/drawdown metrics
- [x] **Phase 8.5** — 192-combo RiskConfig parameter sweep with `_CachingAggregator` (pre-computes signals once, 192x in-memory replay)
- [x] **Phase 8.6** — Feature engineering: 18 ML features (up from 8); dbt models extended with candle structure, multi-horizon returns, trend position, volume conviction; accuracy 50.3% → 54.2%, win rate 32% → 69%
- [x] **Phase 8.7** — XGBoost hyperparameter tuning via GridSearchCV + TimeSeriesSplit; fine-grained 72-combo stop-loss sweep; first profitable backtest configs found (PF 1.19, +0.37% return at SL=1.0xATR, conf>=0.62, size=1.5%)
- [x] **Phase 9** — Rolling walk-forward retraining and chronological replay
- [x] **Phase 10–11** — Experiment ledger, probability caching, CLI experiments, reports, and parity checks
- [x] **Phase 12** — Macro regime candidates and historical exp008 evaluation; exp008's acceptance did not survive the corrected causal re-run (75 trades, below the 100-trade gate)
- [x] **Phase 13** — Causal evaluation, reproducible training, hourly data refresh, and an operations-only paper trial (192 heartbeat-hours, 0 cycle errors, 0 trades; performance inconclusive)
- [ ] **Testnet execution milestone** — On hold: the corrected historical result fails its gate and the paper trial produced no trades

See [the September 2026 next-move plan](docs/next-move-2026-09-23.md) and
[implementation progress](log/phase13-implementation-progress.md). Earlier phase
metrics above are historical records, not current validation claims.

## License

MIT — see [LICENSE](LICENSE).
