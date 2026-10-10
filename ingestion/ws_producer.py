"""Binance WebSocket → Kafka producer.

Subscribes to combined streams (trade + miniTicker) for every symbol in
TRADING_SYMBOLS, serialises each event to JSON, and publishes to the
``price-ticks`` Kafka topic.

Usage:
    python -m ingestion.ws_producer
"""
from __future__ import annotations

import json
import os
import signal
import threading
import time
from datetime import UTC, datetime
from typing import Any

import structlog
import websocket  # websocket-client
from dotenv import load_dotenv
from kafka import KafkaProducer

log = structlog.get_logger()

TOPIC = "price-ticks"
_METRICS_INTERVAL = 60  # seconds


# ---------------------------------------------------------------------------
# Pure formatting functions — module-level so tests can import them directly
# ---------------------------------------------------------------------------

def _to_iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _ms_to_iso(ms: int) -> str:
    return _to_iso(datetime.fromtimestamp(ms / 1000, tz=UTC))


def _now_iso() -> str:
    return _to_iso(datetime.now(UTC))


def _format_trade(data: dict[str, Any]) -> dict[str, Any]:
    """Format raw Binance trade stream data into our canonical schema."""
    return {
        "stream_type": "trade",
        "symbol": data["s"],
        "price": float(data["p"]),
        "quantity": float(data["q"]),
        "timestamp": _ms_to_iso(data["T"]),
        "trade_id": data["t"],
        "is_buyer_maker": data["m"],
        "received_at": _now_iso(),
    }


def _format_mini_ticker(data: dict[str, Any]) -> dict[str, Any]:
    """Format raw Binance 24hr miniTicker stream data into our schema."""
    return {
        "stream_type": "miniTicker",
        "symbol": data["s"],
        "close_price": float(data["c"]),
        "open_price": float(data["o"]),
        "high_price": float(data["h"]),
        "low_price": float(data["l"]),
        "base_volume": float(data["v"]),
        "quote_volume": float(data["q"]),
        "timestamp": _ms_to_iso(data["E"]),
        "received_at": _now_iso(),
    }


# ---------------------------------------------------------------------------
# Producer class
# ---------------------------------------------------------------------------

class BinanceStreamProducer:
    """Streams Binance trade events to Kafka with reconnect and metrics."""

    _MAX_RETRIES = 5

    def __init__(self) -> None:
        load_dotenv()
        testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
        symbols_raw = os.getenv("TRADING_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT")
        bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

        self._symbols = [s.strip().lower() for s in symbols_raw.split(",") if s.strip()]
        self._ws_url = self._build_url(testnet)
        self._stopping = False
        self._ws: websocket.WebSocketApp | None = None
        self._msg_counter = 0
        self._reconnect_count = 0

        # Per-symbol metrics: count, bytes, last_ts
        self._metrics: dict[str, dict] = {
            s.upper(): {"count": 0, "bytes": 0, "last_ts": None}
            for s in self._symbols
        }

        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap,
            key_serializer=lambda k: k.encode("utf-8"),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3,
        )
        self._log = structlog.get_logger().bind(testnet=testnet, topic=TOPIC)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start streaming; blocks until stopped."""
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        self._schedule_metrics()
        self._log.info("producer_starting", symbols=self._symbols, url=self._ws_url)

        try:
            self._connect_with_retry()
        finally:
            self._producer.close()
            self._log.info("producer_stopped")

    def stop(self) -> None:
        self._stopping = True
        if self._ws:
            self._ws.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_url(self, testnet: bool) -> str:
        # Binance has two subdomains: testnet.binance.vision is REST-only;
        # WebSocket streams live on stream.testnet.binance.vision.
        # Same split on prod: api.binance.com vs stream.binance.com.
        base = (
            "wss://stream.testnet.binance.vision/stream"
            if testnet
            else "wss://stream.binance.com:9443/stream"
        )
        streams = "/".join(
            f"{s}@trade" for s in self._symbols
        ) + "/" + "/".join(
            f"{s}@miniTicker" for s in self._symbols
        )
        return f"{base}?streams={streams}"

    # Minimum seconds a connection must stay open to count as «healthy» — once
    # we cross this threshold, the consecutive-failure counter resets so a long-
    # lived stream survives normal disconnect/reconnect cycles.
    _HEALTHY_THRESHOLD_S = 30

    def _connect_with_retry(self) -> None:
        consecutive_failures = 0
        while not self._stopping:
            connect_started = time.monotonic()

            self._ws = websocket.WebSocketApp(
                self._ws_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open,
            )
            self._ws.run_forever()

            if self._stopping:
                break

            # If the connection lasted long enough to be considered healthy,
            # reset the counter — only rapid back-to-back failures are bad.
            connection_duration = time.monotonic() - connect_started
            if connection_duration >= self._HEALTHY_THRESHOLD_S:
                consecutive_failures = 0

            consecutive_failures += 1
            self._reconnect_count += 1

            if consecutive_failures > self._MAX_RETRIES:
                self._log.error(
                    "max_retries_exceeded",
                    max_retries=self._MAX_RETRIES,
                    total_reconnects=self._reconnect_count,
                )
                break

            wait = 2 ** (consecutive_failures - 1)  # 1, 2, 4, 8, 16 seconds
            self._log.warning(
                "reconnecting",
                consecutive_failures=consecutive_failures,
                last_connection_s=round(connection_duration, 1),
                wait_s=wait,
                total_reconnects=self._reconnect_count,
            )
            time.sleep(wait)

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        self._log.info("websocket_connected")

    def _on_close(
        self, ws: websocket.WebSocketApp, close_status_code: int | None, close_msg: str | None
    ) -> None:
        self._log.info(
            "websocket_closed",
            code=close_status_code,
            message=close_msg,
        )

    def _on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        self._log.error("websocket_error", error=str(error))

    def _on_message(self, ws: websocket.WebSocketApp, raw: str) -> None:
        try:
            payload = json.loads(raw)
            data = payload.get("data", {})
            event_type = data.get("e", "")

            if event_type == "trade":
                msg = _format_trade(data)
            elif event_type == "24hrMiniTicker":
                msg = _format_mini_ticker(data)
            else:
                return

            symbol = msg["symbol"]
            future = self._producer.send(TOPIC, key=symbol, value=msg)
            future.add_errback(lambda e: self._log.error("kafka_send_failed", exc=str(e)))

            # Update metrics
            self._msg_counter += 1
            m = self._metrics.setdefault(symbol, {"count": 0, "bytes": 0, "last_ts": None})
            m["count"] += 1
            m["bytes"] += len(json.dumps(msg).encode("utf-8"))
            m["last_ts"] = msg.get("timestamp")

            if self._msg_counter % 100 == 0:
                self._log.info("messages_sent", total=self._msg_counter)

        except Exception:
            self._log.exception("message_processing_error", raw=raw[:200])

    def _log_metrics(self) -> None:
        self._log.info(
            "metrics",
            reconnects=self._reconnect_count,
            total_messages=self._msg_counter,
            per_symbol={
                sym: {"count": m["count"], "kb": round(m["bytes"] / 1024, 1)}
                for sym, m in self._metrics.items()
            },
        )

    def _schedule_metrics(self) -> None:
        if not self._stopping:
            t = threading.Timer(_METRICS_INTERVAL, self._tick_metrics)
            t.daemon = True
            t.start()

    def _tick_metrics(self) -> None:
        self._log_metrics()
        self._schedule_metrics()

    def _handle_shutdown(self, signum: int, frame: object) -> None:
        self._log.info("shutdown_requested", signal=signum)
        self.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BinanceStreamProducer().start()
