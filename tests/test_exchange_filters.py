"""Unit tests for execution.exchange_filters — pure functions, no mocks needed.

Run:
    poetry run pytest tests/test_exchange_filters.py -v
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from execution.exchange_filters import (
    FilterViolation,
    SymbolFilters,
    parse_symbol_filters,
    round_price_to_tick,
    round_qty_down,
    round_step,
    validate_notional,
)


def _symbol_info(notional_filter_type="MIN_NOTIONAL", notional_key="minNotional"):
    return {
        "symbol": "BTCUSDT",
        "filters": [
            {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "9000"},
            {"filterType": "PRICE_FILTER", "tickSize": "0.01",
             "minPrice": "0.01", "maxPrice": "1000000"},
            {"filterType": notional_filter_type, notional_key: "10.00"},
        ],
    }


class TestParseSymbolFilters:
    def test_extracts_step_tick_min_notional_from_symbol_info(self):
        filters = parse_symbol_filters(_symbol_info())
        assert filters.step_size == Decimal("0.001")
        assert filters.min_qty == Decimal("0.001")
        assert filters.tick_size == Decimal("0.01")
        assert filters.min_notional == Decimal("10.00")

    def test_handles_notional_filter_type_variant(self):
        info = _symbol_info(notional_filter_type="NOTIONAL", notional_key="notional")
        filters = parse_symbol_filters(info)
        assert filters.min_notional == Decimal("10.00")


class TestRoundStep:
    def test_floors_to_step_size(self):
        assert round_step(1.2346, Decimal("0.001")) == 1.234

    def test_zero_step_returns_value_unchanged(self):
        assert round_step(1.23456, Decimal("0")) == 1.23456


class TestRoundQtyDown:
    def _filters(self):
        return SymbolFilters(
            step_size=Decimal("0.001"),
            min_qty=Decimal("0.001"),
            tick_size=Decimal("0.01"),
            min_notional=Decimal("10"),
        )

    def test_floors_to_step_size(self):
        assert round_qty_down(1.2346, self._filters()) == 1.234

    def test_returns_zero_below_min_qty(self):
        assert round_qty_down(0.0005, self._filters()) == 0.0

    def test_decimal_precision_no_float_noise(self):
        result = round_qty_down(1.2346, self._filters())
        assert result == 1.234
        assert str(result) == "1.234"


class TestRoundPriceToTick:
    def test_floors_to_tick_size(self):
        filters = SymbolFilters(
            step_size=Decimal("0.001"),
            min_qty=Decimal("0.001"),
            tick_size=Decimal("0.01"),
            min_notional=Decimal("10"),
        )
        assert round_price_to_tick(123.4567, filters) == 123.45


class TestValidateNotional:
    def _filters(self):
        return SymbolFilters(
            step_size=Decimal("0.001"),
            min_qty=Decimal("0.001"),
            tick_size=Decimal("0.01"),
            min_notional=Decimal("10"),
        )

    def test_raises_filter_violation_below_minimum(self):
        with pytest.raises(FilterViolation):
            validate_notional(qty=0.01, price=100.0, filters=self._filters())

    def test_passes_when_at_or_above_minimum(self):
        validate_notional(qty=1.0, price=100.0, filters=self._filters())
