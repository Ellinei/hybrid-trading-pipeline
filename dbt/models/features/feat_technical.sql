{{ config(materialized='table') }}

-- Join returns (log/pct changes) with the OHLCV data to access open/high/low for
-- candle-structure features and ATR. We extract only open/high/low from stg_ohlcv
-- to avoid column-name collisions on close and volume, which both upstream models share.
with returns as (
    select * from {{ ref('int_returns') }}
),
prices as (
    select symbol, timestamp, open, high, low
    from {{ ref('stg_ohlcv') }}
),
joined as (
    select
        r.symbol,
        r.timestamp,
        r.close,
        r.volume,
        r.log_return_1h,
        r.log_return_7d,
        r.pct_change_1h,
        r.pct_change_4h,
        r.pct_change_24h,
        p.open,
        p.high,
        p.low
    from returns r
    join prices p using (symbol, timestamp)
),
windowed as (
    select
        symbol,
        timestamp,
        close,
        open,
        high,
        low,
        volume,
        log_return_1h,
        log_return_7d,
        pct_change_1h,
        pct_change_4h,
        pct_change_24h,
        -- 14-period high/low for RSI proxy and ATR proxy
        max(close) over w14    as max_close_14,
        min(close) over w14    as min_close_14,
        max(high)  over w14    as high_14,
        min(low)   over w14    as low_14,
        -- Moving averages (SMA used as EMA proxy)
        avg(close) over w12    as ema_12_proxy,
        avg(close) over w26    as ema_26_proxy,
        avg(close) over w20    as sma_20,
        avg(close) over w50    as sma_50,
        stddev(close) over w20 as stddev_20,
        -- Volume moving average for conviction signal
        avg(volume) over w20   as vol_ma_20
    from joined
    window
        w12 as (partition by symbol order by timestamp rows between 11 preceding and current row),
        w14 as (partition by symbol order by timestamp rows between 13 preceding and current row),
        w20 as (partition by symbol order by timestamp rows between 19 preceding and current row),
        w26 as (partition by symbol order by timestamp rows between 25 preceding and current row),
        w50 as (partition by symbol order by timestamp rows between 49 preceding and current row)
),
final as (
select
    symbol,
    timestamp,
    close,
    volume,
    log_return_1h,
    log_return_7d,
    pct_change_1h,
    pct_change_4h,
    pct_change_24h,
    -- RSI proxy: position of close within the 14-period high/low range (0–100)
    case
        when (max_close_14 - min_close_14) > 0
        then 100.0 * (close - min_close_14) / (max_close_14 - min_close_14)
        else 50.0
    end as rsi_14_proxy,
    -- MACD line: fast MA minus slow MA
    ema_12_proxy - ema_26_proxy as macd_line,
    -- Bollinger Bands
    sma_20                         as bb_middle,
    sma_20 + (2 * stddev_20)       as bb_upper,
    sma_20 - (2 * stddev_20)       as bb_lower,
    case
        when stddev_20 > 0 then (close - sma_20) / stddev_20
        else 0.0
    end as bb_zscore,
    -- ATR proxy: 14-period high/low range
    high_14 - low_14                                                   as range_14,
    -- Volatility expansion: Bollinger Band width as % of SMA-20
    case when sma_20 > 0 then 4 * stddev_20 / sma_20 * 100 end        as bb_width,
    -- ATR normalized by price (% of close)
    case when close > 0 then (high_14 - low_14) / close * 100 end     as atr_normalized,
    -- Trend position relative to moving averages
    case when sma_20 > 0 then (close - sma_20) / sma_20 * 100 end     as close_vs_sma20,
    case when sma_50 > 0 then (close - sma_50) / sma_50 * 100 end     as close_vs_sma50,
    -- Volume conviction: current volume vs 20-bar average
    case when vol_ma_20 > 0 then volume / vol_ma_20 end                as vol_ma_ratio,
    -- Candle body: fraction of candle range covered by the body (negative = bearish)
    case when (high - low) > 0 then (close - open) / (high - low)
         else 0.0
    end                                                                as candle_body_pct,
    -- Upper wick: upper shadow as fraction of total range (rejection signal)
    (high - greatest(open, close)) / nullif(high - low, 0)            as upper_wick_pct,
    -- Stochastic %K: where close sits in the 14-bar high/low range
    100.0 * (close - low_14) / nullif(high_14 - low_14, 0)            as stoch_k,
    current_timestamp                                                  as computed_at
from windowed
)
-- Postgres can't reference select-list aliases in the same query's WHERE,
-- so the null filter lives outside the CTE that defines them.
select * from final
where log_return_1h is not null
  and upper_wick_pct is not null
  and stoch_k is not null
