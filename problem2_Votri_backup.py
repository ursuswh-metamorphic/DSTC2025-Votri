#!/usr/bin/env python3
"""
run_demo_pipeline.py
A runnable, self-contained implementation of the pipeline you requested.
- Loads /home/ursuswh/DSTC/DSTC.csv
- Data cleaning, target construction (rf_daily), train-only RFE selection
- KMeans clustering (k=5) on selected features, SOM per-cluster to pick representatives (~10 final features)
- Correlation filtering (>0.8)
- Multi-ticker sliding window building (WINDOW=30)
- CNN + LSTM + Multi-Head Attention model with ticker embedding
- Train, evaluate, per-ticker AUC report and simple backtests

This script is intentionally conservative: smaller RF / fewer SOM iterations / short epochs to run quickly for a smoke test.
Edit constants at top to tune.
"""

import os
import gc
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import math
import numpy as np
import pandas as pd
import datetime
from collections import Counter

# ML libs
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# optional SOM
try:
    from minisom import MiniSom
    HAS_SOM = True
except Exception:
    HAS_SOM = False

# Attempt to initialize vendor indicator client (FiinQuantX). The user requested
# that all indicators be computed by the vendor library; if credentials are not
# present we raise an error so the user can provide credentials or switch modes.

from dotenv import load_dotenv
from FiinQuantX import FiinSession
# load_dotenv()
# username = os.getenv("UsernameDSTC")
# password = os.getenv("PasswordDSTC")

username = 'DSTC_36@fiinquant.vn'
password = 'Fiinquant0606'

client = FiinSession(
    username=username,
    password=password,
).login()

fi = client.FiinIndicator()

# TF / Keras
import tensorflow as tf
# Configure GPU before any other TF operations
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set GPU as preferred device
        tf.config.experimental.set_device_policy('explicit')
        print(f"GPU configured successfully. Available GPUs: {len(gpus)}")
        
        # Force specific GPU if multiple available
        with tf.device('/GPU:0'):
            tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        print("GPU test successful - using GPU for computation")
        
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
        print("Falling back to CPU")
else:
    print("No GPU detected - using CPU")

# Memory optimization
import psutil
print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Dropout, LSTM,
                                     LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D,
                                     Dense, Embedding, Flatten, Concatenate)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# -------- CONFIG (edit if needed) --------
CSV_PATH = "DSTC_3y.csv"   # relative to current working dir
START_DATE = None        # example: '2010-12-31' or None to keep all
END_DATE = None          # '2022-12-31' or None
RF_ANNUAL = 0.03
MAX_MISSING_RATIO = 0.10
N_RFE_FEATURES = 40
N_CLUSTERS = 6
FINAL_FEATURE_COUNT = 20
WINDOW = 40
STEP = 1
RFE_ESTIMATORS = 100     # smaller for quick runs
SOM_ITERS = 200
EPOCHS = 100              # short for smoke test; increase when confident
BATCH_SIZE = 64
EMB_DIM = 16
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Triple-Barrier Method settings
TAKE_PROFIT_MULT = 2   # Rào cản chốt lời = 3 * ATR (R:R ~ 1:2)
STOP_LOSS_MULT = 1   # Rào cản cắt lỗ = 1 * ATR
MAX_HOLD_PERIOD = 7   # Giữ vị thế tối đa 7 ngày

CHUNK_SIZE = 10000       # Process data in chunks
MAX_SAMPLES = 50000      # Limit total samples to prevent OOM

# -------- Helpers --------

def check_memory():
    """Monitor memory usage"""
    mem = psutil.virtual_memory()
    print(f"Memory usage: {mem.percent:.1f}% ({(mem.total - mem.available)/(1024**3):.1f}/{mem.total/(1024**3):.1f} GB)")
    if mem.percent > 85:
        print("WARNING: High memory usage detected!")
        gc.collect()
        return True
    return False

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    
    # Load in chunks to manage memory
    print("Loading data in chunks...")
    chunk_list = []
    
    try:
        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
            if 'timestamp' in chunk.columns:
                chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
            chunk_list.append(chunk)
            
            # Check memory after each chunk
            if check_memory():
                print("Memory pressure detected, reducing chunk size...")
                break
                
        df = pd.concat(chunk_list, ignore_index=True)
        del chunk_list
        gc.collect()
        
        # Limit samples if too many
        if len(df) > MAX_SAMPLES:
            print(f"Limiting dataset from {len(df)} to {MAX_SAMPLES} samples")
            df = df.sample(n=MAX_SAMPLES, random_state=42).sort_values(['ticker', 'timestamp']).reset_index(drop=True)
            
        print(f"Data loaded: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        # Try loading smaller subset
        print("Attempting to load smaller subset...")
        df = pd.read_csv(path, nrows=MAX_SAMPLES//2)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

def drop_high_missing(df, max_ratio):
    missing = df.isna().mean()
    drop_cols = missing[missing > max_ratio].index.tolist()
    if drop_cols:
        print(f"Dropping {len(drop_cols)} columns with >{max_ratio*100:.1f}% missing")
        df = df.drop(columns=drop_cols)
    return df


def interp_per_ticker(df, numeric_cols):
    # interpolate per ticker to avoid leaking across tickers
    for t, g in df.groupby('ticker'):
        idx = g.index
        df.loc[idx, numeric_cols] = g[numeric_cols].interpolate(method='linear', limit_direction='both').ffill().bfill().values
    return df


def apply_triple_barrier(df, tp_mult=1.5, sl_mult=1.5, max_period=10):
    """
    Applies the Triple-Barrier Method to generate labels for each ticker.

    Args:
        df (pd.DataFrame): DataFrame containing price data and 'atr_14'.
        tp_mult (float): Multiplier for ATR to set the take-profit barrier.
        sl_mult (float): Multiplier for ATR to set the stop-loss barrier.
        max_period (int): Maximum number of days to hold the position.

    Returns:
        pd.Series: A series with labels (1 for win, 0 for loss, 2 for timeout).
    """
    print(f"Applying Triple-Barrier with TP={tp_mult}*ATR, SL={sl_mult}*ATR, Max Hold={max_period} days...")
    
    # Create an empty Series to store the results, ensuring it can handle NaNs before filling
    all_labels = pd.Series(np.nan, index=df.index)

    # Process each ticker individually
    for ticker, group in df.groupby('ticker'):
        group = group.sort_index() # Ensure data is sorted by time
        labels = pd.Series(np.nan, index=group.index)
        
        # Iterate through each time point to start a "trade"
        for i in range(len(group) - 1):
            entry_idx = group.index[i]
            entry_price = group.loc[entry_idx, 'close']
            atr = group.loc[entry_idx, 'atr_14']

            # Skip if ATR is missing or zero
            if pd.isna(atr) or atr == 0:
                continue

            # 1. Define the barriers
            upper_barrier = entry_price + (atr * tp_mult)
            lower_barrier = entry_price - (atr * sl_mult)

            # 2. Define the time window
            window_end_idx = min(i + max_period, len(group) - 1)
            window_indices = group.index[i+1 : window_end_idx+1]
            price_window = group.loc[window_indices]

            # 3. Find the time when barriers are hit
            hit_tp = price_window[price_window['high'] >= upper_barrier]
            hit_sl = price_window[price_window['low'] <= lower_barrier]

            time_to_tp = hit_tp.index.min() if not hit_tp.empty else pd.NaT
            time_to_sl = hit_sl.index.min() if not hit_sl.empty else pd.NaT
            
            # 4. Determine the label
            # Case 1: Both barriers are hit
            if pd.notna(time_to_tp) and pd.notna(time_to_sl):
                # Choose the one that was hit first
                if time_to_tp < time_to_sl:
                    labels.loc[entry_idx] = 1 # Win
                else:
                    labels.loc[entry_idx] = 0 # Loss
            # Case 2: Only Take Profit is hit
            elif pd.notna(time_to_tp):
                labels.loc[entry_idx] = 1 # Win
            # Case 3: Only Stop Loss is hit
            elif pd.notna(time_to_sl):
                labels.loc[entry_idx] = 0 # Loss
            # Case 4: No barrier is hit (Timeout)
            else:
                labels.loc[entry_idx] = 2 # Timeout

        all_labels.update(labels)
    
    return all_labels


def time_split_train_mask(df, train_frac=0.70):
    # compute global train end date by unique sorted dates
    if 'timestamp' not in df.columns:
        raise RuntimeError('timestamp column required for time split')
    unique_days = np.sort(df['timestamp'].dt.floor('D').unique())
    if len(unique_days) < 3:
        raise RuntimeError('Too few unique days to split')
    train_end = unique_days[int(len(unique_days)*train_frac)-1]
    mask = df['timestamp'].dt.floor('D') <= train_end
    print(f"Time-split train_end date: {train_end} -> train rows: {mask.sum()} / {len(df)}")
    return mask


def train_only_rfe(df, numeric_cols, train_mask, n_select=64, n_estimators=100):
    X = df.loc[train_mask, numeric_cols].fillna(0.0).values
    y = df.loc[train_mask, 'target'].values
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    n_select = min(n_select, X.shape[1])
    rfe = RFE(estimator=rf, n_features_to_select=n_select, step=1)
    print("Fitting RFE on train rows... (this may take time)")
    rfe.fit(X, y)
    feature_ranking = sorted(zip(rfe.ranking_, numeric_cols))
    print("\nFeature Ranking by RFE:")
    for rank, name in feature_ranking:
        print(f"Rank {rank}: {name}")
    keep = [c for c, s in zip(numeric_cols, rfe.support_) if s]
    return keep, rfe


def kmeans_cluster_features(X_train_selected, n_clusters=5):
    # X_train_selected: DataFrame of selected features (train rows only)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train_selected.values)
    # cluster across features -> transpose
    k = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = k.fit_predict(Xs.T)
    return labels, scaler, k


def som_select_per_cluster(X_train_selected, labels_cluster, feature_names, target_count=10):
    # X_train_selected: DataFrame (train_rows x features)
    chosen = []
    features = np.array(feature_names)
    for cl in np.unique(labels_cluster):
        idxs = np.where(labels_cluster == cl)[0]
        if len(idxs) == 0:
            continue
        feats = features[idxs]
        cluster_data = X_train_selected[feats].values  # (n_train_rows, n_feats)
        # transpose -> (n_feats, n_train_rows)
        v = cluster_data.T
        # normalize per feature
        eps = 1e-8
        v = (v - v.mean(axis=1, keepdims=True)) / (v.std(axis=1, keepdims=True) + eps)
        n_feats = v.shape[0]
        som_x = max(1, int(math.sqrt(n_feats)))
        som_y = som_x
        if HAS_SOM:
            som = MiniSom(som_x, som_y, v.shape[1], sigma=0.5, learning_rate=0.5)
            som.random_weights_init(v)
            som.train_random(v, SOM_ITERS)
            mapped = [som.winner(v_i) for v_i in v]
            # count nodes
            node_counts = Counter(mapped)
            # sort nodes by density desc
            nodes_sorted = [n for n, _ in node_counts.most_common()]
            picks = []
            for node in nodes_sorted:
                # features mapped to this node
                idxs_node = [i for i, m in enumerate(mapped) if m == node]
                for ii in idxs_node:
                    f = feats[ii]
                    if f not in picks:
                        picks.append(f)
                    if len(picks) >= max(1, int(len(feats) * target_count / max(1, len(feature_names)))):
                        break
                if len(picks) >= max(1, int(len(feats) * target_count / max(1, len(feature_names)))):
                    break
            if len(picks) == 0:
                # fallback choose top variance
                var_idx = np.argsort(-np.var(v, axis=1))[:1]
                picks = [feats[i] for i in var_idx]
        else:
            # fallback: pick features with highest avg abs corr to others
            if v.shape[0] == 1:
                picks = [feats[0]]
            else:
                corr = np.corrcoef(v)
                avg_abs = np.nanmean(np.abs(corr), axis=1)
                order = np.argsort(-avg_abs)
                n_pick = max(1, int(len(feats) * target_count / max(1, len(feature_names))))
                picks = [feats[i] for i in order[:n_pick]]
        chosen.extend(picks)
    # dedupe preserving order
    seen = set()
    final = [x for x in chosen if not (x in seen or seen.add(x))]
    return final[:target_count]


def correlation_filter(df_full, features, threshold=0.8):
    sub = df_full[features].corr().abs()
    to_drop = set()
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            if sub.iloc[i,j] > threshold:
                # drop j (later) to keep earlier
                to_drop.add(features[j])
    filtered = [f for f in features if f not in to_drop]
    return filtered


def compute_indicators(df):
    """Compute a compact set of technical indicators used by the pipeline.
    Adds columns in-place to df. Implementations are lightweight and avoid external APIs.
    """
    # ensure timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # If a FiinQuantX indicator object exists in globals, prefer it
    use_fi = 'fi' in globals() and globals().get('fi') is not None

    # 1) Moving averages + WMA
    ma_windows = [5, 10, 20, 50]
    for w in ma_windows:
        if use_fi:
            try:
                df[f'ema_{w}'] = fi.ema(df['close'], window=w)
                df[f'sma_{w}'] = fi.sma(df['close'], window=w)
                df[f'wma_{w}'] = fi.wma(df['close'], window=w)
                continue
            except Exception:
                pass
        # pandas group-based EMAs/SMA
        df[f'ema_{w}'] = df.groupby('ticker')['close'].transform(lambda x, p=w: x.ewm(span=p, adjust=False).mean())
        df[f'sma_{w}'] = df.groupby('ticker')['close'].transform(lambda x, p=w: x.rolling(window=p, min_periods=1).mean())
        # WMA implementation
        def _wma(s, p=w):
            weights = np.arange(1, p+1)
            return s.rolling(p, min_periods=1).apply(lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(), raw=True)
        df[f'wma_{w}'] = df.groupby('ticker')['close'].transform(lambda x, p=w: _wma(x, p))

    # RSI
    def _rsi(series, period=14):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.ewm(alpha=1/period, adjust=False).mean()
        ma_down = down.ewm(alpha=1/period, adjust=False).mean()
        rs = ma_up / (ma_down + 1e-8)
        return 100 - (100 / (1 + rs))

    for w in [7, 14, 30]:
        if use_fi:
            try:
                df[f'rsi_{w}'] = fi.rsi(df['close'], window=w)
                continue
            except Exception:
                pass
        df[f'rsi_{w}'] = df.groupby('ticker')['close'].transform(lambda x, p=w: _rsi(x, p))

    # ATR
    def _atr(g, period=14):
        high = g['high']; low = g['low']; close = g['close']
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=1).mean()

    for w in [7, 14, 21]:
        if use_fi:
            try:
                df[f'atr_{w}'] = fi.atr(df['high'], df['low'], df['close'], window=w)
                continue
            except Exception:
                pass
        df[f'atr_{w}'] = df.groupby('ticker').apply(lambda g, p=w: _atr(g, period=p)).reset_index(level=0, drop=True)

    # Bollinger Bands
    bb_windows = [10, 20, 30]
    bb_devs = [1.5, 2.0, 2.5]
    for w in bb_windows:
        for d in bb_devs:
            if use_fi:
                try:
                    df[f'bollinger_hband_{w}_{d}'] = fi.bollinger_hband(df['close'], window=w, window_dev=d)
                    df[f'bollinger_lband_{w}_{d}'] = fi.bollinger_lband(df['close'], window=w, window_dev=d)
                    continue
                except Exception:
                    pass
            m = df.groupby('ticker')['close'].transform(lambda x, p=w: x.rolling(window=p, min_periods=1).mean())
            s = df.groupby('ticker')['close'].transform(lambda x, p=w: x.rolling(window=p, min_periods=1).std())
            df[f'bollinger_hband_{w}_{d}'] = m + d * s
            df[f'bollinger_lband_{w}_{d}'] = m - d * s

    # MACD variants
    def _ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    macd_params = [(12, 26, 9), (5, 35, 5), (20, 50, 10)]
    for fast, slow, signal in macd_params:
        name = f"{fast}_{slow}_{signal}"
        if use_fi:
            try:
                df[f'macd_{name}'] = fi.macd(df['close'], window_fast=fast, window_slow=slow)
                df[f'macd_signal_{name}'] = fi.macd_signal(df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
                df[f'macd_diff_{name}'] = fi.macd_diff(df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
                continue
            except Exception:
                pass
        df[f'macd_{name}'] = df.groupby('ticker')['close'].transform(lambda x, f=fast, s=slow: _ema(x, f) - _ema(x, s))
        df[f'macd_signal_{name}'] = df.groupby('ticker')['macd_' + name].transform(lambda x, p=signal: _ema(x, p))
        df[f'macd_diff_{name}'] = df[f'macd_{name}'] - df[f'macd_signal_{name}']

    # Stochastic Oscillator and signal
    for w in [10, 14, 20]:
        if use_fi:
            try:
                df[f'stoch_{w}'] = fi.stoch(df['high'], df['low'], df['close'], window=w)
                df[f'stoch_signal_{w}'] = fi.stoch_signal(df['high'], df['low'], df['close'], window=w, smooth_window=3)
                continue
            except Exception:
                pass
        k = df.groupby('ticker').apply(lambda g, p=w: (g['close'] - g['low'].rolling(p, min_periods=1).min()) / (g['high'].rolling(p, min_periods=1).max() - g['low'].rolling(p, min_periods=1).min() + 1e-8)).reset_index(level=0, drop=True)
        df[f'stoch_{w}'] = k
        df[f'stoch_signal_{w}'] = df.groupby('ticker')[f'stoch_{w}'].transform(lambda x: x.rolling(3, min_periods=1).mean())

    # ADX and DI
    def _compute_adx(g, period=14):
        high = g['high']; low = g['low']; close = g['close']
        prev_high = high.shift(1); prev_low = low.shift(1); prev_close = close.shift(1)
        plus_dm = (high - prev_high).where((high - prev_high) > (prev_low - low), 0.0)
        minus_dm = (prev_low - low).where((prev_low - low) > (high - prev_high), 0.0)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        plus_di = 100 * (plus_dm.rolling(window=period, min_periods=1).mean() / (atr + 1e-8))
        minus_di = 100 * (minus_dm.rolling(window=period, min_periods=1).mean() / (atr + 1e-8))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
        adx = dx.rolling(window=period, min_periods=1).mean()
        return adx, plus_di, minus_di

    for w in [10, 14, 20]:
        if use_fi:
            try:
                df[f'adx_{w}'] = fi.adx(df['high'], df['low'], df['close'], window=w)
                df[f'adx_pos_{w}'] = fi.adx_pos(df['high'], df['low'], df['close'], window=w)
                df[f'adx_neg_{w}'] = fi.adx_neg(df['high'], df['low'], df['close'], window=w)
                continue
            except Exception:
                pass
        tmp = df.groupby('ticker').apply(lambda g, p=w: _compute_adx(g, period=p)).reset_index(level=0, drop=True)
        # tmp is tuple series; unpack
        df[f'adx_{w}'] = tmp.apply(lambda x: x[0])
        df[f'adx_pos_{w}'] = tmp.apply(lambda x: x[1])
        df[f'adx_neg_{w}'] = tmp.apply(lambda x: x[2])

    # Ichimoku (basic components)
    def _ichimoku(g, p1=9, p2=26, p3=52):
        conv = (g['high'].rolling(p1, min_periods=1).max() + g['low'].rolling(p1, min_periods=1).min()) / 2
        base = (g['high'].rolling(p2, min_periods=1).max() + g['low'].rolling(p2, min_periods=1).min()) / 2
        span_a = ((conv + base) / 2).shift(p2)
        span_b = ((g['high'].rolling(p3, min_periods=1).max() + g['low'].rolling(p3, min_periods=1).min()) / 2).shift(p2)
        return conv, base, span_a, span_b

    def _ichimoku(g, p1=9, p2=26, p3=52):
        conv = (g['high'].rolling(p1, min_periods=1).max() + g['low'].rolling(p1, min_periods=1).min()) / 2
        base = (g['high'].rolling(p2, min_periods=1).max() + g['low'].rolling(p2, min_periods=1).min()) / 2
        span_a = ((conv + base) / 2).shift(p2)
        span_b = ((g['high'].rolling(p3, min_periods=1).max() + g['low'].rolling(p3, min_periods=1).min()) / 2).shift(p2)
        return conv, base, span_a, span_b

    # Prefer vendor functions; if they fail, compute per-group and assign to preserve indices
    if use_fi:
        try:
            df['senkou_span_a'] = fi.ichimoku_a(df['high'], df['low'], df['close'], window1=9, window2=26, window3=52)
            df['senkou_span_b'] = fi.ichimoku_b(df['high'], df['low'], df['close'], window1=9, window2=26, window3=52)
            df['kijun_sen'] = fi.ichimoku_base_line(df['high'], df['low'], df['close'], window1=9, window2=26, window3=52)
            df['tenkan_sen'] = fi.ichimoku_conversion_line(df['high'], df['low'], df['close'], window1=9, window2=26, window3=52)
        except Exception:
            use_fi = False

    if not use_fi:
        # compute per-ticker and assign by index to avoid misalignment
        df['tenkan_sen'] = np.nan
        df['kijun_sen'] = np.nan
        df['senkou_span_a'] = np.nan
        df['senkou_span_b'] = np.nan
        for t, g in df.groupby('ticker'):
            conv, base, span_a, span_b = _ichimoku(g)
            idx = g.index
            df.loc[idx, 'tenkan_sen'] = conv.values
            df.loc[idx, 'kijun_sen'] = base.values
            df.loc[idx, 'senkou_span_a'] = span_a.values
            df.loc[idx, 'senkou_span_b'] = span_b.values
    # Volume indicators
    if use_fi:
        try:
            df['obv'] = fi.obv(df['close'], df['volume'])
        except Exception:
            df['obv'] = df.groupby('ticker').apply(lambda g: (np.sign(g['close'].diff()) * g['volume']).fillna(0).cumsum()).reset_index(level=0, drop=True)
        try:
            df['mfi_14'] = fi.mfi(df['high'], df['low'], df['close'], df['volume'], window=14)
        except Exception:
            # fallback simple MFI approximation
            def _mfi(g, period=14):
                typical = (g['high'] + g['low'] + g['close']) / 3.0
                money = typical * g['volume']
                pos = (typical > typical.shift(1)).astype(int)
                mf_pos = money.where(pos==1, 0).rolling(window=period, min_periods=1).sum()
                mf_neg = money.where(pos==0, 0).rolling(window=period, min_periods=1).sum()
                mfr = mf_pos / (mf_neg + 1e-8)
                return 100 - (100 / (1 + mfr))
            df['mfi_14'] = df.groupby('ticker').apply(lambda g: _mfi(g)).reset_index(level=0, drop=True)
        try:
            df['vwap_14'] = fi.vwap(df['high'], df['low'], df['close'], df['volume'], window=14)
        except Exception:
            df['vwap_14'] = df.groupby('ticker').apply(lambda g: (g['close'] * g['volume']).rolling(window=14, min_periods=1).sum() / (g['volume'].rolling(window=14, min_periods=1).sum() + 1e-8)).reset_index(level=0, drop=True)
    else:
        df['obv'] = df.groupby('ticker').apply(lambda g: (np.sign(g['close'].diff()) * g['volume']).fillna(0).cumsum()).reset_index(level=0, drop=True)
        def _mfi(g, period=14):
            typical = (g['high'] + g['low'] + g['close']) / 3.0
            money = typical * g['volume']
            pos = (typical > typical.shift(1)).astype(int)
            mf_pos = money.where(pos==1, 0).rolling(window=period, min_periods=1).sum()
            mf_neg = money.where(pos==0, 0).rolling(window=period, min_periods=1).sum()
            mfr = mf_pos / (mf_neg + 1e-8)
            return 100 - (100 / (1 + mfr))
        df['mfi_14'] = df.groupby('ticker').apply(lambda g: _mfi(g)).reset_index(level=0, drop=True)
        df['vwap_14'] = df.groupby('ticker').apply(lambda g: (g['close'] * g['volume']).rolling(window=14, min_periods=1).sum() / (g['volume'].rolling(window=14, min_periods=1).sum() + 1e-8)).reset_index(level=0, drop=True)

    # Advanced / vendor-specific indicators: prefer fi.* implementations when available
    adv_names = ['supertrend_14', 'zigzag', 'fvg', 'liquidity']
    for name in adv_names:
        if use_fi:
            try:
                if name == 'supertrend_14':
                    df[name] = fi.supertrend(df['high'], df['low'], df['close'], window=14)
                elif name == 'zigzag':
                    df[name] = fi.zigzag(df['high'], df['low'], dev_threshold=5.0, depth=10)
                elif name == 'fvg':
                    df[name] = fi.fvg(df['open'], df['high'], df['low'], df['close'], join_consecutive=True)
                elif name == 'liquidity':
                    df[name] = fi.liquidity(df['open'], df['high'], df['low'], df['close'])
                continue
            except Exception:
                pass
        # Fallbacks: simple placeholders or NaN
        if name == 'supertrend_14':
            # simple supertrend-like: close - rolling median of ATR
            df[name] = df[f'atr_14'] * 0.0
        else:
            df[name] = np.nan

    # Ensure volume exists
    if 'volume' not in df.columns:
        df['volume'] = 0.0

    # per-ticker stats used for embedding/score
    if 'ticker' in df.columns:
        df['ret'] = df.groupby('ticker')['close'].pct_change()
        ticker_stats = df.groupby('ticker').agg(
            mean_ret=('ret', 'mean'),
            vol_ret=('ret', 'std'),
            pct_pos=('ret', lambda x: (x > 0).mean()),
            avg_vol=('volume', 'mean'),
            count=('ret', 'count')
        ).reset_index()
        ticker_stats['sharpe_like'] = ticker_stats['mean_ret'] / ticker_stats['vol_ret'].replace(0, np.nan).fillna(1e-6)
        _sc = MinMaxScaler()
        ticker_stats['ticker_score'] = _sc.fit_transform(ticker_stats[['sharpe_like']].fillna(0))
        df = df.merge(ticker_stats[['ticker', 'ticker_score']], on='ticker', how='left')
        ticker_to_id = {t: i for i, t in enumerate(sorted(df['ticker'].unique()))}
        df['ticker_id'] = df['ticker'].map(ticker_to_id)

    # --- New Feature Engineering Section ---
    print("Generating advanced features...")
    
    # a. Stationarity Features (Rolling Z-Scores)
    z_score_window = 30
    indicators_to_zscore = ['rsi_14', f'macd_{macd_params[0][0]}_{macd_params[0][1]}_{macd_params[0][2]}', 'stoch_14', 'mfi_14']
    
    # FIX: Ensure columns are numeric before performing rolling calculations
    for indicator in indicators_to_zscore:
        if indicator in df.columns:
            df[indicator] = pd.to_numeric(df[indicator], errors='coerce')

    for indicator in indicators_to_zscore:
        if indicator in df.columns:
            rolling_mean = df.groupby('ticker')[indicator].transform(lambda x: x.rolling(window=z_score_window).mean())
            rolling_std = df.groupby('ticker')[indicator].transform(lambda x: x.rolling(window=z_score_window).std())
            df[f'{indicator}_zscore'] = (df[indicator] - rolling_mean) / (rolling_std + 1e-8)

    # b. Relational Features
    if 'ema_20' in df.columns and 'ema_5' in df.columns:
        df['price_vs_ema20'] = df['close'] / df['ema_20']
        df['ema5_vs_ema20'] = df['ema_5'] / df['ema_20']
    
    if 'bollinger_lband_20_2.0' in df.columns and 'bollinger_hband_20_2.0' in df.columns:
        df['price_in_bb'] = (df['close'] - df['bollinger_lband_20_2.0']) / (df['bollinger_hband_20_2.0'] - df['bollinger_lband_20_2.0'] + 1e-8)

    # c. Market Regime Features
    if 'atr_14' in df.columns:
        df['normalized_atr_14'] = df['atr_14'] / df['close']
    
    if 'ret' in df.columns:
        df['volatility_60d'] = df.groupby('ticker')['ret'].transform(lambda x: x.rolling(60).std())
        
    if 'adx_14' in df.columns:
        df['is_trending'] = (df['adx_14'] > 25).astype(int)
        
    print("Advanced features generated.")
    # --- End of New Feature Engineering Section ---

    # --- Start of Super Advanced Feature Engineering ---
    print("Generating interaction, confirmation, and RoC features...")

    # a. Interaction Features
    if 'rsi_14' in df.columns and 'adx_14' in df.columns:
        df['rsi_x_adx14'] = df['rsi_14'] * df['adx_14']

    if 'volume' in df.columns and 'volatility_60d' in df.columns:
        # Normalize volume before multiplying
        df['volume_norm'] = df.groupby('ticker')['volume'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
        df['vol_x_volatility'] = df['volume_norm'] * df['volatility_60d']

    # b. Confirmation Features
    # Ensure all necessary columns exist before creating the list
    bullish_conditions = []
    if 'ema_50' in df.columns:
        bullish_conditions.append(df['close'] > df['ema_50'])
    if f'macd_diff_12_26_9' in df.columns:
        bullish_conditions.append(df[f'macd_diff_12_26_9'] > 0)
    if 'rsi_14' in df.columns:
        bullish_conditions.append(df['rsi_14'] > 50)
    if 'adx_pos_14' in df.columns and 'adx_neg_14' in df.columns:
        bullish_conditions.append(df['adx_pos_14'] > df['adx_neg_14'])
    
    if bullish_conditions:
        df['bullish_confirmation_score'] = np.sum(bullish_conditions, axis=0)

    # c. Rate of Change (RoC) Features
    if 'ema_5' in df.columns:
        df['ema_5_roc_10'] = df.groupby('ticker')['ema_5'].transform(lambda x: x.pct_change(10))
    if 'rsi_14' in df.columns:
        df['rsi_14_roc_5'] = df.groupby('ticker')['rsi_14'].transform(lambda x: x.pct_change(5))
        
    print("Super advanced features generated.")
    # --- End of Super Advanced Feature Engineering ---
    
    return df


def build_windows_multi_ticker(df, feature_cols, window=30, step=1):
    all_X, all_y, all_tids, all_times, all_open, all_close, all_prev_close = [], [], [], [], [], [], []
    ticker_to_id = {t: i for i, t in enumerate(sorted(df['ticker'].unique()))}
    for ticker, g in df.groupby('ticker'):
        g = g.sort_values('timestamp').reset_index(drop=True)
        data = g[feature_cols].values
        targets = g['target'].values
        opens = g['open'].values
        closes = g['close'].values
        n = len(g)
        if n <= window:
            continue
        Xs, ys, times, opens_s, closes_s, prev_closes = [], [], [], [], [], []
        for start in range(0, n-window, step):
            end = start + window
            Xs.append(data[start:end])
            ys.append(targets[end])
            times.append(g.loc[end, 'timestamp'])
            opens_s.append(opens[end])
            closes_s.append(closes[end])
            prev_closes.append(closes[end-1])
        if len(ys) == 0:
            continue
        all_X.append(np.array(Xs))
        all_y.append(np.array(ys))
        all_tids.append(np.full(len(ys), ticker_to_id[ticker], dtype=int))
        all_times.append(np.array(times))
        all_open.append(np.array(opens_s))
        all_close.append(np.array(closes_s))
        all_prev_close.append(np.array(prev_closes))
    if len(all_X) == 0:
        return None
    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    tid_all = np.concatenate(all_tids)
    times_all = np.concatenate(all_times)
    opens_all = np.concatenate(all_open)
    closes_all = np.concatenate(all_close)
    prev_all = np.concatenate(all_prev_close)
    # sort by times to preserve chronology across tickers
    order = np.argsort(times_all)
    return X_all[order], y_all[order], tid_all[order], times_all[order], opens_all[order], closes_all[order], prev_all[order], ticker_to_id


def build_model(window, n_features, n_tickers, emb_dim=8,
                lstm_units=128, conv_filters=32, att_heads=4, att_key_dim=32, dropout=0.5):
    seq_in = Input(shape=(window, n_features), name='seq')
    tick_in = Input(shape=(), dtype='int32', name='ticker')
    
    emb = Embedding(input_dim=n_tickers, output_dim=emb_dim, name='emb')(tick_in)
    emb = Flatten()(emb)
    
    x = Conv1D(filters=conv_filters, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.001))(seq_in)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout)(x)
    
    x = LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(0.001))(x)
    x = LayerNormalization()(x)
    
    mha = MultiHeadAttention(num_heads=att_heads, key_dim=att_key_dim)(x, x)
    x = LayerNormalization()(x + mha)
    
    gap = GlobalAveragePooling1D()(x)
    
    merged = Concatenate()([gap, emb])
    d = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(merged)
    d = Dropout(dropout)(d)
    
    # THAY ĐỔI LẠI: 1 unit và activation 'sigmoid' cho phân loại nhị phân
    out = Dense(1, activation='sigmoid')(d)
    
    model = Model(inputs=[seq_in, tick_in], outputs=out)
    return model

# -------- Main flow --------

def main():
    # Force GPU context
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        print("Starting pipeline with device:", tf.config.list_physical_devices('GPU')[0] if tf.config.list_physical_devices('GPU') else 'CPU')
        
        print("Loading data...", CSV_PATH)
        df = load_data(CSV_PATH)
        check_memory()
        
        # Reduce data size if needed
        unique_tickers = df['ticker'].nunique()
        if unique_tickers > 20:  # Limit to 20 tickers for demo
            top_tickers = df['ticker'].value_counts().head(20).index
            df = df[df['ticker'].isin(top_tickers)].reset_index(drop=True)
            print(f"Limited to top {len(top_tickers)} tickers")
        
        # compute technical indicators used later in feature selection
        print("Computing indicators...")
        df = compute_indicators(df)
        check_memory()
        
        print("Rows,cols:", df.shape)
        if START_DATE is not None or END_DATE is not None:
            if START_DATE is not None:
                df = df[df['timestamp'] >= pd.to_datetime(START_DATE)]
            if END_DATE is not None:
                df = df[df['timestamp'] <= pd.to_datetime(END_DATE)]
            df = df.reset_index(drop=True)
            print("Filtered date range ->", df['timestamp'].min(), df['timestamp'].max())

        # drop cols with many missing
        df = drop_high_missing(df, MAX_MISSING_RATIO)

        # numeric cols for interpolation/selection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # ensure 'target' etc not in numericcols yet

        # interpolate per ticker
        df = interp_per_ticker(df, numeric_cols)

        # build target using Triple Barrier Method
        # Ensure 'atr_14' is computed in compute_indicators
        df['target'] = apply_triple_barrier(df, 
                                            tp_mult=TAKE_PROFIT_MULT, 
                                            sl_mult=STOP_LOSS_MULT, 
                                            max_period=MAX_HOLD_PERIOD)
        # Drop rows where a label could not be assigned
        df = df.dropna(subset=['target']).copy()
        df['target'] = df['target'].astype(int)

        # ===============================================
        # ======= THÊM DÒNG CODE MỚI VÀO ĐÂY =======
        print(f"Original data size before filtering timeouts: {len(df)}")
        df = df[df['target'] != 2].copy()
        print(f"Data size after filtering timeouts (target=2): {len(df)}")
        # ===============================================

        # time split mask for train-only selection
        train_mask = time_split_train_mask(df, train_frac=0.70)

        # numeric columns used in RFE: exclude target, ret, rf_daily
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['target','ret','rf_daily']]
        print("Numeric cols available for selection:", len(num_cols))

        # train-only RFE
        selected_rfe, rfe_obj = train_only_rfe(df, num_cols, train_mask, n_select=N_RFE_FEATURES, n_estimators=RFE_ESTIMATORS)
        print("RFE selected:", selected_rfe)

        # # KMeans on selected features (train rows only)
        # X_train_sel = df.loc[train_mask, selected_rfe]
        # labels_cluster, scaler_k, km_obj = kmeans_cluster_features(X_train_sel, n_clusters=N_CLUSTERS)

        # # SOM per cluster selection to get ~FINAL_FEATURE_COUNT features
        # selected_final = som_select_per_cluster(X_train_sel, labels_cluster, selected_rfe, target_count=FINAL_FEATURE_COUNT)
        # print("SOM selected (pre-corr):", selected_final)
        
        selected_final = selected_rfe 
        # correlation filter
        final_feats = correlation_filter(df.loc[train_mask], selected_final, threshold=0.8)
        if len(final_feats) > FINAL_FEATURE_COUNT:
            final_feats = final_feats[:FINAL_FEATURE_COUNT]
        print("Final features after corr filtering:", final_feats)

        # ensure we have features
        if len(final_feats) == 0:
            raise RuntimeError("No features selected - check data and parameters")
        n_features = len(final_feats)
                # Chuẩn hóa dữ liệu theo từng Ticker (Per-Ticker Scaling)
        # Chúng ta sẽ fit scaler trên dữ liệu training của mỗi ticker và transform cho toàn bộ dữ liệu của ticker đó.
        print("Performing per-ticker scaling...")
        df_scaled = df.copy() # Tạo bản sao để chứa dữ liệu đã chuẩn hóa
        for ticker, group in df_scaled.groupby('ticker'):
            # Tạo một scaler riêng cho mỗi ticker
            scaler = StandardScaler()
            
            # Xác định dữ liệu training cho ticker này
            train_data_ticker = group.loc[train_mask[df['ticker'] == ticker], final_feats]
            
            # Fit scaler CHỈ trên dữ liệu training của ticker
            if not train_data_ticker.empty:
                scaler.fit(train_data_ticker)
                
                # Áp dụng scaler đã fit để transform toàn bộ dữ liệu của ticker đó
                group_scaled_values = scaler.transform(group[final_feats])
                
                # Gán lại giá trị đã chuẩn hóa vào DataFrame
                df_scaled.loc[group.index, final_feats] = group_scaled_values

        print("Per-ticker scaling complete.")
        # build multi-ticker windows (also returns ticker mapping)
        res = build_windows_multi_ticker(df_scaled, final_feats, window=WINDOW, step=STEP)
        if res is None:
            raise RuntimeError('No windows created (data too short for given window)')
        X_all, y_all, tids_all, times_all, opens_all, closes_all, prev_all, ticker_to_id = res
        n_total = X_all.shape[0]
        print(f"Built {n_total} samples across {len(ticker_to_id)} tickers")

        # split by chronological order
        train_end = int(n_total * 0.70)
        val_end = int(n_total * 0.85)
        X_train, y_train, t_train = X_all[:train_end], y_all[:train_end], tids_all[:train_end]
        X_val, y_val, t_val = X_all[train_end:val_end], y_all[train_end:val_end], tids_all[train_end:val_end]
        X_test, y_test, t_test = X_all[val_end:], y_all[val_end:], tids_all[val_end:]
        times_test = times_all[val_end:] 
        opens_test = opens_all[val_end:]; closes_test = closes_all[val_end:]; prev_test = prev_all[val_end:]
        print("Split shapes", X_train.shape, X_val.shape, X_test.shape)


        # build model
        n_tickers = len(ticker_to_id)
        model = build_model(WINDOW, n_features, n_tickers, emb_dim=EMB_DIM)
        model.compile(optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()

        # Tăng trọng số cho lớp 1 (win) lên, ví dụ gấp 2.5 lần so với lớp 0
        n0 = (y_train == 0).sum()
        n1 = (y_train == 1).sum()
        class_weight_dict = {0: 1.0, 1: 2 * (n0 / n1) if n1 > 0 else 1.0} 
        print('class_weight used in fit (manual adjustment):', class_weight_dict)

        es = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
        ckpt = ModelCheckpoint(os.path.join(OUT_DIR, 'best_model.h5'), monitor='val_loss', mode='min', save_best_only=True)

        # Train with GPU monitoring
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            history = model.fit(
                [X_train, t_train], y_train,
                validation_data=([X_val, t_val], y_val),
                epochs=EPOCHS, 
                batch_size=BATCH_SIZE,
                class_weight=class_weight_dict,
                callbacks=[es, ckpt], 
                verbose=2
            )

        # save training plots
        plt.figure(figsize=(10,4))
        plt.plot(history.history.get('loss', []), label='train_loss')
        plt.plot(history.history.get('val_loss', []), label='val_loss')
        plt.legend(); plt.title('Loss'); plt.savefig(os.path.join(OUT_DIR, 'loss.png'))
        plt.close()

        plt.figure(figsize=(10,4))
        plt.plot(history.history.get('accuracy', []), label='train_accuracy')
        plt.plot(history.history.get('val_accuracy', []), label='val_accuracy')
        plt.legend(); plt.title('Accuracy'); plt.savefig(os.path.join(OUT_DIR, 'accuracy.png'))
        plt.close()

        # predict and evaluate
        y_prob = model.predict([X_test, t_test]).ravel() # .ravel() để làm phẳng array
        y_val_prob = model.predict([X_val, t_val]).ravel()
        
        best_threshold = 0.75 # Đặt thẳng ngưỡng
        print(f"\nUsing fixed high threshold: {best_threshold:.2f}")

        # Áp dụng ngưỡng tối ưu lên tập test
        y_pred = (y_prob >= best_threshold).astype(int)
        
        # ==================================================================
        # ======= BẮT ĐẦU CODE MỚI: TÌM VÀ IN CÁC CỔ PHIẾU TIỀM NĂNG =======

        # Tạo một dictionary để map từ ID về lại tên ticker
        id_to_ticker = {v: k for k, v in ticker_to_id.items()}

        # Tạo một DataFrame để tổng hợp kết quả trên tập test
        results_df = pd.DataFrame({
            'timestamp': times_test,
            'ticker_id': t_test,
            'probability': y_prob, # Xác suất thô từ mô hình
            'prediction': y_pred  # Dự đoán cuối cùng (0 hoặc 1)
        })
        results_df['ticker'] = results_df['ticker_id'].map(id_to_ticker)

        # Lọc ra những tín hiệu được dự đoán là "win" (tiềm năng)
        potential_signals = results_df[results_df['prediction'] == 1].copy()
        potential_signals = potential_signals.sort_values(by='timestamp')

        print("\n" + "="*50)
        print("       tín hiệu mua tiềm năng được phát hiện trên tập test")
        print("="*50)

        if potential_signals.empty:
            print("Không tìm thấy tín hiệu nào.")
        else:
            # In ra kết quả
            print(potential_signals[['timestamp', 'ticker', 'probability']].to_string(index=False))

        print("="*50 + "\n")

        # ======= KẾT THÚC CODE MỚI =======
        # ==================================================================
        
        print("\nClassification Report on Test Set using Optimal Threshold:")

        # HIỂN THỊ LẠI BÁO CÁO CHO 2 LỚP
        print(classification_report(y_test, y_pred, target_names=['loss', 'win']))
        cm = confusion_matrix(y_test, y_pred)
        print('Confusion matrix:\n', cm)
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=['loss', 'win'], yticklabels=['loss', 'win']); plt.title('Confusion Matrix'); plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix.png')); plt.close()

        # per-ticker AUC is not directly applicable for multi-class, skipping
        
        # simple backtests on test set using opens/closes stored earlier
        # Only consider 'win' predictions (label 1) for long positions
        profits_tf = np.where(y_pred == 1, closes_test - prev_test, 0.0)
        profits_ls = np.where(y_pred == 1, closes_test - opens_test, 0.0) # Simplified for long-only on win signal
        print('Trend-following cum profit (long on "win" signal):', np.nansum(profits_tf))
        print('Long-short cum profit (long on "win" signal):', np.nansum(profits_ls))

        model.save(os.path.join(OUT_DIR, 'model_final.h5'))
        print('Done. Outputs saved to', OUT_DIR)


if __name__ == '__main__':
    main()
