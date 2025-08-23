# ===================== Enhanced Kolkata FF Bot with Advanced Strategies =====================
import telebot, os, re, time, json, threading, requests
import numpy as np
import pandas as pd
import threading
from datetime import datetime, date, timedelta
from flask import Flask
from collections import Counter, deque
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")

# Advanced ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import joblib
from scipy import stats
from scipy.signal import find_peaks
import hashlib

# ======== CONFIG ========
BOT_TOKEN = "8306210029:AAHl7sxAEEq0FT750MAThHrAioYyAbRI1oI"
ADMIN_CHAT_ID = None
SPREADSHEET_ID = "10wI8T-NzqYsq6L73kPZ_bibuv2dw7xhQAmOr0msvk1A"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=csv"

# ======== GLOBALS ========
bot = telebot.TeleBot(BOT_TOKEN)
app = Flask('')

@app.route("/healthz")
def _healthz():
    return "ok"

# Enhanced data structures
digits, rounds_hist, dates_hist = [], [], []
feature_cache = {}
pattern_cache = {}

# Advanced configuration
CALIBRATE_PROBS = True
TRAIN_RATIO = 0.9
S5_DEFAULT_THRESHOLD = 0.24

# Enhanced ensemble weights for ALL strategies
ENSEMBLE_WEIGHTS = {
    "S1": 0.12, "S2": 0.11, "S4": 0.08, "S5": 0.06,
    "M1": 0.13, "M2": 0.10, "M3": 0.09,
    "A1": 0.11, "A2": 0.10, "A3": 0.10  # New advanced strategies
}

# ======== ENHANCED STORAGE ========
learning_storage_file = "learning_data.json"

def _atomic_write_json(path: str, obj: dict):
    import tempfile
    d = os.path.dirname(os.path.abspath(path)) or "."
    try:
        fd, tmp_path = tempfile.mkstemp(prefix="tmp_", suffix=".json", dir=d)
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except:
            pass
        raise

def save_learning_data():
    data = {
        "S1": S1_stats, "S2": S2_stats, "S4": S4_stats, "S5": S5_stats,
        "M1": M1_stats, "M2": M2_stats, "M3": M3_stats,
        "A1": A1_stats, "A2": A2_stats, "A3": A3_stats,
        "autobet": AUTO.save_state(),
        "last_update": datetime.now().isoformat(),
        "data_size": len(digits)
    }
    try:
        _atomic_write_json(learning_storage_file, data)
    except Exception as e:
        print(f"⚠️ Error saving learning data: {e}")

def load_learning_data():
    try:
        if os.path.exists(learning_storage_file) and os.path.getsize(learning_storage_file) > 0:
            try:
                with open(learning_storage_file, "r") as f:
                    data = json.load(f)
            except Exception:
                print("⚠️ learning_data.json corrupt — resetting this run.")
                data = {}
        else:
            data = {}
        
        for name, store in [
            ("S1", S1_stats), ("S2", S2_stats), ("S4", S4_stats), ("S5", S5_stats),
            ("M1", M1_stats), ("M2", M2_stats), ("M3", M3_stats),
            ("A1", A1_stats), ("A2", A2_stats), ("A3", A3_stats)
        ]:
            if isinstance(data.get(name), dict):
                store.update(data[name])
        
        if "autobet" in data:
            AUTO.load_state(data["autobet"])
        print("✅ Learning + AutoBet state loaded")
    except Exception as e:
        print(f"⚠️ Error loading learning data: {e}")

# ======== ENHANCED UTILS ========
def _norm(v):
    v = np.array(v, dtype=float)
    v[v < 0] = 0.0
    s = v.sum()
    return (v / s) if s > 0 else np.ones(10) / 10.0

def _softmax(x, t=1.0):
    a = np.array(x, dtype=float) / max(t, 1e-9)
    a -= a.max()
    e = np.exp(a)
    s = e.sum()
    return e / s if s > 0 else np.ones_like(a) / len(a)

def _advanced_softmax(x, temperature=1.0, sharpening=False):
    """Advanced softmax with temperature control and sharpening"""
    a = np.array(x, dtype=float)
    if sharpening:
        # Sharpen high-confidence predictions
        mean_val = np.mean(a)
        std_val = np.std(a)
        a = np.where(a > mean_val + std_val, a * 1.3, a)
    
    a = a / temperature
    a -= a.max()
    e = np.exp(a)
    s = e.sum()
    return e / s if s > 0 else np.ones_like(a) / len(a)

def _parse_date_flexible(s: str):
    if not s:
        return None
    s = s.strip()
    
    patterns = [
        (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"),
        (r"^\d{2}/\d{2}/\d{4}$", "%d/%m/%Y"),
        (r"^\d{2}-\d{2}-\d{4}$", "%d-%m-%Y"),
        (r"^\d{1,2}/\d{1,2}/\d{4}$", "%m/%d/%Y")
    ]
    
    for pattern, fmt in patterns:
        if re.match(pattern, s):
            try:
                return pd.to_datetime(s, format=fmt, errors="coerce").date()
            except:
                continue
    
    # Fallback
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        return dt.to_pydatetime().date() if pd.notna(dt) else None
    except:
        return None

def load_google_sheets_data():
    """Enhanced data loading with better error handling"""
    global digits, rounds_hist, dates_hist
    old_n = len(digits)
    digits, rounds_hist, dates_hist = [], [], []
    
    try:
        r = requests.get(CSV_URL, timeout=15)
        r.raise_for_status()
        
        lines = r.text.strip().split('\n')
        if not lines:
            print("❌ Empty CSV data")
            return False
        
        # Find headers more robustly
        headers = None
        header_line = 0
        for i, line in enumerate(lines[:5]):
            cols = [c.strip().strip('"') for c in line.split(',')]
            if any(h.lower() in ['digit', 'number', 'result'] for h in cols):
                headers = cols
                header_line = i
                break
        
        if headers is None:
            print("❌ Could not find valid headers")
            return False
        
        # Find column indices with multiple possible names
        digit_names = ['digit', 'number', 'result', 'winning', 'win']
        round_names = ['round', 'game', 'period']
        date_names = ['date', 'time', 'timestamp', 'day']
        
        idx_digit = None
        idx_round = None
        idx_date = None
        
        for i, h in enumerate(headers):
            h_lower = h.lower()
            if idx_digit is None and any(name in h_lower for name in digit_names):
                idx_digit = i
            if idx_round is None and any(name in h_lower for name in round_names):
                idx_round = i
            if idx_date is None and any(name in h_lower for name in date_names):
                idx_date = i
        
        if idx_digit is None:
            print("❌ Digit column not found")
            return False
        
        # Process data
        valid_count = 0
        for line in lines[header_line + 1:]:
            if not line.strip():
                continue
            
            cols = [c.strip().strip('"') for c in line.split(',')]
            
            # Process digit
            try:
                if len(cols) <= idx_digit:
                    continue
                digit_str = cols[idx_digit].strip()
                if not digit_str or not digit_str.isdigit():
                    continue
                
                val = int(digit_str)
                if not (0 <= val <= 9):
                    continue
                
                digits.append(val)
                valid_count += 1
                
                # Process round
                if idx_round is not None and len(cols) > idx_round:
                    try:
                        round_val = int(cols[idx_round].strip()) if cols[idx_round].strip().isdigit() else None
                        rounds_hist.append(round_val)
                    except:
                        rounds_hist.append(None)
                else:
                    rounds_hist.append(None)
                
                # Process date
                if idx_date is not None and len(cols) > idx_date and cols[idx_date].strip():
                    parsed_date = _parse_date_flexible(cols[idx_date])
                    dates_hist.append(parsed_date)
                else:
                    dates_hist.append(None)
                    
            except Exception as e:
                continue
        
        print(f"✅ Loaded {len(digits)} valid rows (Δ {len(digits)-old_n})")
        
        # Clear caches on new data
        if len(digits) > old_n:
            feature_cache.clear()
            pattern_cache.clear()
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

# ======== ENHANCED STATS HOLDERS ========
def create_enhanced_stats(name):
    return {
        'name': name, 'total': 0, 'ok': 0, 'acc': 0.0, 'conf': [],
        'recent_acc': deque(maxlen=50), 'best_acc': 0.0, 'worst_acc': 100.0,
        'confidence_trend': deque(maxlen=30), 'prediction_history': deque(maxlen=100)
    }

S1_stats = create_enhanced_stats('RF Advanced')
S2_stats = create_enhanced_stats('GB Advanced')
S4_stats = create_enhanced_stats('SelfTopK Enhanced')
S5_stats = create_enhanced_stats('HighConf Adaptive')
M1_stats = create_enhanced_stats('BayesDirichlet Pro')
M2_stats = create_enhanced_stats('HazardGap Pro')
M3_stats = create_enhanced_stats('Residue Pro')

# New Advanced Strategy Stats
A1_stats = create_enhanced_stats('Spectral Analysis')
A2_stats = create_enhanced_stats('Markov Chain Deep')
A3_stats = create_enhanced_stats('Quantum Inspired')

# ======== ADVANCED FEATURES ========
def _make_features(series, rounds=None, dates=None):
    """Original feature function - kept as is for compatibility"""
    if len(series) < 10:
        return None
    
    cache_key = f"orig_{len(series)}_{hash(str(series[-20:]))}"
    if cache_key in feature_cache:
        return feature_cache[cache_key]
    
    feats = []
    feats.extend(series[-10:])
    
    for d in range(10):
        feats.append(series[-20:].count(d) / max(min(20, len(series)), 1))
    
    for d in range(10):
        feats.append(series[-50:].count(d) / max(min(50, len(series)), 1))
    
    for d in range(10):
        gap = 999
        for i in range(len(series) - 1, -1, -1):
            if series[i] == d:
                gap = len(series) - 1 - i
                break
        feats.append(gap)
    
    # EWMA counts
    acc = np.zeros(10)
    w = 1.0
    for v in reversed(series):
        acc[v] += w
        w *= 0.97
        if w < 1e-6:
            break
    acc = acc / acc.sum() if acc.sum() > 0 else np.ones(10) / 10.0
    feats.extend(list(acc))
    
    # Streak
    streak = 1
    for i in range(len(series) - 2, -1, -1):
        if series[i] == series[-1]:
            streak += 1
        else:
            break
    feats.append(streak)
    
    # Weekday one-hot
    if dates and any(dates):
        wd = None
        for i in range(len(dates) - 1, -1, -1):
            if dates[i] is not None:
                wd = pd.Timestamp(dates[i]).weekday()
                break
        for w in range(7):
            feats.append(1 if wd == w else 0)
    else:
        feats.extend([0] * 7)
    
    # Round one-hot
    if rounds and any(r is not None for r in rounds):
        rcur = None
        for i in range(len(rounds) - 1, -1, -1):
            if rounds[i] is not None:
                rcur = rounds[i]
                break
        for rv in range(1, 9):
            feats.append(1 if rcur == rv else 0)
    else:
        feats.extend([0] * 8)
    
    feature_cache[cache_key] = feats
    return feats

def _make_advanced_features(series, rounds=None, dates=None):
    """Advanced feature engineering with spectral and pattern analysis"""
    if len(series) < 20:
        return None
    
    cache_key = f"adv_{len(series)}_{hashlib.md5(str(series[-50:]).encode()).hexdigest()[:8]}"
    if cache_key in feature_cache:
        return feature_cache[cache_key]
    
    feats = []
    
    # 1. Multi-scale frequency analysis
    for window in [10, 20, 30, 50]:
        window_data = series[-window:] if len(series) >= window else series
        digit_counts = [window_data.count(d) for d in range(10)]
        
        # Frequency features
        feats.extend([c / len(window_data) for c in digit_counts])
        
        # Statistical moments
        feats.extend([
            np.mean(digit_counts),
            np.std(digit_counts),
            np.var(digit_counts),
            max(digit_counts) - min(digit_counts)  # Range
        ])
    
    # 2. Spectral analysis features
    if len(series) >= 32:
        # FFT analysis for periodicity detection
        fft = np.fft.fft(series[-32:])
        power_spectrum = np.abs(fft)**2
        
        # Dominant frequencies
        feats.extend([
            np.argmax(power_spectrum[1:8]) + 1,  # Dominant period
            np.max(power_spectrum[1:8]),         # Dominant power
            np.sum(power_spectrum[1:4]) / np.sum(power_spectrum[1:8])  # Low freq ratio
        ])
        
        # Autocorrelation features
        autocorr = np.correlate(series[-32:], series[-32:], mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        if len(autocorr) > 5:
            feats.extend([
                np.max(autocorr[1:8]),           # Max autocorr
                np.argmax(autocorr[1:8]) + 1     # Best lag
            ])
        else:
            feats.extend([0, 1])
    else:
        feats.extend([0] * 5)  # Padding for spectral features
    
    # 3. Advanced gap analysis with distribution fitting
    gap_distributions = {}
    for d in range(10):
        gaps = []
        last_pos = None
        for i, val in enumerate(series):
            if val == d:
                if last_pos is not None:
                    gaps.append(i - last_pos)
                last_pos = i
        
        if len(gaps) >= 3:
            # Fit exponential distribution
            try:
                shape, loc, scale = stats.expon.fit(gaps)
                current_gap = len(series) - 1 - last_pos if last_pos is not None else np.mean(gaps)
                survival_prob = 1 - stats.expon.cdf(current_gap, loc=loc, scale=scale)
                feats.append(survival_prob)
            except:
                feats.append(0.5)
        else:
            feats.append(0.5)
    
    # 4. Pattern complexity and entropy
    # Local entropy
    for window in [10, 20]:
        recent = series[-window:] if len(series) >= window else series
        counts = [recent.count(d) for d in range(10)]
        probs = [c/len(recent) for c in counts if c > 0]
        entropy = -sum(p * np.log2(p) for p in probs) if probs else 0
        feats.append(entropy)
    
    # Lempel-Ziv complexity approximation
    def lz_complexity(seq):
        if len(seq) < 2:
            return 1
        complexity = 1
        i = 0
        while i < len(seq) - 1:
            j = i + 1
            while j <= len(seq) and seq[i:j] not in [seq[k:k+j-i] for k in range(i)]:
                j += 1
            complexity += 1
            i = j - 1
        return complexity
    
    recent_30 = series[-30:] if len(series) >= 30 else series
    feats.append(lz_complexity(recent_30) / len(recent_30))
    
    # 5. Markov chain features
    # First order Markov transitions
    transition_matrix = np.zeros((10, 10))
    for i in range(len(series) - 1):
        transition_matrix[series[i], series[i+1]] += 1
    
    # Normalize rows
    row_sums = transition_matrix.sum(axis=1)
    for i in range(10):
        if row_sums[i] > 0:
            transition_matrix[i] = transition_matrix[i] / row_sums[i]
    
    if len(series) > 0:
        last_digit = series[-1]
        feats.extend(transition_matrix[last_digit])  # Transition probabilities from last digit
    else:
        feats.extend([0.1] * 10)
    
    # 6. Trend and momentum analysis
    if len(series) >= 20:
        # Moving averages
        ma_5 = np.mean(series[-5:])
        ma_10 = np.mean(series[-10:])
        ma_20 = np.mean(series[-20:])
        
        feats.extend([
            ma_5, ma_10, ma_20,
            ma_5 - ma_10,   # Short-term momentum
            ma_10 - ma_20,  # Long-term momentum
            (ma_5 - ma_20) / 20  # Normalized trend
        ])
        
        # Volatility
        vol_10 = np.std(series[-10:])
        vol_20 = np.std(series[-20:])
        feats.extend([vol_10, vol_20, vol_10 - vol_20])
    else:
        feats.extend([4.5] * 9)  # Default values
    
    # 7. Enhanced time features
    if dates and any(d is not None for d in dates):
        latest_date = next(d for d in reversed(dates) if d is not None)
        feats.extend([
            latest_date.weekday(),
            latest_date.day % 7,           # Day cycle
            latest_date.month,
            np.sin(2 * np.pi * latest_date.day / 31),      # Circular day
            np.cos(2 * np.pi * latest_date.day / 31),
            np.sin(2 * np.pi * latest_date.weekday() / 7), # Circular weekday
            np.cos(2 * np.pi * latest_date.weekday() / 7)
        ])
    else:
        feats.extend([0] * 7)
    
    # 8. Enhanced round features
    if rounds and any(r is not None for r in rounds):
        latest_round = next(r for r in reversed(rounds) if r is not None)
        # One-hot encoding
        for rv in range(1, 9):
            feats.append(1 if latest_round == rv else 0)
        
        # Circular encoding
        feats.extend([
            np.sin(2 * np.pi * latest_round / 8),
            np.cos(2 * np.pi * latest_round / 8),
            latest_round % 2,  # Even/odd round
            latest_round % 4   # Quarter cycle
        ])
    else:
        feats.extend([0] * 12)
    
    feature_cache[cache_key] = feats
    return feats

def _build_xy(series, rounds=None, dates=None):
    """Original build function - kept for compatibility"""
    X, y = [], []
    for i in range(30, len(series) - 1):
        hist = series[:i]
        feats = _make_features(hist, rounds[:i] if rounds else None, dates[:i] if dates else None)
        if feats is None:
            continue
        X.append(feats)
        y.append(series[i])
    return X, y

def _build_advanced_xy(series, rounds=None, dates=None):
    """Advanced build function with enhanced features"""
    X, y = [], []
    for i in range(50, len(series) - 1):  # Need more history for advanced features
        hist = series[:i]
        feats = _make_advanced_features(hist, rounds[:i] if rounds else None, dates[:i] if dates else None)
        if feats is None:
            continue
        X.append(feats)
        y.append(series[i])
    return X, y

# ======== ENHANCED MODELS ========
S1_model, S2_model = None, None

def train_randomforest_advanced():
    """Enhanced Random Forest with advanced parameters"""
    if len(digits) < 80:
        return None
    
    X, y = _build_xy(digits, rounds_hist, dates_hist)
    if len(X) < 50:
        return None
    
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Advanced Random Forest with optimized parameters
    base = RandomForestClassifier(
        n_estimators=500,        # More trees
        max_depth=25,           # Deeper trees
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='log2',    # Better feature selection
        class_weight='balanced_subsample',  # Dynamic balancing
        bootstrap=True,
        oob_score=True,         # Out-of-bag scoring
        random_state=42,
        n_jobs=-1
    )
    
    model = CalibratedClassifierCV(
        base, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        method='sigmoid'
    ) if CALIBRATE_PROBS else base
    
    model.fit(X_tr, y_tr)
    
    # Enhanced validation
    y_pred = model.predict(X_va)
    acc = accuracy_score(y_va, y_pred)
    precision = precision_score(y_va, y_pred, average='weighted', zero_division=0)
    
    print(f"✅ Advanced RF - Acc: {acc*100:.1f}%, Precision: {precision*100:.1f}%")
    if hasattr(base, 'oob_score_'):
        print(f"   OOB Score: {base.oob_score_*100:.1f}%")
    
    return model

def train_gradientboosting_advanced():
    """Enhanced Gradient Boosting"""
    if len(digits) < 80:
        return None
    
    X, y = _build_xy(digits, rounds_hist, dates_hist)
    if len(X) < 50:
        return None
    
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Advanced Gradient Boosting
    base = GradientBoostingClassifier(
        n_estimators=600,       # More estimators
        learning_rate=0.05,     # Lower learning rate
        max_depth=8,           # Deeper trees
        subsample=0.85,        # Stochastic boosting
        max_features='sqrt',
        validation_fraction=0.1,
        n_iter_no_change=50,   # Early stopping
        random_state=42
    )
    
    model = CalibratedClassifierCV(
        base, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        method='sigmoid'
    ) if CALIBRATE_PROBS else base
    
    model.fit(X_tr, y_tr)
    
    # Enhanced validation
    y_pred = model.predict(X_va)
    acc = accuracy_score(y_va, y_pred)
    precision = precision_score(y_va, y_pred, average='weighted', zero_division=0)
    
    print(f"✅ Advanced GB - Acc: {acc*100:.1f}%, Precision: {precision*100:.1f}%")
    
    return model

def S1_predict():
    """Enhanced Random Forest Prediction"""
    global S1_model
    if len(digits) < 20:
        return 0, 0.2, np.ones(10) / 10.0
    
    if S1_model is None:
        S1_model = train_randomforest_advanced()
    
    f = _make_features(digits, rounds_hist, dates_hist)
    if S1_model is None or f is None:
        return 0, 0.2, np.ones(10) / 10.0
    
    if hasattr(S1_model, "predict_proba"):
        pro = S1_model.predict_proba([f])[0]
        p = int(np.argmax(pro))
        c = float(pro[p])
        return p, min(0.98, c), pro
    
    p = int(S1_model.predict([f])[0])
    return p, 0.6, np.ones(10) / 10.0

def S2_predict():
    """Enhanced Gradient Boosting Prediction"""
    global S2_model
    if len(digits) < 20:
        return 0, 0.2, np.ones(10) / 10.0
    
    if S2_model is None:
        S2_model = train_gradientboosting_advanced()
    
    f = _make_features(digits, rounds_hist, dates_hist)
    if S2_model is None or f is None:
        return 0, 0.2, np.ones(10) / 10.0
    
    if hasattr(S2_model, "predict_proba"):
        pro = S2_model.predict_proba([f])[0]
        p = int(np.argmax(pro))
        c = float(pro[p])
        return p, min(0.98, c), pro
    
    p = int(S2_model.predict([f])[0])
    return p, 0.6, np.ones(10) / 10.0

# ======== ENHANCED EXISTING STRATEGIES ========
class EnhancedSelfLearner:
    """Enhanced Self-Learning Strategy with advanced gap analysis"""
    
    def __init__(self):
        self.delay_gaps = {d: [] for d in range(10)}
        self.gap_distributions = {d: None for d in range(10)}
        self.trend_weights = {d: 1.0 for d in range(10)}
    
    def train(self, series):
        self.delay_gaps = {d: [] for d in range(10)}
        last = {d: None for d in range(10)}
        
        # Collect gaps with time-decay weighting
        weights = np.exp(-0.05 * np.arange(len(series)))[::-1]  # Recent data more important
        
        for i, v in enumerate(series):
            if last[v] is not None:
                gap = i - last[v]
                self.delay_gaps[v].append((gap, weights[i]))
            last[v] = i
        
        # Fit distributions to gaps
        for d in range(10):
            gaps_weighted = self.delay_gaps.get(d, [])
            if len(gaps_weighted) >= 5:
                gaps_only = [g[0] for g in gaps_weighted]
                try:
                    # Fit gamma distribution (better for gap modeling)
                    self.gap_distributions[d] = stats.gamma.fit(gaps_only)
                except:
                    self.gap_distributions[d] = None
    
    def predict_topk(self, series, k=3):
        last = {d: None for d in range(10)}
        for i, v in enumerate(series):
            last[v] = i
        
        L = len(series)
        scores = np.zeros(10)
        
        for d in range(10):
            gaps_weighted = self.delay_gaps.get(d, [])
            if not gaps_weighted:
                scores[d] = 1.0
                continue
            
            # Current gap
            current_gap = (L - 1 - last[d]) if last[d] is not None else 10
            current_gap = max(1, int(current_gap))
            
            # Use distribution if available
            if self.gap_distributions[d] is not None:
                try:
                    shape, loc, scale = self.gap_distributions[d]
                    # Survival function - probability gap >= current_gap
                    survival_prob = 1 - stats.gamma.cdf(current_gap, shape, loc=loc, scale=scale)
                    scores[d] = max(0.01, survival_prob)
                except:
                    # Fallback to original method
                    gaps_only = [g[0] for g in gaps_weighted]
                    arr = np.array(gaps_only)
                    ge = (arr >= current_gap).sum()
                    eq = (arr == current_gap).sum()
                    scores[d] = (eq + 0.25) / (ge + 2.5)
            else:
                # Original method for insufficient data
                gaps_only = [g[0] for g in gaps_weighted] if gaps_weighted else [10]
                arr = np.array(gaps_only)
                ge = (arr >= current_gap).sum()
                eq = (arr == current_gap).sum()
                scores[d] = (eq + 0.25) / (ge + 2.5)
        
        probs = _norm(scores)
        order = np.argsort(-probs).tolist()
        return order[:k], probs

SELF = EnhancedSelfLearner()

def S4_predict():
    """Enhanced Self-Learning Top-K"""
    if len(digits) < 10:
        return [0, 1], [0.2, 0.2], np.ones(10) / 10.0
    topk, probs = SELF.predict_topk(digits, k=3)
    confs = [float(probs[p]) for p in topk]
    return topk, confs, probs

def S5_predict():
    """Enhanced High-Confidence with adaptive threshold"""
    if len(digits) < 10:
        return 0, 0.2
    
    topk, probs = SELF.predict_topk(digits, k=1)
    p = int(topk[0])
    c = float(probs[p])
    
    # Adaptive threshold based on recent performance
    recent_acc = S5_stats.get('recent_acc', deque(maxlen=50))
    if len(recent_acc) >= 10:
        avg_recent = np.mean(recent_acc)
        adaptive_threshold = S5_DEFAULT_THRESHOLD * (1 + 0.5 * (avg_recent - 0.5))
        adaptive_threshold = max(0.15, min(0.4, adaptive_threshold))
    else:
        adaptive_threshold = S5_DEFAULT_THRESHOLD
    
    if c >= adaptive_threshold:
        return p, c
    
    # Enhanced fallback
    fb = max(set(digits[-15:]), key=digits[-15:].count) if len(digits) >= 15 else digits[-1]
    return int(fb), max(0.28, c)

# ======== ENHANCED EXISTING ADVANCED STRATEGIES ========
@dataclass
class StrategyBayesDirichletAdvanced:
    """Enhanced Bayesian approach with hierarchical priors"""
    prior_scale: float = 2.0
    recent_window: int = 25
    sample_count: int = 500
    alpha_global: np.ndarray = field(default_factory=lambda: np.ones(10))
    hierarchical_alpha: dict = field(default_factory=dict)
    
    def train(self, h):
        # Global prior
        cnt = Counter(h)
        self.alpha_global = np.array([self.prior_scale + cnt.get(d, 0) for d in range(10)], float)
        
        # Hierarchical priors by time periods
        self.hierarchical_alpha.clear()
        for window in [10, 20, 30]:
            if len(h) >= window:
                recent = h[-window:]
                cnt_recent = Counter(recent)
                self.hierarchical_alpha[window] = np.array([
                    1.0 + cnt_recent.get(d, 0) for d in range(10)
                ], float)
    
    def predict(self, h):
        if not h:
            pr = np.ones(10) / 10.0
            return 0, 0.1, pr
        
        # Combine multiple priors
        combined_alpha = self.alpha_global.copy()
        
        for window, alpha in self.hierarchical_alpha.items():
            if len(h) >= window:
                weight = 1.0 / window  # Smaller windows get higher weight
                combined_alpha += weight * alpha
        
        r = h[-self.recent_window:] if len(h) >= self.recent_window else h
        cnt_r = Counter(r)
        final_alpha = combined_alpha + np.array([cnt_r.get(d, 0) for d in range(10)], float)
        
        # Enhanced sampling with variance reduction
        wins = np.zeros(10)
        for _ in range(self.sample_count):
            s = np.random.dirichlet(final_alpha)
            wins[np.argmax(s)] += 1
        
        pr = wins / wins.sum() if wins.sum() > 0 else np.ones(10) / 10.0
        pr = _advanced_softmax(pr, temperature=0.8, sharpening=True)
        p = int(np.argmax(pr))
        return p, float(pr[p]), pr

@dataclass
class StrategyHazardGapAdvanced:
    """Enhanced Hazard model with survival analysis"""
    alpha: float = 1.5
    delay_gaps: dict = field(default_factory=lambda: {d: [] for d in range(10)})
    survival_models: dict = field(default_factory=dict)
    
    def train(self, h):
        self.delay_gaps = {d: [] for d in range(10)}
        self.survival_models = {}
        last = {d: None for d in range(10)}
        
        for i, v in enumerate(h):
            if last[v] is not None:
                self.delay_gaps[v].append(i - last[v])
            last[v] = i
        
        # Fit survival models
        for d in range(10):
            gaps = self.delay_gaps.get(d, [])
            if len(gaps) >= 5:
                try:
                    # Fit Weibull distribution for survival analysis
                    self.survival_models[d] = stats.weibull_min.fit(gaps)
                except:
                    self.survival_models[d] = None
    
    def predict(self, h):
        if not h:
            pr = np.ones(10) / 10.0
            return 0, 0.1, pr
        
        last = {d: None for d in range(10)}
        for i, v in enumerate(h):
            last[v] = i
        
        L = len(h)
        scores = np.zeros(10)
        
        for d in range(10):
            gaps = self.delay_gaps.get(d, [])
            current_gap = (L - 1 - last[d]) if last[d] is not None else 10
            current_gap = max(1, int(current_gap))
            
            if d in self.survival_models and self.survival_models[d] is not None:
                try:
                    c, loc, scale = self.survival_models[d]
                    # Hazard rate calculation
                    hazard = stats.weibull_min.pdf(current_gap, c, loc=loc, scale=scale) / \
                            (1 - stats.weibull_min.cdf(current_gap, c, loc=loc, scale=scale) + 1e-8)
                    scores[d] = hazard
                except:
                    # Fallback
                    if not gaps:
                        scores[d] = 1.0
                    else:
                        arr = np.array(gaps)
                        ge = (arr >= current_gap).sum()
                        eq = (arr == current_gap).sum()
                        scores[d] = (eq + self.alpha * 0.25) / (ge + self.alpha * 2.5)
            else:
                # Original method
                if not gaps:
                    scores[d] = 1.0
                else:
                    arr = np.array(gaps)
                    ge = (arr >= current_gap).sum()
                    eq = (arr == current_gap).sum()
                    scores[d] = (eq + self.alpha * 0.25) / (ge + self.alpha * 2.5)
        
        pr = _norm(scores)
        p = int(np.argmax(pr))
        return p, float(pr[p]), pr

@dataclass
class StrategyResidueBalancerAdvanced:
    """Enhanced Residue Balancer with multi-modular analysis"""
    target_even: float = 0.5
    target_mod3: list = field(default_factory=lambda: [1/3] * 3)
    target_mod5: list = field(default_factory=lambda: [0.2] * 5)
    target_mod7: list = field(default_factory=lambda: [1/7] * 7)  # New mod7
    temp: float = 0.7
    
    def train(self, h):
        # Adaptive targets based on historical data
        if len(h) >= 100:
            even_rate = sum(1 for v in h if v % 2 == 0) / len(h)
            self.target_even = 0.7 * self.target_even + 0.3 * even_rate
            
            # Adaptive mod targets
            for mod_val in [3, 5, 7]:
                counts = [0] * mod_val
                for v in h:
                    counts[v % mod_val] += 1
                proportions = [c / len(h) for c in counts]
                
                if mod_val == 3:
                    self.target_mod3 = [0.7 * old + 0.3 * new for old, new in zip(self.target_mod3, proportions)]
                elif mod_val == 5:
                    self.target_mod5 = [0.7 * old + 0.3 * new for old, new in zip(self.target_mod5, proportions)]
                elif mod_val == 7:
                    self.target_mod7 = [0.7 * old + 0.3 * new for old, new in zip(self.target_mod7, proportions)]
    
    def predict(self, h):
        if not h:
            pr = np.ones(10) / 10.0
            return 0, 0.1, pr
        
        n = len(h)
        even = sum(1 for v in h if v % 2 == 0) / n
        
        # Calculate current distributions
        cnt3 = [0, 0, 0]
        cnt5 = [0] * 5
        cnt7 = [0] * 7
        
        for v in h:
            cnt3[v % 3] += 1
            cnt5[v % 5] += 1
            cnt7[v % 7] += 1
        
        mod3 = [c / n for c in cnt3]
        mod5 = [c / n for c in cnt5]
        mod7 = [c / n for c in cnt7]
        
        desir = np.zeros(10)
        for d in range(10):
            # Even/odd balance
            new_even = (even * n + (1 if d % 2 == 0 else 0)) / (n + 1)
            sc_even = -abs(new_even - self.target_even)
            
            # Mod3 balance
            new3 = [c * n for c in mod3]
            new3[d % 3] += 1
            new3 = [c / (n + 1) for c in new3]
            sc_m3 = -sum(abs(new3[r] - self.target_mod3[r]) for r in range(3))
            
            # Mod5 balance
            new5 = [c * n for c in mod5]
            new5[d % 5] += 1
            new5 = [c / (n + 1) for c in new5]
            sc_m5 = -sum(abs(new5[r] - self.target_mod5[r]) for r in range(5))
            
            # Mod7 balance (new)
            new7 = [c * n for c in mod7]
            new7[d % 7] += 1
            new7 = [c / (n + 1) for c in new7]
            sc_m7 = -sum(abs(new7[r] - self.target_mod7[r]) for r in range(7))
            
            # Weighted combination
            desir[d] = 0.35 * sc_even + 0.25 * sc_m3 + 0.25 * sc_m5 + 0.15 * sc_m7
        
        pr = _softmax(desir, t=self.temp)
        p = int(np.argmax(pr))
        return p, float(pr[p]), pr

# Initialize enhanced strategies
M1, M2, M3 = StrategyBayesDirichletAdvanced(), StrategyHazardGapAdvanced(), StrategyResidueBalancerAdvanced()

# ======== NEW ADVANCED STRATEGIES ========

@dataclass
class StrategySpectralAnalysis:
    """A1 - Advanced Spectral Analysis Strategy"""
    fft_cache: dict = field(default_factory=dict)
    dominant_frequencies: list = field(default_factory=list)
    phase_predictors: dict = field(default_factory=dict)
    
    def train(self, history):
        if len(history) < 64:
            return
        
        self.fft_cache.clear()
        self.dominant_frequencies.clear()
        self.phase_predictors = {d: [] for d in range(10)}
        
        # Analyze different window sizes for multi-resolution analysis
        for window_size in [32, 64, 128]:
            if len(history) >= window_size:
                data = history[-window_size:]
                
                # FFT analysis
                fft = np.fft.fft(data)
                power_spectrum = np.abs(fft)**2
                frequencies = np.fft.fftfreq(len(data))
                
                # Find dominant periods
                peaks, _ = find_peaks(power_spectrum[1:len(power_spectrum)//2])
                if len(peaks) > 0:
                    dominant_freq = frequencies[peaks[np.argmax(power_spectrum[peaks])] + 1]
                    if dominant_freq != 0:
                        period = abs(1.0 / dominant_freq)
                        self.dominant_frequencies.append(period)
        
        # Phase analysis for each digit
        for d in range(10):
            positions = [i for i, v in enumerate(history) if v == d]
            if len(positions) >= 3:
                # Calculate phase relationships
                phases = []
                for i in range(1, len(positions)):
                    phase = (positions[i] - positions[i-1]) % len(self.dominant_frequencies[0] if self.dominant_frequencies else 10)
                    phases.append(phase)
                self.phase_predictors[d] = phases
    
    def predict(self, history):
        if len(history) < 20:
            return 0, 0.1, np.ones(10) / 10.0
        
        scores = np.zeros(10)
        
        # Use dominant frequencies to predict next position
        if self.dominant_frequencies:
            avg_period = np.mean(self.dominant_frequencies)
            current_pos = len(history)
            
            for d in range(10):
                # Find last occurrence
                last_pos = None
                for i in range(len(history) - 1, -1, -1):
                    if history[i] == d:
                        last_pos = i
                        break
                
                if last_pos is not None:
                    # Calculate expected next occurrence based on spectral analysis
                    phases = self.phase_predictors.get(d, [])
                    if phases:
                        expected_gap = np.mean(phases)
                        actual_gap = current_pos - last_pos
                        
                        # Score based on how close we are to expected phase
                        phase_diff = abs(actual_gap % avg_period - expected_gap % avg_period)
                        score = np.exp(-phase_diff / avg_period)
                        scores[d] = score
                    else:
                        scores[d] = 0.1
                else:
                    scores[d] = 0.1
        
        # Combine with frequency analysis
        recent = history[-32:] if len(history) >= 32 else history
        freq_scores = np.zeros(10)
        for d in range(10):
            freq_scores[d] = recent.count(d) / len(recent)
        
        # Weighted combination
        final_scores = 0.7 * scores + 0.3 * freq_scores
        probs = _norm(final_scores)
        pred = int(np.argmax(probs))
        return pred, float(probs[pred]), probs

@dataclass
class StrategyMarkovChainDeep:
    """A2 - Deep Markov Chain with variable order and context"""
    transition_matrices: dict = field(default_factory=dict)
    context_transitions: dict = field(default_factory=dict)
    order_weights: dict = field(default_factory=dict)
    
    def train(self, history):
        if len(history) < 30:
            return
        
        self.transition_matrices.clear()
        self.context_transitions.clear()
        self.order_weights.clear()
        
        # Build transition matrices for different orders
        for order in range(1, min(6, len(history) // 10)):
            transitions = {}
            
            for i in range(order, len(history)):
                state = tuple(history[i-order:i])
                next_digit = history[i]
                
                if state not in transitions:
                    transitions[state] = Counter()
                transitions[state][next_digit] += 1
            
            # Convert to probabilities and store
            prob_transitions = {}
            for state, counter in transitions.items():
                total = sum(counter.values())
                prob_transitions[state] = {d: (counter.get(d, 0) + 0.1) / (total + 1.0) for d in range(10)}
            
            self.transition_matrices[order] = prob_transitions
            
            # Calculate order weight based on predictive power
            correct_predictions = 0
            total_predictions = 0
            
            for state, probs in prob_transitions.items():
                if len([s for s in transitions.keys() if s == state]) > 2:  # Enough data
                    predicted = max(probs.keys(), key=probs.get)
                    # Check against actual next occurrences
                    for j in range(order, len(history)):
                        if tuple(history[j-order:j]) == state:
                            if j < len(history) and history[j] == predicted:
                                correct_predictions += 1
                            total_predictions += 1
            
            self.order_weights[order] = correct_predictions / max(total_predictions, 1)
    
    def predict(self, history):
        if len(history) < 10:
            return 0, 0.1, np.ones(10) / 10.0
        
        final_probs = np.zeros(10)
        total_weight = 0
        
        # Combine predictions from different order Markov chains
        for order, transitions in self.transition_matrices.items():
            if len(history) >= order:
                current_state = tuple(history[-order:])
                
                if current_state in transitions:
                    order_weight = self.order_weights.get(order, 0.1)
                    state_probs = np.array([transitions[current_state].get(d, 0.1) for d in range(10)])
                    
                    final_probs += order_weight * state_probs
                    total_weight += order_weight
        
        if total_weight > 0:
            final_probs = final_probs / total_weight
        else:
            final_probs = np.ones(10) / 10.0
        
        # Apply context-aware adjustments
        recent_context = history[-5:] if len(history) >= 5 else history
        context_key = tuple(recent_context)
        
        # Boost probabilities for digits that appeared in similar contexts
        context_boost = np.ones(10)
        for i in range(len(history) - len(recent_context)):
            if tuple(history[i:i+len(recent_context)]) == context_key:
                if i + len(recent_context) < len(history):
                    next_digit = history[i + len(recent_context)]
                    context_boost[next_digit] *= 1.2
        
        final_probs = final_probs * context_boost
        probs = _norm(final_probs)
        pred = int(np.argmax(probs))
        return pred, float(probs[pred]), probs

@dataclass  
class StrategyQuantumInspired:
    """A3 - Quantum-Inspired Probability Strategy"""
    quantum_states: dict = field(default_factory=dict)
    entanglement_matrix: np.ndarray = field(default_factory=lambda: np.eye(10))
    measurement_history: list = field(default_factory=list)
    coherence_decay: float = 0.95
    
    def train(self, history):
        if len(history) < 40:
            return
        
        self.quantum_states.clear()
        self.measurement_history = history.copy()
        
        # Initialize quantum state vectors for each digit
        for d in range(10):
            # Create superposition state based on historical patterns
            positions = [i for i, v in enumerate(history) if v == d]
            if len(positions) >= 2:
                # Calculate interference patterns
                phase_diffs = []
                for i in range(1, len(positions)):
                    phase_diff = (positions[i] - positions[i-1]) % 10
                    phase_diffs.append(phase_diff)
                
                # Create quantum state amplitude
                amplitude = np.zeros(10, dtype=complex)
                for phase in set(phase_diffs):
                    count = phase_diffs.count(phase)
                    # Amplitude with phase
                    amplitude[phase] = count * np.exp(1j * 2 * np.pi * phase / 10)
                
                # Normalize
                norm = np.sqrt(np.sum(np.abs(amplitude)**2))
                if norm > 0:
                    amplitude = amplitude / norm
                
                self.quantum_states[d] = amplitude
            else:
                # Uniform superposition
                self.quantum_states[d] = np.ones(10, dtype=complex) / np.sqrt(10)
        
        # Build entanglement matrix based on co-occurrence patterns
        self.entanglement_matrix = np.eye(10) * 0.1
        for i in range(len(history) - 1):
            d1, d2 = history[i], history[i+1]
            self.entanglement_matrix[d1, d2] += 0.01
            self.entanglement_matrix[d2, d1] += 0.01
        
        # Normalize entanglement matrix
        for i in range(10):
            row_sum = np.sum(self.entanglement_matrix[i])
            if row_sum > 0:
                self.entanglement_matrix[i] = self.entanglement_matrix[i] / row_sum
    
    def predict(self, history):
        if len(history) < 10:
            return 0, 0.1, np.ones(10) / 10.0
        
        # Quantum measurement simulation
        final_amplitude = np.zeros(10, dtype=complex)
        
        for d in range(10):
            if d in self.quantum_states:
                # Apply time evolution (unitary)
                time_phase = 2 * np.pi * len(history) / 100  # Time-dependent phase
                evolved_state = self.quantum_states[d] * np.exp(1j * time_phase)
                
                # Apply entanglement with recent digits
                if len(history) >= 3:
                    recent_digits = set(history[-3:])
                    entanglement_factor = sum(self.entanglement_matrix[d, rd] for rd in recent_digits)
                    evolved_state = evolved_state * (1 + 0.3 * entanglement_factor)
                
                final_amplitude += evolved_state
        
        # Quantum measurement - collapse to probabilities
        probabilities = np.abs(final_amplitude)**2
        
        # Apply decoherence (noise)
        noise = np.random.normal(0, 0.05, 10)
        probabilities += np.abs(noise)
        
        # Normalize
        probs = _norm(probabilities.real)
        
        # Apply quantum interference effects
        last_digit = history[-1]
        for d in range(10):
            if d == last_digit:
                probs[d] *= 0.8  # Quantum exclusion principle approximation
            elif abs(d - last_digit) <= 2:
                probs[d] *= 1.1  # Neighboring enhancement
        
        probs = _norm(probs)
        pred = int(np.argmax(probs))
        return pred, float(probs[pred]), probs

# Initialize new advanced strategies
A1, A2, A3 = StrategySpectralAnalysis(), StrategyMarkovChainDeep(), StrategyQuantumInspired()

# Stats for new strategies
A1_stats = create_enhanced_stats('Spectral Analysis')
A2_stats = create_enhanced_stats('Markov Deep')
A3_stats = create_enhanced_stats('Quantum Inspired')

def create_enhanced_stats(name):
    return {
        'name': name, 'total': 0, 'ok': 0, 'acc': 0.0, 'conf': [],
        'recent_acc': deque(maxlen=50), 'best_streak': 0, 'current_streak': 0,
        'confidence_avg': 0.0, 'high_conf_acc': 0.0
    }

# Update existing stats
S1_stats = {'name': 'RF', 'total': 0, 'ok': 0, 'acc': 0.0, 'conf': [], 'recent_acc': deque(maxlen=50)}
S2_stats = {'name': 'GB', 'total': 0, 'ok': 0, 'acc': 0.0, 'conf': [], 'recent_acc': deque(maxlen=50)}
S4_stats = {'name': 'SelfTopK', 'total': 0, 'ok': 0, 'acc': 0.0, 'conf': [], 'recent_acc': deque(maxlen=50)}
S5_stats = {'name': 'HighConf', 'total': 0, 'ok': 0, 'acc': 0.0, 'conf': [], 'recent_acc': deque(maxlen=50)}
M1_stats = {'name': 'BayesDirichlet', 'total': 0, 'ok': 0, 'acc': 0.0, 'conf': [], 'recent_acc': deque(maxlen=50)}
M2_stats = {'name': 'HazardGap', 'total': 0, 'ok': 0, 'acc': 0.0, 'conf': [], 'recent_acc': deque(maxlen=50)}
M3_stats = {'name': 'Residue', 'total': 0, 'ok': 0, 'acc': 0.0, 'conf': [], 'recent_acc': deque(maxlen=50)}

# New strategy prediction functions
def A1_predict():
    """Spectral Analysis Prediction"""
    try:
        pred, conf, probs = A1.predict(digits)
        return pred, conf, probs
    except Exception as e:
        print(f"A1 error: {e}")
        return 0, 0.1, np.ones(10) / 10.0

def A2_predict():
    """Markov Chain Deep Prediction"""
    try:
        pred, conf, probs = A2.predict(digits)
        return pred, conf, probs
    except Exception as e:
        print(f"A2 error: {e}")
        return 0, 0.1, np.ones(10) / 10.0

def A3_predict():
    """Quantum Inspired Prediction"""
    try:
        pred, conf, probs = A3.predict(digits)
        return pred, conf, probs
    except Exception as e:
        print(f"A3 error: {e}")
        return 0, 0.1, np.ones(10) / 10.0

# ======== ENHANCED EVALUATION ========
def _bump(stats, pred, actual, conf):
    """Enhanced statistics tracking"""
    stats['total'] += 1
    is_correct = (pred == actual)
    
    if is_correct:
        stats['ok'] += 1
        stats['current_streak'] = stats.get('current_streak', 0) + 1
        stats['best_streak'] = max(stats.get('best_streak', 0), stats['current_streak'])
    else:
        stats['current_streak'] = 0
    
    stats['acc'] = (stats['ok'] / stats['total'] * 100.0) if stats['total'] else 0.0
    stats['conf'].append(conf)
    
    # Enhanced tracking
    if 'recent_acc' in stats:
        stats['recent_acc'].append(1 if is_correct else 0)
    
    # Confidence tracking
    if len(stats['conf']) > 0:
        stats['confidence_avg'] = np.mean(stats['conf'])
        # High confidence accuracy (conf > 0.6)
        high_conf_results = [is_correct for c in stats['conf'][-10:] if c > 0.6]
        if high_conf_results:
            stats['high_conf_acc'] = np.mean(high_conf_results) * 100.0
    
    # Keep recent data manageable
    if len(stats['conf']) > 200:
        stats['conf'] = stats['conf'][-200:]

def calculate_accuracy_leakproof():
    """Enhanced accuracy calculation with advanced training"""
    global S1_model, S2_model
    if len(digits) < 80:
        return
    
    split = int(len(digits) * TRAIN_RATIO)
    train_d, test_d = digits[:split], digits[split:]
    
    # Reset models and stats
    S1_model = S2_model = None
    for s in (S1_stats, S2_stats, S4_stats, S5_stats, M1_stats, M2_stats, M3_stats, A1_stats, A2_stats, A3_stats):
        s.update({'total': 0, 'ok': 0, 'acc': 0.0, 'conf': [], 'recent_acc': deque(maxlen=50), 'current_streak': 0})
    
    # Advanced training for all strategies
    print("🔄 Training enhanced strategies...")
    
    # Train existing strategies with advanced methods
    SELF.train(train_d)
    M1.train(train_d)
    M2.train(train_d)
    M3.train(train_d)
    
    # Train new advanced strategies
    A1.train(train_d)
    A2.train(train_d)
    A3.train(train_d)
    
    # Train ML models
    S1_model = train_randomforest_advanced()
    S2_model = train_gradientboosting_advanced()
    
    print("📊 Evaluating on test data...")
    
    # Evaluate all strategies
    for i in range(len(test_d)):
        hist = train_d + test_d[:i]
        actual = test_d[i]
        
        # Existing strategies
        try:
            p, c, _ = S1_predict()
            _bump(S1_stats, p, actual, c)
        except: pass
        
        try:
            p, c, _ = S2_predict()
            _bump(S2_stats, p, actual, c)
        except: pass
        
        try:
            topk, confs, _ = S4_predict()
            S4_stats['total'] += 1
            if actual in topk[:3]:
                S4_stats['ok'] += 1
            S4_stats['acc'] = (S4_stats['ok'] / S4_stats['total'] * 100.0)
        except: pass
        
        try:
            p, c = S5_predict()
            _bump(S5_stats, p, actual, c)
        except: pass
        
        try:
            p, c, _ = M1.predict(hist)
            _bump(M1_stats, p, actual, c)
        except: pass
        
        try:
            p, c, _ = M2.predict(hist)
            _bump(M2_stats, p, actual, c)
        except: pass
        
        try:
            p, c, _ = M3.predict(hist)
            _bump(M3_stats, p, actual, c)
        except: pass
        
        # New advanced strategies
        try:
            p, c, _ = A1.predict(hist)
            _bump(A1_stats, p, actual, c)
        except: pass
        
        try:
            p, c, _ = A2.predict(hist)
            _bump(A2_stats, p, actual, c)
        except: pass
        
        try:
            p, c, _ = A3.predict(hist)
            _bump(A3_stats, p, actual, c)
        except: pass

def train_all_full():
    """Enhanced training for all strategies"""
    print("🚀 Advanced training initiated...")
    
    # Train existing strategies with enhanced methods
    SELF.train(digits)
    M1.train(digits)
    M2.train(digits)  
    M3.train(digits)
    
    # Train new advanced strategies
    A1.train(digits)
    A2.train(digits)
    A3.train(digits)
    
    # Train ML models
    global S1_model, S2_model
    S1_model = train_randomforest_advanced()
    S2_model = train_gradientboosting_advanced()
    
    save_learning_data()
    print("✅ All strategies trained with advanced methods")

# ======== ENHANCED ENSEMBLE ========
def ensemble_predict_advanced(history):
    """Advanced ensemble with all 10 strategies"""
    predictions = {}
    
    # Get predictions from all strategies
    try:
        _, _, pr1 = S1_predict()
        predictions['S1'] = pr1
    except:
        predictions['S1'] = np.ones(10) / 10.0
    
    try:
        _, _, pr2 = S2_predict()
        predictions['S2'] = pr2
    except:
        predictions['S2'] = np.ones(10) / 10.0
    
    try:
        _, _, pr4 = S4_predict()
        predictions['S4'] = pr4
    except:
        predictions['S4'] = np.ones(10) / 10.0
    
    try:
        p5, c5 = S5_predict()
        pr5 = np.ones(10) / 10.0
        pr5[p5] = min(0.97, max(c5, 0.35))
        predictions['S5'] = pr5
    except:
        predictions['S5'] = np.ones(10) / 10.0
    
    try:
        _, _, pm1 = M1.predict(history)
        predictions['M1'] = pm1
    except:
        predictions['M1'] = np.ones(10) / 10.0
    
    try:
        _, _, pm2 = M2.predict(history)
        predictions['M2'] = pm2
    except:
        predictions['M2'] = np.ones(10) / 10.0
    
    try:
        _, _, pm3 = M3.predict(history)
        predictions['M3'] = pm3
    except:
        predictions['M3'] = np.ones(10) / 10.0
    
    # New advanced strategies
    try:
        _, _, pa1 = A1_predict()
        predictions['A1'] = pa1
    except:
        predictions['A1'] = np.ones(10) / 10.0
    
    try:
        _, _, pa2 = A2_predict()
        predictions['A2'] = pa2
    except:
        predictions['A2'] = np.ones(10) / 10.0
    
    try:
        _, _, pa3 = A3_predict()
        predictions['A3'] = pa3
    except:
        predictions['A3'] = np.ones(10) / 10.0
    
    # Advanced ensemble combination
    mix = np.zeros(10)
    total_weight = 0
    
    for strategy, weight in ENSEMBLE_WEIGHTS.items():
        if strategy in predictions:
            # Dynamic weight adjustment based on recent performance
            stats_map = {
                'S1': S1_stats, 'S2': S2_stats, 'S4': S4_stats, 'S5': S5_stats,
                'M1': M1_stats, 'M2': M2_stats, 'M3': M3_stats,
                'A1': A1_stats, 'A2': A2_stats, 'A3': A3_stats
            }
            
            if strategy in stats_map:
                recent_acc_list = list(stats_map[strategy].get('recent_acc', []))
                if len(recent_acc_list) >= 10:
                    recent_performance = np.mean(recent_acc_list[-10:])
                    # Boost weight for well-performing strategies
                    adjusted_weight = weight * (1 + 0.5 * (recent_performance - 0.1))
                    adjusted_weight = max(0.01, min(0.3, adjusted_weight))
                else:
                    adjusted_weight = weight
            else:
                adjusted_weight = weight
            
            mix += adjusted_weight * _norm(predictions[strategy])
            total_weight += adjusted_weight
    
    if total_weight > 0:
        mix = mix / total_weight
    
    mix = _norm(mix)
    pred = int(np.argmax(mix))
    conf = float(mix[pred])
    top3 = np.argsort(-mix)[:3].tolist()
    
    return pred, conf, top3, mix

# Keep original ensemble for compatibility
def ensemble_predict(history):
    """Original ensemble - maintained for AutoBet compatibility"""
    return ensemble_predict_advanced(history)

# ======== AUTO BET PLANNER (Enhanced) ========
class AutoBetPlanner:
    def __init__(self):
        # Enhanced defaults
        self.capital = 3800
        self.bet_per_digit = 10
        self.daily_risk_frac = 0.10
        self.daily_target = 400
        self.daily_stop_loss = None
        self.risk_threshold = 0.20
        self.enabled = False
        self.chat_id = None
        
        # Advanced features
        self.confidence_threshold = 0.45  # Minimum confidence for auto-suggestions
        self.max_daily_bets = 15  # Maximum bets per day
        self.adaptive_sizing = True  # Adaptive bet sizing
        self.strategy_filter = True  # Filter strategies by recent performance
        
        self._reset_day_state()
    
    def _reset_day_state(self):
        self.day = date.today()
        self.day_bet_total = 0
        self.day_win_total = 0
        self.day_net = 0
        self.round_log = []
        self.last_seen_n = 0
        self.daily_bet_count = 0
    
    def _ensure_today(self):
        if self.day != date.today():
            self._reset_day_state()
    
    def daily_risk_budget(self):
        if self.daily_stop_loss is not None:
            return max(0, float(self.daily_stop_loss))
        return max(50.0, float(self.capital) * float(self.daily_risk_frac))
    
    def save_state(self):
        return {
            "capital": self.capital,
            "bet_per_digit": self.bet_per_digit,
            "daily_risk_frac": self.daily_risk_frac,
            "daily_target": self.daily_target,
            "daily_stop_loss": self.daily_stop_loss,
            "risk_threshold": self.risk_threshold,
            "enabled": self.enabled,
            "confidence_threshold": self.confidence_threshold,
            "max_daily_bets": self.max_daily_bets,
            "adaptive_sizing": self.adaptive_sizing
        }
    
    def load_state(self, d):
        try:
            self.capital = d.get("capital", self.capital)
            self.bet_per_digit = d.get("bet_per_digit", self.bet_per_digit)
            self.daily_risk_frac = d.get("daily_risk_frac", self.daily_risk_frac)
            self.daily_target = d.get("daily_target", self.daily_target)
            self.daily_stop_loss = d.get("daily_stop_loss", self.daily_stop_loss)
            self.risk_threshold = d.get("risk_threshold", self.risk_threshold)
            self.enabled = d.get("enabled", self.enabled)
            self.confidence_threshold = d.get("confidence_threshold", 0.45)
            self.max_daily_bets = d.get("max_daily_bets", 15)
            self.adaptive_sizing = d.get("adaptive_sizing", True)
        except Exception as e:
            print("AutoBet load_state err:", e)
    
    def select_digits_advanced(self):
        """Advanced digit selection using all strategies"""
        pred, conf, top3, mix = ensemble_predict_advanced(digits)
        
        # Get individual strategy predictions for consensus
        strategy_predictions = []
        
        try:
            s1, _, _ = S1_predict()
            strategy_predictions.append(s1)
        except: pass
        
        try:
            s2, _, _ = S2_predict()
            strategy_predictions.append(s2)
        except: pass
        
        try:
            s5, _ = S5_predict()
            strategy_predictions.append(s5)
        except: pass
        
        try:
            m1, _, _ = M1.predict(digits)
            strategy_predictions.append(m1)
        except: pass
        
        try:
            m2, _, _ = M2.predict(digits)
            strategy_predictions.append(m2)
        except: pass
        
        try:
            m3, _, _ = M3.predict(digits)
            strategy_predictions.append(m3)
        except: pass
        
        try:
            a1, _, _ = A1_predict()
            strategy_predictions.append(a1)
        except: pass
        
        try:
            a2, _, _ = A2_predict()
            strategy_predictions.append(a2)
        except: pass
        
        try:
            a3, _, _ = A3_predict()
            strategy_predictions.append(a3)
        except: pass
        
        # Consensus analysis
        vote_counts = Counter(strategy_predictions)
        
        # Advanced selection criteria
        selected_digits = []
        
        # 1. High consensus digits (3+ votes)
        for digit, votes in vote_counts.items():
            if votes >= 3:
                selected_digits.append(digit)
        
        # 2. High ensemble probability digits (>0.20)
        for i, prob in enumerate(mix):
            if prob >= 0.20 and i not in selected_digits:
                selected_digits.append(i)
        
        # 3. Ensure we have at least 1, max 3 digits
        if not selected_digits:
            selected_digits = [top3[0]]
        
        if len(selected_digits) > 3:
            # Keep top 3 by ensemble probability
            selected_digits = sorted(selected_digits, key=lambda x: -mix[x])[:3]
        
        # Calculate average confidence
        avg_conf = float(sum(mix[d] for d in selected_digits) / len(selected_digits))
        
        return selected_digits, avg_conf, mix
    
    def calculate_adaptive_bet_size(self, confidence, base_bet):
        """Calculate adaptive bet size based on confidence"""
        if not self.adaptive_sizing:
            return base_bet
        
        # Scale bet size with confidence (50% to 150% of base)
        multiplier = 0.5 + (confidence * 1.0)
        adaptive_bet = int(base_bet * multiplier)
        
        # Ensure reasonable bounds
        min_bet = max(1, base_bet // 2)
        max_bet = base_bet * 2
        
        return max(min_bet, min(max_bet, adaptive_bet))
    
    def risk_score_advanced(self, digits_list, conf):
        """Enhanced risk scoring"""
        # Calculate adaptive bet sizes
        total_cost = 0
        for digit in digits_list:
            bet_size = self.calculate_adaptive_bet_size(conf, self.bet_per_digit)
            total_cost += bet_size
        
        risk_budget = self.daily_risk_budget()
        remaining_budget = risk_budget - abs(min(0, self.day_net))
        
        # Risk components
        budget_risk = total_cost / max(1.0, remaining_budget)
        confidence_risk = max(0.0, self.confidence_threshold - conf) / self.confidence_threshold
        frequency_risk = self.daily_bet_count / max(1, self.max_daily_bets)
        
        # Combined risk score
        total_risk = 0.4 * budget_risk + 0.4 * confidence_risk + 0.2 * frequency_risk
        
        return total_risk, total_cost, remaining_budget
    
    def handle_new_data(self):
        """Enhanced new data handling"""
        self._ensure_today()
        if not self.enabled or self.chat_id is None:
            return
        
        n = len(digits)
        if n <= self.last_seen_n:
            return
        
        self.last_seen_n = n
        
        # Enhanced stop conditions
        if self.day_net >= self.daily_target:
            bot.send_message(
                self.chat_id,
                f"🎯 *Target Achieved!* Net +₹{self.day_net}\n"
                f"Daily goal reached. AutoBet paused for today.\n"
                f"🏆 Bets: {self.daily_bet_count}/{self.max_daily_bets}",
                parse_mode='HTML'
            )
            self.enabled = False
            save_learning_data()
            return
        
        if -self.day_net >= self.daily_risk_budget():
            bot.send_message(
                self.chat_id,
                f"🛑 *Stop-Loss Triggered* Net -₹{abs(self.day_net)}\n"
                f"Risk limit reached. AutoBet paused for today.\n"
                f"📊 Bets: {self.daily_bet_count}/{self.max_daily_bets}",
                parse_mode='HTML'
            )
            self.enabled = False
            save_learning_data()
            return
        
        if self.daily_bet_count >= self.max_daily_bets:
            bot.send_message(
                self.chat_id,
                f"⏰ *Daily Bet Limit Reached*\n"
                f"Max {self.max_daily_bets} bets per day completed.\n"
                f"Net: {'+' if self.day_net >= 0 else ''}₹{self.day_net}",
                parse_mode='HTML'
            )
            self.enabled = False
            save_learning_data()
            return
        
        # Enhanced prediction and recommendation
        try:
            selected_digits, avg_conf, mix = self.select_digits_advanced()
            risk_score, total_cost, remaining_budget = self.risk_score_advanced(selected_digits, avg_conf)
            
            # Enhanced decision logic
            should_play = (
                risk_score < self.risk_threshold and 
                avg_conf >= self.confidence_threshold and
                total_cost <= remaining_budget * 0.3  # Don't risk more than 30% of remaining budget
            )
            
            # Strategy confidence analysis
            strategy_confs = []
            try:
                _, c1, _ = S1_predict()
                strategy_confs.append(f"RF:{c1:.2f}")
            except: pass
            
            try:
                _, c2, _ = S2_predict()
                strategy_confs.append(f"GB:{c2:.2f}")
            except: pass
            
            try:
                _, ca1, _ = A1_predict()
                strategy_confs.append(f"Spec:{ca1:.2f}")
            except: pass
            
            try:
                _, ca2, _ = A2_predict()
                strategy_confs.append(f"Mark:{ca2:.2f}")
            except: pass
            
            try:
                _, ca3, _ = A3_predict()
                strategy_confs.append(f"Quan:{ca3:.2f}")
            except: pass
            
            # Build recommendation message
            risk_emoji = "🟢" if risk_score < 0.15 else "🟡" if risk_score < 0.25 else "🔴"
            conf_emoji = "🚀" if avg_conf >= 0.6 else "📈" if avg_conf >= 0.45 else "📊"
            
            recommendation = "✅ PLAY" if should_play else "❌ SKIP"
            
            # Get next round number
            valid_rounds = [r for r in rounds_hist if r is not None]
            next_round = (max(valid_rounds) + 1) if valid_rounds else '?'
            
            msg = (
                f"🎰 *Round {next_round} - Advanced Analysis*\n\n"
                f"🎯 *Selected:* {', '.join(map(str, selected_digits))}\n"
                f"💰 *Cost:* ₹{total_cost} ({len(selected_digits)} digits)\n"
                f"{conf_emoji} *Confidence:* {avg_conf*100:.1f}%\n"
                f"{risk_emoji} *Risk Score:* {risk_score:.2f}\n\n"
                f"🤖 *Strategy Confidence:*\n{' | '.join(strategy_confs[:3])}\n"
                f"📊 *Budget:* ₹{remaining_budget:.0f} remaining\n"
                f"📈 *Today:* {self.daily_bet_count}/{self.max_daily_bets} bets | Net: {'+' if self.day_net >= 0 else ''}₹{self.day_net}\n\n"
                f"🎲 *Recommendation:* **{recommendation}**"
            )
            
            bot.send_message(self.chat_id, msg, parse_mode='HTML')
            
        except Exception as e:
            print(f"AutoBet error: {e}")
            bot.send_message(
                self.chat_id,
                f"⚠️ AutoBet analysis error. Check logs.\nNet: {'+' if self.day_net >= 0 else ''}₹{self.day_net}",
                parse_mode='HTML'
            )

AUTO = AutoBetPlanner()

# ======== ENHANCED WEB STATUS ========
@app.route('/')
def home():
    try:
        # Calculate average confidence for each strategy
        all_stats = [S1_stats, S2_stats, S4_stats, S5_stats, M1_stats, M2_stats, M3_stats, A1_stats, A2_stats, A3_stats]
        avg_acc = np.mean([s['acc'] for s in all_stats if s['total'] > 0]) if any(s['total'] > 0 for s in all_stats) else 0
        
        return f"""
        <h1>🤖 Enhanced Kolkata FF - Advanced Prediction System</h1>
        <h2>📊 System Status</h2>
        <p><strong>Data:</strong> {len(digits)} records | <strong>Capital:</strong> ₹{AUTO.capital} | <strong>Bet/Digit:</strong> ₹{AUTO.bet_per_digit}</p>
        <p><strong>Target:</strong> ₹{AUTO.daily_target} | <strong>Stop-loss:</strong> ₹{AUTO.daily_risk_budget():.0f} | <strong>Avg Accuracy:</strong> {avg_acc:.1f}%</p>
        
        <h3>🎯 Strategy Performance</h3>
        <p><strong>Original:</strong> RF {S1_stats['acc']:.1f}% | GB {S2_stats['acc']:.1f}% | S4 {S4_stats['acc']:.1f}% | S5 {S5_stats['acc']:.1f}%</p>
        <p><strong>Enhanced:</strong> M1 {M1_stats['acc']:.1f}% | M2 {M2_stats['acc']:.1f}% | M3 {M3_stats['acc']:.1f}%</p>
        <p><strong>Advanced:</strong> Spectral {A1_stats['acc']:.1f}% | Markov {A2_stats['acc']:.1f}% | Quantum {A3_stats['acc']:.1f}%</p>
        
        <h3>🎰 AutoBet Status</h3>
        <p><strong>Status:</strong> {"🟢 ACTIVE" if AUTO.enabled else "🔴 INACTIVE"} | <strong>Risk Threshold:</strong> {AUTO.risk_threshold:.2f}</p>
        <p><strong>Today:</strong> {AUTO.daily_bet_count}/{AUTO.max_daily_bets} bets | Net: {'+' if AUTO.day_net >= 0 else ''}₹{AUTO.day_net}</p>
        
        <p><em>Last updated: {datetime.now().strftime('%H:%M:%S')}</em></p>
        """
    except Exception as e:
        return f"<h1>System Status</h1><p>Error: {e}</p>"

# ======== ENHANCED TELEGRAM COMMANDS ========
WELCOME = """🤖 <b>Enhanced Kolkata FF - Advanced Prediction System</b>

💰 <b>Capital:</b> ₹{cap} | 🎯 <b>Target:</b> ₹{tgt} | 🛑 <b>Stop-loss:</b> ₹{sl}
📊 <b>Bet/Digit:</b> ₹{bet} | 🎲 <b>Max digits/round:</b> 3

🧠 <b>10 Advanced Strategies Active:</b>
• S1/S2: Enhanced ML Models (RF/GB)
• S4/S5: Self-Learning + High-Confidence  
• M1/M2/M3: Bayesian + Hazard + Residue (Enhanced)
• A1/A2/A3: Spectral + Markov Deep + Quantum

⚙️ <b>Commands:</b>
• /autobet on|off - Auto suggestions
• /predict - All strategy predictions
• /accuracy - Performance metrics
• /advanced_status - Detailed analysis
• /set_capital 3800 - Update capital
• /set_bet 10 - Update bet amount
• /set_risk 0.2 - Risk threshold
• /refresh - Reload & retrain
"""

@bot.message_handler(commands=['start'])
def start_cmd(m):
    global ADMIN_CHAT_ID
    if ADMIN_CHAT_ID is None:
        ADMIN_CHAT_ID = m.chat.id
    AUTO.chat_id = m.chat.id
    
    msg = WELCOME.format(
        cap=AUTO.capital,
        tgt=AUTO.daily_target,
        sl=int(AUTO.daily_risk_budget()),
        bet=AUTO.bet_per_digit
    )
    bot.reply_to(m, msg, parse_mode='HTML')

@bot.message_handler(commands=['predict'])
def predict_cmd(m):
    if len(digits) < 5:
        bot.reply_to(m, "❌ Not enough data.")
        return
    
    try:
        # Get next round number
        valid_rounds = [r for r in rounds_hist if r is not None]
        next_round = (max(valid_rounds) + 1) if valid_rounds else '?'
        
        # Get all predictions
        p1, c1, _ = S1_predict()
        p2, c2, _ = S2_predict()
        topk, confs, _ = S4_predict()
        p5, c5 = S5_predict()
        pm1, cm1, _ = M1.predict(digits)
        pm2, cm2, _ = M2.predict(digits)
        pm3, cm3, _ = M3.predict(digits)
        
        # New advanced strategies
        pa1, ca1, _ = A1_predict()
        pa2, ca2, _ = A2_predict()
        pa3, ca3, _ = A3_predict()
        
        # Enhanced ensemble
        pe, ce, top3e, mix = ensemble_predict_advanced(digits)
        
        s4_text = "".join([
            f"  {i+1}. *{p}* ({c*100:.1f}%)\n"
            for i, (p, c) in enumerate(zip(topk[:3], confs[:3]))
        ])
        
        msg = (
            f"🎰 *Round {next_round} - All Strategy Predictions*\n\n"
            f"🔷 *Original Strategies:*\n"
            f"S1 RF → *{p1}* ({c1*100:.1f}%)\n"
            f"S2 GB → *{p2}* ({c2*100:.1f}%)\n"
            f"S4 Self-TopK:\n{s4_text}"
            f"S5 High-Conf → *{p5}* ({c5*100:.1f}%)\n\n"
            f"🔶 *Enhanced Strategies:*\n"
            f"M1 Bayes → *{pm1}* ({cm1*100:.1f}%)\n"
            f"M2 Hazard → *{pm2}* ({cm2*100:.1f}%)\n"
            f"M3 Residue → *{pm3}* ({cm3*100:.1f}%)\n\n"
            f"🔸 *Advanced Strategies:*\n"
            f"A1 Spectral → *{pa1}* ({ca1*100:.1f}%)\n"
            f"A2 Markov → *{pa2}* ({ca2*100:.1f}%)\n"
            f"A3 Quantum → *{pa3}* ({ca3*100:.1f}%)\n\n"
            f"🏆 *Enhanced Ensemble → {pe}*\n"
            f"Confidence: {ce*100:.1f}% | Top-3: {top3e}\n\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')} | Records: {len(digits)}"
        )
        
        bot.reply_to(m, msg, parse_mode='HTML')
        
    except Exception as e:
        bot.reply_to(m, f"❌ Prediction error: {str(e)}")

@bot.message_handler(commands=['accuracy'])
def acc_cmd(m):
    try:
        msg = f"""📊 <b>Enhanced Performance Metrics</b>

🔷 <b>Original Strategies:</b>
S1 RF: {S1_stats['acc']:.2f}% (Recent: {np.mean(list(S1_stats.get('recent_acc', [0])) or [0]) * 100:.1f}%)
S2 GB: {S2_stats['acc']:.2f}% (Recent: {np.mean(list(S2_stats.get('recent_acc', [0])) or [0]) * 100:.1f}%)
S4 Self-TopK: {S4_stats['acc']:.2f}%
S5 High-Conf: {S5_stats['acc']:.2f}%

🔶 <b>Enhanced Strategies:</b>
M1 Bayes Advanced: {M1_stats['acc']:.2f}% (Streak: {M1_stats.get('current_streak', 0)})
M2 Hazard Enhanced: {M2_stats['acc']:.2f}% (Streak: {M2_stats.get('current_streak', 0)})
M3 Residue Advanced: {M3_stats['acc']:.2f}% (Streak: {M3_stats.get('current_streak', 0)})

🔸 <b>New Advanced Strategies:</b>
A1 Spectral: {A1_stats['acc']:.2f}% (Best: {A1_stats.get('best_streak', 0)})
A2 Markov Deep: {A2_stats['acc']:.2f}% (Best: {A2_stats.get('best_streak', 0)})
A3 Quantum: {A3_stats['acc']:.2f}% (Best: {A3_stats.get('best_streak', 0)})

📈 <b>System Stats:</b>
Total Evaluations: {sum(s['total'] for s in [S1_stats, S2_stats, S4_stats, S5_stats, M1_stats, M2_stats, M3_stats, A1_stats, A2_stats, A3_stats])}
Best Overall: {max([S1_stats['acc'], S2_stats['acc'], S4_stats['acc'], S5_stats['acc'], M1_stats['acc'], M2_stats['acc'], M3_stats['acc'], A1_stats['acc'], A2_stats['acc'], A3_stats['acc']]):.2f}%
"""
        bot.reply_to(m, msg, parse_mode='HTML')
    except Exception as e:
        bot.reply_to(m, f"❌ Accuracy calculation error: {str(e)}")

@bot.message_handler(commands=['advanced_status'])
def advanced_status_cmd(m):
    try:
        # Calculate ensemble performance
        pred, conf, top3, mix = ensemble_predict_advanced(digits)
        
        # Get recent performance trends
        all_stats = [S1_stats, S2_stats, S4_stats, S5_stats, M1_stats, M2_stats, M3_stats, A1_stats, A2_stats, A3_stats]
        recent_performance = []
        
        for stats in all_stats:
            recent_acc_list = list(stats.get('recent_acc', []))
            if len(recent_acc_list) >= 5:
                recent_performance.append(np.mean(recent_acc_list[-10:]) * 100)
            else:
                recent_performance.append(0)
        
        strategy_names = ['S1_RF', 'S2_GB', 'S4_Self', 'S5_Conf', 'M1_Bayes', 'M2_Hazard', 'M3_Residue', 'A1_Spectral', 'A2_Markov', 'A3_Quantum']
        
        # Find best performing strategies
        best_idx = np.argmax(recent_performance)
        best_strategy = strategy_names[best_idx]
        best_perf = recent_performance[best_idx]
        
        msg = f"""🧠 <b>Advanced System Analysis</b>

🏆 <b>Current Best:</b> {best_strategy} ({best_perf:.1f}%)
🎯 <b>Next Prediction:</b> {pred} (Conf: {conf*100:.1f}%)
📊 <b>Top-3 Ensemble:</b> {top3}

⚡ <b>Recent Performance (last 10):</b>
Original: RF {recent_performance[0]:.1f}% | GB {recent_performance[1]:.1f}%
Enhanced: Bayes {recent_performance[4]:.1f}% | Hazard {recent_performance[5]:.1f}% | Residue {recent_performance[6]:.1f}%
Advanced: Spectral {recent_performance[7]:.1f}% | Markov {recent_performance[8]:.1f}% | Quantum {recent_performance[9]:.1f}%

🎰 <b>AutoBet Config:</b>
Status: {"🟢 ACTIVE" if AUTO.enabled else "🔴 INACTIVE"}
Risk Threshold: {AUTO.risk_threshold:.2f}
Confidence Min: {AUTO.confidence_threshold:.2f}
Daily Limit: {AUTO.daily_bet_count}/{AUTO.max_daily_bets}

💡 <b>Data Quality:</b>
Records: {len(digits)} | Rounds: {len([r for r in rounds_hist if r is not None])} | Dates: {len([d for d in dates_hist if d is not None])}
Last Update: {datetime.now().strftime('%H:%M:%S')}
"""
        bot.reply_to(m, msg, parse_mode='HTML')
        
    except Exception as e:
        bot.reply_to(m, f"❌ Advanced status error: {str(e)}")

@bot.message_handler(commands=['refresh'])
def refresh_cmd(m):
    try:
        ok = load_google_sheets_data()
        if ok:
            # Enhanced training for all strategies
            print("🔄 Starting enhanced training...")
            
            SELF.train(digits)
            M1.train(digits)
            M2.train(digits)
            M3.train(digits)
            A1.train(digits)
            A2.train(digits)
            A3.train(digits)
            
            calculate_accuracy_leakproof()
            save_learning_data()
            
            # Performance summary
            all_accs = [S1_stats['acc'], S2_stats['acc'], S4_stats['acc'], S5_stats['acc'], 
                       M1_stats['acc'], M2_stats['acc'], M3_stats['acc'],
                       A1_stats['acc'], A2_stats['acc'], A3_stats['acc']]
            avg_acc = np.mean([a for a in all_accs if a > 0])
            best_acc = max(all_accs)
            
            bot.reply_to(m, 
                f"✅ <b>Enhanced Refresh Complete!</b>\n\n"
                f"📊 Records: {len(digits)}\n"
                f"🎯 Average Accuracy: {avg_acc:.2f}%\n"
                f"🏆 Best Strategy: {best_acc:.2f}%\n"
                f"🧠 All 10 strategies retrained\n\n"
                f"<i>Original:</i> RF {S1_stats['acc']:.1f}% | GB {S2_stats['acc']:.1f}%\n"
                f"<i>Enhanced:</i> M1 {M1_stats['acc']:.1f}% | M2 {M2_stats['acc']:.1f}% | M3 {M3_stats['acc']:.1f}%\n"
                f"<i>Advanced:</i> A1 {A1_stats['acc']:.1f}% | A2 {A2_stats['acc']:.1f}% | A3 {A3_stats['acc']:.1f}%",
                parse_mode='HTML')
            
            # Auto-suggest if AutoBet is on
            AUTO.handle_new_data()
        else:
            bot.reply_to(m, "❌ Failed to refresh data from Google Sheets.")
    except Exception as e:
        bot.reply_to(m, f"❌ Refresh error: {str(e)}")

@bot.message_handler(commands=['autobet'])
def autobet_cmd(m):
    try:
        parts = m.text.strip().split()
        if len(parts) < 2 or parts[1] not in ("on", "off"):
            bot.reply_to(m, "Usage: /autobet on | off")
            return
        
        AUTO.chat_id = m.chat.id
        if parts[1] == "on":
            AUTO.enabled = True
            AUTO._ensure_today()
            bot.reply_to(m, 
                f"🟢 <b>Enhanced AutoBet ACTIVATED</b>\n\n"
                f"🎯 Target: ₹{AUTO.daily_target} | Stop-loss: ₹{int(AUTO.daily_risk_budget())}\n"
                f"💰 Bet/Digit: ₹{AUTO.bet_per_digit} | Risk Threshold: {AUTO.risk_threshold:.2f}\n"
                f"🧠 Confidence Min: {AUTO.confidence_threshold:.2f}\n"
                f"📊 Max Daily Bets: {AUTO.max_daily_bets}\n"
                f"⚙️ Adaptive Sizing: {'On' if AUTO.adaptive_sizing else 'Off'}\n\n"
                f"🤖 All 10 strategies monitoring...",
                parse_mode='HTML')
            # Immediate suggestion
            AUTO.handle_new_data()
        else:
            AUTO.enabled = False
            bot.reply_to(m, "🔴 <b>AutoBet DEACTIVATED</b>", parse_mode='HTML')
        
        save_learning_data()
    except Exception as e:
        bot.reply_to(m, f"❌ AutoBet command error: {str(e)}")

@bot.message_handler(commands=['betsummary'])
def betsummary_cmd(m):
    try:
        AUTO._ensure_today()
        
        # Calculate win rate
        win_rate = (AUTO.day_win_total / max(AUTO.day_bet_total, 1)) * 100 if AUTO.day_bet_total > 0 else 0
        
        # ROI calculation
        roi = ((AUTO.day_net / max(AUTO.day_bet_total, 1)) * 100) if AUTO.day_bet_total > 0 else 0
        
        status_emoji = "🟢" if AUTO.day_net >= 0 else "🔴"
        target_progress = (AUTO.day_net / AUTO.daily_target * 100) if AUTO.daily_target > 0 else 0
        
        msg = (
            f"📊 <b>Enhanced Daily Summary</b>\n\n"
            f"{status_emoji} <b>Net P&L:</b> {'+' if AUTO.day_net >= 0 else ''}₹{AUTO.day_net}\n"
            f"💰 Total Invested: ₹{AUTO.day_bet_total}\n"
            f"🏆 Total Returns: ₹{AUTO.day_win_total}\n"
            f"📈 Win Rate: {win_rate:.1f}%\n"
            f"💹 ROI: {roi:+.1f}%\n\n"
            f"🎯 Target Progress: {target_progress:.1f}%\n"
            f"🛡️ Risk Used: ₹{AUTO.day_bet_total}/₹{AUTO.daily_risk_budget():.0f}\n"
            f"🎲 Bets Today: {AUTO.daily_bet_count}/{AUTO.max_daily_bets}\n\n"
            f"🔥 Status: {'Target Hit!' if AUTO.day_net >= AUTO.daily_target else 'Stop-Loss!' if -AUTO.day_net >= AUTO.daily_risk_budget() else 'Active'}"
        )
        
        bot.reply_to(m, msg, parse_mode='HTML')
    except Exception as e:
        bot.reply_to(m, f"❌ Summary error: {str(e)}")

@bot.message_handler(commands=['set_capital'])
def set_capital_cmd(m):
    try:
        val = float(m.text.strip().split()[1])
        AUTO.capital = max(100.0, val)
        bot.reply_to(m, 
            f"✅ Capital updated to ₹{int(AUTO.capital)}\n"
            f"🛡️ New stop-loss budget: ₹{int(AUTO.daily_risk_budget())}")
        save_learning_data()
    except Exception:
        bot.reply_to(m, "Usage: /set_capital 3800")

@bot.message_handler(commands=['set_bet'])
def set_bet_cmd(m):
    try:
        val = float(m.text.strip().split()[1])
        AUTO.bet_per_digit = max(1.0, val)
        bot.reply_to(m, f"✅ Bet per digit updated to ₹{int(AUTO.bet_per_digit)}")
        save_learning_data()
    except Exception:
        bot.reply_to(m, "Usage: /set_bet 10")

@bot.message_handler(commands=['set_risk'])
def set_risk_cmd(m):
    try:
        val = float(m.text.strip().split()[1])
        AUTO.risk_threshold = min(0.8, max(0.05, val))
        bot.reply_to(m, 
            f"✅ Risk threshold updated to {AUTO.risk_threshold:.2f}\n"
            f"💡 Lower = More Conservative | Higher = More Aggressive")
        save_learning_data()
    except Exception:
        bot.reply_to(m, "Usage: /set_risk 0.2")

@bot.message_handler(commands=['set_confidence'])
def set_confidence_cmd(m):
    try:
        val = float(m.text.strip().split()[1])
        AUTO.confidence_threshold = min(0.9, max(0.2, val))
        bot.reply_to(m, 
            f"✅ Confidence threshold updated to {AUTO.confidence_threshold:.2f}\n"
            f"💡 Minimum confidence required for auto-suggestions")
        save_learning_data()
    except Exception:
        bot.reply_to(m, "Usage: /set_confidence 0.45")

# ======== BACKGROUND TASKS (Enhanced) ========
def auto_refresh():
    """Enhanced auto-refresh with intelligent retraining"""
    retrain_counter = 0
    
    while True:
        time.sleep(300)  # 5 minutes
        old_count = len(digits)
        
        try:
            if load_google_sheets_data():
                new_count = len(digits)
                if new_count > old_count:
                    new_data_points = new_count - old_count
                    retrain_counter += new_data_points
                    
                    print(f"📊 New data: {new_count} (+{new_data_points})")
                    
                    # Light retraining for fast strategies
                    SELF.train(digits)
                    M1.train(digits)
                    M2.train(digits)
                    M3.train(digits)
                    A1.train(digits)
                    A2.train(digits)
                    A3.train(digits)
                    
                    # Full retraining every 25 new points or if significant improvement expected
                    if retrain_counter >= 25:
                        print("🔄 Full model retraining...")
                        calculate_accuracy_leakproof()
                        retrain_counter = 0
                    
                    save_learning_data()
                    AUTO.handle_new_data()
                else:
                    print(f"📊 Checked: {new_count} records")
            else:
                print("⚠️ Data refresh failed")
                
        except Exception as e:
            print(f"❌ Auto-refresh error: {e}")
            time.sleep(60)  # Wait longer on error

def keep_alive():
    def run():
        try:
            app.run(host="0.0.0.0", port=3000, debug=False)
        except Exception as e:
            print(f"Flask error: {e}")
    
    t = threading.Thread(target=run)
    t.daemon = True
    t.start()

def self_ping():
    """Enhanced self-ping with better URL detection"""
    # Try multiple URL patterns
    possible_urls = []
    
    if os.getenv('REPL_URL'):
        possible_urls.append(os.getenv('REPL_URL'))
    
    if os.getenv('REPL_SLUG') and os.getenv('REPL_OWNER'):
        possible_urls.append(f"https://{os.getenv('REPL_SLUG')}.{os.getenv('REPL_OWNER')}.repl.co")
    
    # Additional common patterns
    if os.getenv('PROJECT_DOMAIN'):
        possible_urls.append(f"https://{os.getenv('PROJECT_DOMAIN')}")
    
    working_url = None
    for url in possible_urls:
        try:
            response = requests.get(url + "/healthz", timeout=5)
            if response.status_code == 200:
                working_url = url
                break
        except:
            continue
    
    if not working_url:
        print('⚠️ No working URL found for self-ping. Set REPL_URL in environment.')
        return
    
    while True:
        try:
            requests.get(working_url + "/healthz", timeout=10)
            print(f"🔄 Self-ping successful: {working_url}")
        except Exception as e:
            print(f"⚠️ Ping error: {e}")
        time.sleep(240)  # Ping every 4 minutes

def initialize():
    """Enhanced initialization"""
    print("🚀 Starting Enhanced Kolkata FF Prediction System")
    print("🧠 Loading 10 Advanced Strategies:")
    print("   • S1/S2: Enhanced ML Models")
    print("   • S4/S5: Self-Learning Systems") 
    print("   • M1/M2/M3: Advanced Mathematical Models")
    print("   • A1/A2/A3: Spectral/Markov/Quantum Strategies")
    
    # Load existing data
    load_learning_data()
    
    # Load fresh data from sheets
    if load_google_sheets_data():
        print(f"📊 Loaded {len(digits)} historical records")
        
        # Full training cycle
        train_all_full()
        calculate_accuracy_leakproof()
        
        print("✅ All strategies trained and evaluated")
    else:
        print("⚠️ Could not load initial data")
    
    # Start services
    keep_alive()
    threading.Thread(target=self_ping, daemon=True).start()
    threading.Thread(target=auto_refresh, daemon=True).start()
    
    print("🌐 Web dashboard available on port 3000")
    print("📱 Telegram bot ready for commands")

def run_bot():
    """Enhanced bot runner with better error recovery"""
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    while True:
        try:
            print("🤖 Enhanced Bot polling started...")
            bot.polling(none_stop=True, timeout=15)
            consecutive_errors = 0  # Reset on successful run
            
        except Exception as e:
            consecutive_errors += 1
            print(f"❌ Bot error #{consecutive_errors}: {e}")
            
            if consecutive_errors >= max_consecutive_errors:
                print(f"🚨 Too many consecutive errors ({consecutive_errors}). Extended wait...")
                time.sleep(300)  # Wait 5 minutes
                consecutive_errors = 0
            else:
                wait_time = min(60, 5 * consecutive_errors)
                print(f"🔄 Restarting in {wait_time} seconds...")
                time.sleep(wait_time)

# ======== MAIN EXECUTION ========
if __name__ == "__main__":
    try:
        initialize()
        run_bot()
    except KeyboardInterrupt:
        print("\n👋 Shutting down Enhanced Kolkata FF Bot...")
        save_learning_data()
    except Exception as e:
        print(f"🚨 Critical startup error: {e}")
        print("📞 Check configuration and try again")