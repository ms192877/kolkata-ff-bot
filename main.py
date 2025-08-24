# ===================== Enhanced Kolkata FF Bot with Advanced ML Strategies =====================
import telebot, os, re, time, json, threading, requests
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from flask import Flask
from collections import Counter, deque, defaultdict
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")

# Advanced ML imports with error handling
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import joblib
    from scipy import stats
    from scipy.signal import find_peaks
    from scipy.stats import entropy
    ML_AVAILABLE = True
except ImportError as e:
    print("⚠️ ML libraries not available: {}".format(e))
    ML_AVAILABLE = False

import hashlib
import math

# ======== CONFIG ========
BOT_TOKEN = os.getenv("BOT_TOKEN", "8306210029:AAHl7sxAEEq0FT750MAThHrAioYyAbRI1oI")
ADMIN_CHAT_ID = None
SPREADSHEET_ID = "10wI8T-NzqYsq6L73kPZ_bibuv2dw7xhQAmOr0msvk1A"
CSV_URL = "https://docs.google.com/spreadsheets/d/{}/export?format=csv".format(SPREADSHEET_ID)

# ======== GLOBALS ========
bot = telebot.TeleBot(BOT_TOKEN)
app = Flask(__name__)

@app.route("/")
def home():
    return "Enhanced Kolkata FF Bot is running!"

@app.route("/healthz")
def _healthz():
    return "ok"

# Enhanced data structures
digits, rounds_hist, dates_hist = [], [], []
feature_cache = {}
pattern_cache = {}
model_cache = {}

# Advanced ML Configuration
ENSEMBLE_WEIGHTS = {
    "S1": 0.15,  # Advanced Random Forest
    "S2": 0.13,  # Advanced Gradient Boosting
    "S3": 0.12,  # Neural Network
    "S4": 0.10,  # Enhanced Pattern Recognition
    "S5": 0.08,  # Adaptive Confidence
    "M1": 0.12,  # Advanced Bayesian
    "M2": 0.08,  # Markov Chain Enhanced
    "M3": 0.07,  # Spectral Analysis
    "A1": 0.08,  # Deep Sequence Analysis
    "A2": 0.07   # Quantum-Inspired
}

CONFIDENCE_THRESHOLD = 0.3  # Higher threshold for better precision

# ======== ENHANCED STORAGE ========
learning_storage_file = "enhanced_learning_data.json"

def _atomic_write_json(path, obj):
    import tempfile
    d = os.path.dirname(os.path.abspath(path)) or "."
    try:
        fd, tmp_path = tempfile.mkstemp(prefix="tmp_", suffix=".json", dir=d)
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception as e:
        print("⚠️ Write error: {}".format(e))
        try:
            if 'tmp_path' in locals():
                os.remove(tmp_path)
        except:
            pass

# Enhanced stats with more metrics
def create_enhanced_stats(name):
    return {
        'name': name, 'total': 0, 'ok': 0, 'acc': 0.0, 'conf': [],
        'recent_acc': deque(maxlen=100), 'best_acc': 0.0, 'worst_acc': 100.0,
        'confidence_trend': deque(maxlen=50), 'prediction_history': deque(maxlen=200),
        'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'top3_acc': 0.0,
        'entropy_scores': deque(maxlen=30), 'calibration_error': 0.0
    }

# Initialize enhanced stats
S1_stats = create_enhanced_stats('Advanced Random Forest')
S2_stats = create_enhanced_stats('Enhanced Gradient Boosting')
S3_stats = create_enhanced_stats('Neural Network Ensemble')
S4_stats = create_enhanced_stats('Pattern Recognition Pro')
S5_stats = create_enhanced_stats('Adaptive High Confidence')
M1_stats = create_enhanced_stats('Advanced Bayesian Inference')
M2_stats = create_enhanced_stats('Markov Chain Deep Learning')
M3_stats = create_enhanced_stats('Spectral Frequency Analysis')
A1_stats = create_enhanced_stats('Deep Sequence Analyzer')
A2_stats = create_enhanced_stats('Quantum Pattern Matcher')

def save_learning_data():
    try:
        data = {
            "S1": dict(S1_stats), "S2": dict(S2_stats), "S3": dict(S3_stats),
            "S4": dict(S4_stats), "S5": dict(S5_stats), "M1": dict(M1_stats),
            "M2": dict(M2_stats), "M3": dict(M3_stats), "A1": dict(A1_stats), "A2": dict(A2_stats),
            "last_update": datetime.now().isoformat(),
            "data_size": len(digits),
            "model_performance": {name: stats['acc'] for name, stats in [
                ('S1', S1_stats), ('S2', S2_stats), ('S3', S3_stats), ('S4', S4_stats),
                ('S5', S5_stats), ('M1', M1_stats), ('M2', M2_stats), ('M3', M3_stats),
                ('A1', A1_stats), ('A2', A2_stats)
            ]}
        }
        _atomic_write_json(learning_storage_file, data)
        print("✅ Enhanced learning data saved")
    except Exception as e:
        print("⚠️ Error saving learning data: {}".format(e))

def load_learning_data():
    try:
        if os.path.exists(learning_storage_file) and os.path.getsize(learning_storage_file) > 0:
            with open(learning_storage_file, "r") as f:
                data = json.load(f)
        else:
            data = {}
        
        # Load all strategy stats
        for name, store in [
            ("S1", S1_stats), ("S2", S2_stats), ("S3", S3_stats), ("S4", S4_stats),
            ("S5", S5_stats), ("M1", M1_stats), ("M2", M2_stats), ("M3", M3_stats),
            ("A1", A1_stats), ("A2", A2_stats)
        ]:
            if isinstance(data.get(name), dict):
                for key, value in data[name].items():
                    if key in ['recent_acc', 'confidence_trend', 'prediction_history', 'entropy_scores']:
                        store[key] = deque(value, maxlen=store[key].maxlen)
                    else:
                        store[key] = value
        
        print("✅ Enhanced learning state loaded")
        return True
    except Exception as e:
        print("⚠️ Error loading learning data: {}".format(e))
        return False

# ======== ENHANCED FEATURE ENGINEERING ========
def _calculate_entropy(sequence):
    """Calculate Shannon entropy of a sequence"""
    try:
        if not sequence:
            return 0
        counts = Counter(sequence)
        total = len(sequence)
        return -sum((count/total) * math.log2(count/total) for count in counts.values())
    except:
        return 0

def _calculate_autocorr(sequence, max_lag=10):
    """Calculate autocorrelation features"""
    try:
        if len(sequence) < max_lag * 2:
            return [0] * max_lag
        
        seq = np.array(sequence)
        autocorrs = []
        for lag in range(1, max_lag + 1):
            if len(seq) > lag:
                corr = np.corrcoef(seq[:-lag], seq[lag:])[0, 1]
                autocorrs.append(0 if np.isnan(corr) else corr)
            else:
                autocorrs.append(0)
        return autocorrs
    except:
        return [0] * max_lag

def _fourier_features(sequence, n_components=5):
    """Extract Fourier transform features"""
    try:
        if len(sequence) < n_components * 2:
            return [0] * n_components * 2
        
        fft = np.fft.fft(sequence)
        features = []
        for i in range(1, n_components + 1):
            if i < len(fft):
                features.append(abs(fft[i]))
                features.append(np.angle(fft[i]))
            else:
                features.extend([0, 0])
        return features
    except:
        return [0] * n_components * 2

def _trend_features(sequence):
    """Calculate trend and momentum features"""
    try:
        if len(sequence) < 10:
            return [0, 0, 0, 0]
        
        recent_10 = sequence[-10:]
        recent_20 = sequence[-20:] if len(sequence) >= 20 else sequence
        
        # Linear trend
        x = np.arange(len(recent_10))
        slope, _, r_value, _, _ = stats.linregress(x, recent_10)
        
        # Momentum
        momentum = np.mean(recent_10) - np.mean(recent_20)
        
        # Volatility
        volatility = np.std(recent_20)
        
        # Mean reversion indicator
        mean_20 = np.mean(recent_20)
        current_dev = abs(sequence[-1] - mean_20) / (volatility + 1e-6)
        
        return [slope, r_value**2, momentum, current_dev]
    except:
        return [0, 0, 0, 0]

def _pattern_features(sequence):
    """Advanced pattern recognition features"""
    try:
        if len(sequence) < 20:
            return [0] * 15
        
        features = []
        
        # N-gram patterns
        for n in [2, 3]:
            ngrams = {}
            for i in range(len(sequence) - n + 1):
                pattern = tuple(sequence[i:i+n])
                ngrams[pattern] = ngrams.get(pattern, 0) + 1
            
            # Most common pattern frequency
            if ngrams:
                max_freq = max(ngrams.values())
                features.append(max_freq / len(sequence))
            else:
                features.append(0)
            
            # Pattern diversity (number of unique patterns)
            features.append(len(ngrams) / max(1, len(sequence) - n + 1))
        
        # Cyclical patterns
        for period in [5, 7, 10]:
            if len(sequence) >= period * 2:
                cycles = []
                for start in range(0, len(sequence) - period, period):
                    cycles.append(sequence[start:start + period])
                
                if len(cycles) >= 2:
                    # Cycle consistency
                    cycle_corrs = []
                    for i in range(len(cycles) - 1):
                        corr = np.corrcoef(cycles[i], cycles[i+1])[0, 1]
                        if not np.isnan(corr):
                            cycle_corrs.append(corr)
                    
                    features.append(np.mean(cycle_corrs) if cycle_corrs else 0)
                else:
                    features.append(0)
            else:
                features.append(0)
        
        # Digit transition probabilities
        transition_entropy = 0
        transitions = defaultdict(int)
        for i in range(len(sequence) - 1):
            transitions[(sequence[i], sequence[i+1])] += 1
        
        if transitions:
            total_transitions = sum(transitions.values())
            probs = [count / total_transitions for count in transitions.values()]
            transition_entropy = entropy(probs)
        
        features.append(transition_entropy)
        
        # Gap analysis enhanced
        for target in [0, 5, 9]:  # Analyze specific digits
            gaps = []
            last_pos = None
            for i, val in enumerate(sequence):
                if val == target:
                    if last_pos is not None:
                        gaps.append(i - last_pos)
                    last_pos = i
            
            if gaps:
                features.append(np.mean(gaps))
                features.append(np.std(gaps))
            else:
                features.extend([0, 0])
        
        return features
    except:
        return [0] * 15

def _make_enhanced_features(series, rounds=None, dates=None):
    """Enhanced feature extraction with advanced techniques"""
    try:
        if len(series) < 30:
            return None
        
        cache_key = "enhanced_{}_{}".format(len(series), hash(str(series[-30:])))
        if cache_key in feature_cache:
            return feature_cache[cache_key]
        
        features = []
        
        # Basic features (last N digits)
        for n in [5, 10, 15]:
            if len(series) >= n:
                features.extend(series[-n:])
            else:
                features.extend(series + [0] * (n - len(series)))
        
        # Enhanced frequency analysis with multiple windows
        for window_size in [20, 50, 100]:
            window = series[-window_size:] if len(series) >= window_size else series
            for d in range(10):
                count = window.count(d)
                freq = count / len(window)
                features.append(freq)
        
        # Statistical moments
        recent_50 = series[-50:] if len(series) >= 50 else series
        features.extend([
            np.mean(recent_50),
            np.std(recent_50),
            stats.skew(recent_50),
            stats.kurtosis(recent_50)
        ])
        
        # Entropy features
        features.append(_calculate_entropy(series[-30:]))
        features.append(_calculate_entropy(series[-10:]))
        
        # Autocorrelation features
        features.extend(_calculate_autocorr(series[-100:] if len(series) >= 100 else series))
        
        # Fourier features
        features.extend(_fourier_features(series[-50:] if len(series) >= 50 else series))
        
        # Trend and momentum features
        features.extend(_trend_features(series))
        
        # Advanced pattern features
        features.extend(_pattern_features(series))
        
        # Gap analysis for all digits
        for d in range(10):
            gap = 999
            for i in range(len(series) - 1, -1, -1):
                if series[i] == d:
                    gap = len(series) - 1 - i
                    break
            features.append(min(gap, 100))
        
        # EWMA with multiple decay rates
        for alpha in [0.9, 0.95, 0.99]:
            ewma_counts = np.zeros(10)
            weight = 1.0
            for val in reversed(series[-50:] if len(series) >= 50 else series):
                if 0 <= val <= 9:
                    ewma_counts[val] += weight
                weight *= alpha
                if weight < 1e-6:
                    break
            
            total = ewma_counts.sum()
            if total > 0:
                ewma_counts = ewma_counts / total
            features.extend(ewma_counts)
        
        # Sequential pattern features
        last_digit = series[-1] if series else 0
        for i in range(10):
            features.append(1 if last_digit == i else 0)
        
        # Streak analysis
        streak = 1
        if len(series) >= 2:
            for i in range(len(series) - 2, -1, -1):
                if series[i] == series[-1]:
                    streak += 1
                else:
                    break
        features.append(min(streak, 20))
        
        # Time-based features
        if dates and any(d is not None for d in dates):
            try:
                current_date = None
                for i in range(len(dates) - 1, -1, -1):
                    if dates[i] is not None:
                        current_date = pd.Timestamp(dates[i])
                        break
                
                if current_date:
                    features.extend([
                        current_date.weekday(),
                        current_date.day,
                        current_date.hour if hasattr(current_date, 'hour') else 12,
                        int(current_date.timestamp() % (24 * 3600)) // 3600  # Hour of day
                    ])
                else:
                    features.extend([0, 1, 12, 12])
            except:
                features.extend([0, 1, 12, 12])
        else:
            features.extend([0, 1, 12, 12])
        
        # Round-based features
        if rounds and any(r is not None for r in rounds):
            try:
                current_round = None
                for i in range(len(rounds) - 1, -1, -1):
                    if rounds[i] is not None:
                        current_round = rounds[i]
                        break
                
                for r in range(1, 9):
                    features.append(1 if current_round == r else 0)
            except:
                features.extend([0] * 8)
        else:
            features.extend([0] * 8)
        
        # Validate and clean features
        features = [float(f) if isinstance(f, (int, float, np.number)) and not (np.isnan(f) or np.isinf(f)) else 0.0 for f in features]
        
        feature_cache[cache_key] = features
        return features
        
    except Exception as e:
        print("⚠️ Enhanced feature extraction error: {}".format(e))
        return None

def _build_enhanced_xy(series, rounds=None, dates=None, validation_split=0.2):
    """Build training data with enhanced features and validation split"""
    try:
        if len(series) < 100:
            return [], [], [], []
        
        X, y = [], []
        start_idx = max(50, len(series) // 3)
        
        for i in range(start_idx, len(series) - 1):
            hist = series[:i]
            feats = _make_enhanced_features(hist, rounds[:i] if rounds else None, dates[:i] if dates else None)
            if feats is None:
                continue
            
            X.append(feats)
            y.append(series[i])
        
        if len(X) < 50:
            return [], [], [], []
        
        # Split into train and validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print("✅ Built enhanced training data: {} train, {} val samples".format(len(X_train), len(X_val)))
        return X_train, y_train, X_val, y_val
    except Exception as e:
        print("⚠️ Enhanced XY building error: {}".format(e))
        return [], [], [], []

# ======== ADVANCED ML MODELS ========
S1_model, S2_model, S3_model = None, None, None
scalers = {}

def train_advanced_randomforest():
    """Advanced Random Forest with feature selection and hyperparameter tuning"""
    try:
        if not ML_AVAILABLE or len(digits) < 150:
            return None
        
        X_train, y_train, X_val, y_val = _build_enhanced_xy(digits, rounds_hist, dates_hist)
        if len(X_train) < 100:
            return None
        
        print("🤖 Training Advanced Random Forest...")
        
        # Feature scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Feature selection
        selector = SelectKBest(mutual_info_classif, k=min(50, len(X_train[0])//2))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_val_selected = selector.transform(X_val_scaled)
        
        # Advanced Random Forest with optimized parameters
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='log2',
            bootstrap=True,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=1,
            warm_start=False,
            max_samples=0.8
        )
        
        model.fit(X_train_selected, y_train)
        
        # Validation scores
        train_score = model.score(X_train_selected, y_train)
        val_score = model.score(X_val_selected, y_val)
        
        print("✅ Advanced RF trained - Train: {:.3f}, Val: {:.3f}".format(train_score, val_score))
        
        scalers['S1'] = (scaler, selector)
        return model
        
    except Exception as e:
        print("❌ Advanced RF training error: {}".format(e))
        return None

def train_enhanced_gradientboosting():
    """Enhanced Gradient Boosting with advanced configuration"""
    try:
        if not ML_AVAILABLE or len(digits) < 150:
            return None
        
        X_train, y_train, X_val, y_val = _build_enhanced_xy(digits, rounds_hist, dates_hist)
        if len(X_train) < 100:
            return None
        
        print("🤖 Training Enhanced Gradient Boosting...")
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Enhanced Gradient Boosting
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            subsample=0.8,
            max_features='sqrt',
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-4
        )
        
        model.fit(X_train_scaled, y_train)
        
        train_score = model.score(X_train_scaled, y_train)
        val_score = model.score(X_val_scaled, y_val)
        
        print("✅ Enhanced GB trained - Train: {:.3f}, Val: {:.3f}".format(train_score, val_score))
        
        scalers['S2'] = scaler
        return model
        
    except Exception as e:
        print("❌ Enhanced GB training error: {}".format(e))
        return None

def train_neural_network_ensemble():
    """Neural Network ensemble for complex pattern recognition"""
    try:
        if not ML_AVAILABLE or len(digits) < 200:
            return None
        
        X_train, y_train, X_val, y_val = _build_enhanced_xy(digits, rounds_hist, dates_hist)
        if len(X_train) < 150:
            return None
        
        print("🤖 Training Neural Network Ensemble...")
        
        # Feature scaling
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Neural Network with multiple hidden layers
        model = MLPClassifier(
            hidden_layer_sizes=(200, 100, 50),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            tol=1e-4
        )
        
        model.fit(X_train_scaled, y_train)
        
        train_score = model.score(X_train_scaled, y_train)
        val_score = model.score(X_val_scaled, y_val)
        
        print("✅ Neural Network trained - Train: {:.3f}, Val: {:.3f}".format(train_score, val_score))
        
        scalers['S3'] = scaler
        return model
        
    except Exception as e:
        print("❌ Neural Network training error: {}".format(e))
        return None

# ======== SAFE UTILITY FUNCTIONS ========
def _safe_array(v, default_size=10):
    try:
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                return np.ones(default_size) / default_size
            arr = np.array(v, dtype=float)
        elif isinstance(v, (int, float)):
            arr = np.array([v], dtype=float)
        elif isinstance(v, np.ndarray):
            arr = v.astype(float)
        else:
            return np.ones(default_size) / default_size
        
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        arr = np.abs(arr)
        return arr
    except:
        return np.ones(default_size) / default_size

def _norm(v):
    try:
        arr = _safe_array(v, 10)
        arr[arr < 0] = 0.0
        s = arr.sum()
        if s > 0:
            return arr / s
        else:
            return np.ones(10) / 10.0
    except:
        return np.ones(10) / 10.0

def _advanced_softmax(x, temperature=1.0, sharpening=True):
    try:
        a = _safe_array(x, 10)
        if len(a) == 0:
            return np.ones(10) / 10.0
        
        if sharpening:
            # Enhanced sharpening based on confidence
            mean_val = np.mean(a)
            std_val = np.std(a)
            if std_val > 0:
                # Boost high-confidence predictions
                confidence_threshold = mean_val + 0.5 * std_val
                a = np.where(a > confidence_threshold, a * 1.5, a)
        
        a = a / max(temperature, 1e-9)
        a = a - a.max()  # Numerical stability
        e = np.exp(a)
        s = e.sum()
        
        if s > 0:
            result = e / s
        else:
            result = np.ones(len(a)) / len(a)
        
        return result
    except:
        return np.ones(10) / 10.0

# ======== ENHANCED PREDICTION STRATEGIES ========
def predict_S1():
    """Advanced Random Forest prediction"""
    try:
        global S1_model
        if S1_model is None or len(digits) < 150:
            S1_model = train_advanced_randomforest()
        
        if S1_model is None or 'S1' not in scalers:
            return _norm([1] * 10)
        
        feats = _make_enhanced_features(digits, rounds_hist, dates_hist)
        if feats is None:
            return _norm([1] * 10)
        
        try:
            scaler, selector = scalers['S1']
            feats_scaled = scaler.transform([feats])
            feats_selected = selector.transform(feats_scaled)
            probs = S1_model.predict_proba(feats_selected)[0]
            return _advanced_softmax(probs, temperature=0.8)
        except Exception as e:
            print("⚠️ S1 prediction error: {}".format(e))
            S1_model = train_advanced_randomforest()
            if S1_model and 'S1' in scalers:
                try:
                    scaler, selector = scalers['S1']
                    feats_scaled = scaler.transform([feats])
                    feats_selected = selector.transform(feats_scaled)
                    probs = S1_model.predict_proba(feats_selected)[0]
                    return _advanced_softmax(probs, temperature=0.8)
                except:
                    pass
            return _norm([1] * 10)
    except:
        return _norm([1] * 10)

def predict_S2():
    """Enhanced Gradient Boosting prediction"""
    try:
        global S2_model
        if S2_model is None or len(digits) < 150:
            S2_model = train_enhanced_gradientboosting()
        
        if S2_model is None or 'S2' not in scalers:
            return _norm([1] * 10)
        
        feats = _make_enhanced_features(digits, rounds_hist, dates_hist)
        if feats is None:
            return _norm([1] * 10)
        
        try:
            scaler = scalers['S2']
            feats_scaled = scaler.transform([feats])
            probs = S2_model.predict_proba(feats_scaled)[0]
            return _advanced_softmax(probs, temperature=0.9)
        except Exception as e:
            print("⚠️ S2 prediction error: {}".format(e))
            S2_model = train_enhanced_gradientboosting()
            if S2_model and 'S2' in scalers:
                try:
                    scaler = scalers['S2']
                    feats_scaled = scaler.transform([feats])
                    probs = S2_model.predict_proba(feats_scaled)[0]
                    return _advanced_softmax(probs, temperature=0.9)
                except:
                    pass
            return _norm([1] * 10)
    except:
        return _norm([1] * 10)

def predict_S3():
    """Neural Network ensemble prediction"""
    try:
        global S3_model
        if S3_model is None or len(digits) < 200:
            S3_model = train_neural_network_ensemble()
        
        if S3_model is None or 'S3' not in scalers:
            return _norm([1] * 10)
        
        feats = _make_enhanced_features(digits, rounds_hist, dates_hist)
        if feats is None:
            return _norm([1] * 10)
        
        try:
            scaler = scalers['S3']
            feats_scaled = scaler.transform([feats])
            probs = S3_model.predict_proba(feats_scaled)[0]
            return _advanced_softmax(probs, temperature=1.1, sharpening=True)
        except Exception as e:
            print("⚠️ S3 prediction error: {}".format(e))
            S3_model = train_neural_network_ensemble()
            if S3_model and 'S3' in scalers:
                try:
                    scaler = scalers['S3']
                    feats_scaled = scaler.transform([feats])
                    probs = S3_model.predict_proba(feats_scaled)[0]
                    return _advanced_softmax(probs, temperature=1.1)
                except:
                    pass
            return _norm([1] * 10)
    except:
        return _norm([1] * 10)

def predict_S4():
    """Enhanced Pattern Recognition with deep analysis"""
    try:
        if len(digits) < 50:
            return _norm([1] * 10)
        
        # Multi-scale pattern analysis
        patterns = np.zeros(10)
        
        # Short-term patterns (last 20)
        recent_20 = digits[-20:] if len(digits) >= 20 else digits
        for d in range(10):
            count = recent_20.count(d)
            patterns[d] += count * 0.4
        
        # Medium-term patterns (last 50)
        recent_50 = digits[-50:] if len(digits) >= 50 else digits
        for d in range(10):
            count = recent_50.count(d)
            expected = len(recent_50) / 10.0
            deviation = (count - expected) / max(expected, 1)
            patterns[d] += deviation * 0.3
        
        # Long-term reversion (last 100)
        if len(digits) >= 100:
            recent_100 = digits[-100:]
            for d in range(10):
                count = recent_100.count(d)
                expected = 10.0
                if count < expected * 0.8:  # Under-represented
                    patterns[d] += 0.3
        
        # Gap-based urgency
        for d in range(10):
            gap = 999
            for i in range(len(digits) - 1, -1, -1):
                if digits[i] == d:
                    gap = len(digits) - 1 - i
                    break
            
            # Higher urgency for longer gaps
            if gap > 15:
                patterns[d] += 0.4
            elif gap > 10:
                patterns[d] += 0.2
        
        # Sequential pattern boost
        if len(digits) >= 3:
            last_3 = digits[-3:]
            # Check for arithmetic progressions
            if last_3[1] - last_3[0] == last_3[2] - last_3[1]:
                diff = last_3[1] - last_3[0]
                next_pred = (last_3[2] + diff) % 10
                patterns[next_pred] += 0.5
        
        return _norm(patterns)
    except:
        return _norm([1] * 10)

def predict_S5():
    """Adaptive High Confidence with dynamic thresholding"""
    try:
        if len(digits) < 30:
            return _norm([1] * 10)
        
        confidence_scores = np.zeros(10)
        
        # Multi-window analysis
        windows = [10, 20, 30, 50]
        weights = [0.4, 0.3, 0.2, 0.1]
        
        for window_size, weight in zip(windows, weights):
            if len(digits) >= window_size:
                window_data = digits[-window_size:]
                
                # Frequency-based confidence
                for d in range(10):
                    count = window_data.count(d)
                    freq = count / len(window_data)
                    expected_freq = 0.1
                    
                    # Confidence based on deviation from expected
                    if freq > expected_freq:
                        confidence_scores[d] += (freq - expected_freq) * weight * 5
        
        # Gap-based confidence
        for d in range(10):
            gap = 999
            for i in range(len(digits) - 1, -1, -1):
                if digits[i] == d:
                    gap = len(digits) - 1 - i
                    break
            
            # Exponential decay confidence
            if gap < 50:
                gap_confidence = np.exp(-gap / 10.0)
                confidence_scores[d] += gap_confidence * 0.3
        
        # Trend-based confidence
        if len(digits) >= 20:
            for d in range(10):
                recent_positions = []
                for i in range(len(digits) - 1, max(-1, len(digits) - 21), -1):
                    if digits[i] == d:
                        recent_positions.append(len(digits) - 1 - i)
                
                if len(recent_positions) >= 2:
                    # Decreasing gaps indicate increasing probability
                    if recent_positions[0] < recent_positions[-1]:
                        confidence_scores[d] += 0.2
        
        return _advanced_softmax(confidence_scores, temperature=0.7, sharpening=True)
    except:
        return _norm([1] * 10)

def predict_M1():
    """Advanced Bayesian Inference with conjugate priors"""
    try:
        if len(digits) < 30:
            return _norm([1] * 10)
        
        # Hierarchical Bayesian approach
        prior_alpha = np.ones(10) * 0.5  # Weak prior
        
        # Evidence from different time scales
        time_scales = [10, 25, 50, 100]
        scale_weights = [0.4, 0.3, 0.2, 0.1]
        
        posterior_alpha = prior_alpha.copy()
        
        for scale, weight in zip(time_scales, scale_weights):
            if len(digits) >= scale:
                window_data = digits[-scale:]
                
                # Update posterior with weighted evidence
                for d in range(10):
                    count = window_data.count(d)
                    posterior_alpha[d] += count * weight
        
        # Context-dependent adjustment
        if len(digits) >= 5:
            last_5 = digits[-5:]
            context_boost = np.zeros(10)
            
            # Pattern completion heuristics
            unique_in_last_5 = len(set(last_5))
            if unique_in_last_5 < 5:  # Some digits repeated
                missing_digits = set(range(10)) - set(last_5)
                for d in missing_digits:
                    context_boost[d] += 0.5
            
            posterior_alpha += context_boost
        
        # Sample from Dirichlet distribution (approximated)
        expected_probs = posterior_alpha / posterior_alpha.sum()
        
        # Add uncertainty estimation
        concentration = posterior_alpha.sum()
        if concentration > 50:  # High certainty
            temperature = 0.8
        else:  # Low certainty
            temperature = 1.2
        
        return _advanced_softmax(expected_probs, temperature=temperature)
    except:
        return _norm([1] * 10)

def predict_M2():
    """Enhanced Markov Chain with variable order"""
    try:
        if len(digits) < 50:
            return _norm([1] * 10)
        
        # Multi-order Markov chains
        predictions = []
        weights = []
        
        # Order 1 Markov chain
        if len(digits) >= 20:
            transitions_1 = np.zeros((10, 10))
            for i in range(len(digits) - 1):
                curr, next_d = digits[i], digits[i + 1]
                if 0 <= curr <= 9 and 0 <= next_d <= 9:
                    transitions_1[curr][next_d] += 1
            
            # Normalize with Laplace smoothing
            for i in range(10):
                transitions_1[i] = (transitions_1[i] + 0.1) / (transitions_1[i].sum() + 1.0)
            
            if 0 <= digits[-1] <= 9:
                pred_1 = transitions_1[digits[-1]]
                predictions.append(pred_1)
                weights.append(0.4)
        
        # Order 2 Markov chain
        if len(digits) >= 30:
            transitions_2 = {}
            for i in range(len(digits) - 2):
                state = (digits[i], digits[i + 1])
                next_d = digits[i + 2]
                if state not in transitions_2:
                    transitions_2[state] = np.zeros(10)
                transitions_2[state][next_d] += 1
            
            # Normalize
            for state in transitions_2:
                total = transitions_2[state].sum()
                if total > 0:
                    transitions_2[state] = transitions_2[state] / total
            
            if len(digits) >= 2:
                state = (digits[-2], digits[-1])
                if state in transitions_2:
                    pred_2 = transitions_2[state]
                    predictions.append(pred_2)
                    weights.append(0.35)
        
        # Order 3 Markov chain (if enough data)
        if len(digits) >= 100:
            transitions_3 = {}
            for i in range(len(digits) - 3):
                state = (digits[i], digits[i + 1], digits[i + 2])
                next_d = digits[i + 3]
                if state not in transitions_3:
                    transitions_3[state] = np.zeros(10)
                transitions_3[state][next_d] += 1
            
            for state in transitions_3:
                total = transitions_3[state].sum()
                if total > 0:
                    transitions_3[state] = transitions_3[state] / total
            
            if len(digits) >= 3:
                state = (digits[-3], digits[-2], digits[-1])
                if state in transitions_3:
                    pred_3 = transitions_3[state]
                    predictions.append(pred_3)
                    weights.append(0.25)
        
        # Combine predictions
        if predictions:
            combined = np.zeros(10)
            total_weight = sum(weights)
            
            for pred, weight in zip(predictions, weights):
                combined += (weight / total_weight) * pred
            
            return _norm(combined)
        else:
            return _norm([1] * 10)
    except:
        return _norm([1] * 10)

def predict_M3():
    """Advanced Spectral Frequency Analysis"""
    try:
        if len(digits) < 60:
            return _norm([1] * 10)
        
        spectral_scores = np.zeros(10)
        
        # Multi-resolution spectral analysis
        for d in range(10):
            # Extract positions of digit d
            positions = []
            for i, val in enumerate(digits):
                if val == d:
                    positions.append(i)
            
            if len(positions) < 3:
                spectral_scores[d] = 0.1
                continue
            
            # Analyze inter-arrival times
            intervals = np.diff(positions)
            
            if len(intervals) > 0:
                # Spectral density estimation
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                
                # Current gap
                current_gap = len(digits) - 1 - positions[-1] if positions else 50
                
                # Probability based on spectral characteristics
                if std_interval > 0:
                    # Z-score of current gap
                    z_score = abs(current_gap - mean_interval) / std_interval
                    
                    # Higher probability for gaps close to mean
                    spectral_scores[d] = np.exp(-z_score / 2.0)
                else:
                    # Constant intervals
                    if abs(current_gap - mean_interval) < 2:
                        spectral_scores[d] = 0.8
                    else:
                        spectral_scores[d] = 0.2
        
        # Harmonic analysis
        if len(digits) >= 100:
            # Look for periodic patterns
            for period in [5, 7, 10, 12]:
                phase_scores = np.zeros(10)
                current_phase = len(digits) % period
                
                for d in range(10):
                    phase_counts = np.zeros(period)
                    
                    for i, val in enumerate(digits):
                        if val == d:
                            phase = i % period
                            phase_counts[phase] += 1
                    
                    # Boost score if digit is frequent at current phase
                    if phase_counts.sum() > 0:
                        phase_prob = phase_counts[current_phase] / phase_counts.sum()
                        phase_scores[d] = phase_prob
                
                # Weight by period importance
                weight = 0.3 / period
                spectral_scores += weight * phase_scores
        
        return _advanced_softmax(spectral_scores, temperature=0.9)
    except:
        return _norm([1] * 10)

def predict_A1():
    """Deep Sequence Analysis with LSTM-like patterns"""
    try:
        if len(digits) < 40:
            return _norm([1] * 10)
        
        sequence_scores = np.zeros(10)
        
        # Long short-term memory inspired analysis
        # Short-term memory (last 10)
        if len(digits) >= 10:
            short_term = digits[-10:]
            for d in range(10):
                recent_count = short_term.count(d)
                sequence_scores[d] += recent_count * 0.3
        
        # Medium-term memory (10-30 ago)
        if len(digits) >= 30:
            medium_term = digits[-30:-10]
            for d in range(10):
                medium_count = medium_term.count(d)
                # Decay factor for older information
                sequence_scores[d] += medium_count * 0.2
        
        # Long-term memory (30+ ago)
        if len(digits) >= 60:
            long_term = digits[-60:-30]
            for d in range(10):
                long_count = long_term.count(d)
                sequence_scores[d] += long_count * 0.1
        
        # Sequential pattern detection
        if len(digits) >= 20:
            # Look for repeating subsequences
            subseq_scores = np.zeros(10)
            
            for subseq_len in [3, 4, 5]:
                if len(digits) >= subseq_len * 2:
                    recent_subseq = digits[-subseq_len:]
                    
                    # Find similar subsequences in history
                    for i in range(len(digits) - subseq_len * 2):
                        hist_subseq = digits[i:i + subseq_len]
                        
                        # Calculate similarity
                        similarity = sum(1 for a, b in zip(recent_subseq, hist_subseq) if a == b)
                        similarity_ratio = similarity / subseq_len
                        
                        if similarity_ratio >= 0.6 and i + subseq_len < len(digits):
                            # What came next in history?
                            next_digit = digits[i + subseq_len]
                            if 0 <= next_digit <= 9:
                                # Weight by similarity and recency
                                recency_weight = 1.0 / (len(digits) - i)
                                subseq_scores[next_digit] += similarity_ratio * recency_weight
            
            # Normalize and add to main scores
            if subseq_scores.sum() > 0:
                subseq_scores = subseq_scores / subseq_scores.sum()
                sequence_scores += subseq_scores * 0.4
        
        # Attention mechanism - focus on important positions
        if len(digits) >= 30:
            attention_scores = np.zeros(10)
            
            # Calculate attention weights based on recent importance
            for lookback in range(1, min(20, len(digits))):
                pos = len(digits) - 1 - lookback
                digit_at_pos = digits[pos]
                
                # Attention weight decreases with distance but varies by pattern
                base_attention = 1.0 / (lookback + 1)
                
                # Boost attention for digits that appeared in similar contexts
                if lookback < 10:
                    context_match = 0
                    recent_context = digits[-min(5, lookback):]
                    
                    for i in range(max(0, pos - 2), min(len(digits) - 2, pos + 3)):
                        if i != pos:
                            hist_context = digits[max(0, i - 2):i + 3]
                            context_similarity = len(set(recent_context) & set(hist_context))
                            context_match += context_similarity
                    
                    attention_weight = base_attention * (1 + context_match * 0.1)
                else:
                    attention_weight = base_attention
                
                if 0 <= digit_at_pos <= 9:
                    attention_scores[digit_at_pos] += attention_weight
            
            # Normalize and combine
            if attention_scores.sum() > 0:
                attention_scores = attention_scores / attention_scores.sum()
                sequence_scores += attention_scores * 0.3
        
        return _norm(sequence_scores)
    except:
        return _norm([1] * 10)

def predict_A2():
    """Quantum-Inspired Pattern Matching with superposition states"""
    try:
        if len(digits) < 30:
            return _norm([1] * 10)
        
        # Quantum-inspired superposition of all possible states
        quantum_amplitudes = np.ones(10, dtype=complex)
        
        # Phase evolution based on historical patterns
        for d in range(10):
            # Calculate phase based on historical occurrences
            occurrences = []
            for i, val in enumerate(digits):
                if val == d:
                    occurrences.append(i)
            
            if occurrences:
                # Phase based on interference patterns
                current_pos = len(digits) - 1
                phase_sum = 0
                
                for occ in occurrences:
                    # Distance-based phase
                    distance = current_pos - occ
                    phase = (distance * np.pi) / 50.0  # Normalize phase
                    phase_sum += np.cos(phase)
                
                # Amplitude modulation
                amplitude = abs(phase_sum) / len(occurrences)
                phase_angle = np.arctan2(phase_sum, len(occurrences))
                
                quantum_amplitudes[d] = amplitude * np.exp(1j * phase_angle)
        
        # Measurement collapse - convert to probabilities
        probabilities = np.abs(quantum_amplitudes) ** 2
        
        # Quantum entanglement effects - correlations between digits
        if len(digits) >= 50:
            entanglement_matrix = np.zeros((10, 10))
            
            # Build correlation matrix
            for i in range(len(digits) - 1):
                curr, next_d = digits[i], digits[i + 1]
                if 0 <= curr <= 9 and 0 <= next_d <= 9:
                    entanglement_matrix[curr][next_d] += 1
            
            # Apply quantum corrections
            if 0 <= digits[-1] <= 9:
                entanglement_effects = entanglement_matrix[digits[-1]]
                if entanglement_effects.sum() > 0:
                    entanglement_effects = entanglement_effects / entanglement_effects.sum()
                    probabilities = 0.7 * probabilities + 0.3 * entanglement_effects
        
        # Uncertainty principle - add controlled randomness for very certain states
        max_prob = np.max(probabilities)
        if max_prob > 0.8:  # Too certain, add quantum uncertainty
            uncertainty = np.random.normal(0, 0.1, 10)
            probabilities += np.abs(uncertainty)
        
        return _norm(probabilities)
    except:
        return _norm([1] * 10)

# ======== ENHANCED ENSEMBLE PREDICTION ========
def calculate_strategy_confidence(predictions, recent_performance):
    """Calculate dynamic confidence scores for each strategy"""
    try:
        confidence_scores = {}
        
        for strategy_name, pred in predictions.items():
            base_confidence = ENSEMBLE_WEIGHTS.get(strategy_name, 0.1)
            
            # Adjust based on recent performance
            if strategy_name in recent_performance and recent_performance[strategy_name]:
                recent_acc = np.mean(recent_performance[strategy_name])
                performance_multiplier = min(2.0, max(0.5, recent_acc / 0.2))  # Scale around 20% baseline
                adjusted_confidence = base_confidence * performance_multiplier
            else:
                adjusted_confidence = base_confidence
            
            # Entropy-based confidence adjustment
            entropy_score = entropy(pred)
            max_entropy = np.log2(10)  # Maximum entropy for uniform distribution
            normalized_entropy = entropy_score / max_entropy
            
            # Lower entropy (more certain) gets higher confidence
            entropy_multiplier = 1.0 + (1.0 - normalized_entropy) * 0.5
            final_confidence = adjusted_confidence * entropy_multiplier
            
            confidence_scores[strategy_name] = final_confidence
        
        return confidence_scores
    except:
        # Fallback to equal weights
        return {name: 1.0/len(predictions) for name in predictions.keys()}

def get_enhanced_ensemble_prediction():
    """Get sophisticated ensemble prediction with dynamic weighting"""
    try:
        predictions = {}
        
        # Get all strategy predictions
        strategies = {
            'S1': predict_S1, 'S2': predict_S2, 'S3': predict_S3, 
            'S4': predict_S4, 'S5': predict_S5, 'M1': predict_M1, 
            'M2': predict_M2, 'M3': predict_M3, 'A1': predict_A1, 'A2': predict_A2
        }
        
        for name, func in strategies.items():
            try:
                pred = func()
                predictions[name] = pred
            except Exception as e:
                print("⚠️ Strategy {} failed: {}".format(name, e))
                predictions[name] = _norm([1] * 10)
        
        # Collect recent performance data
        recent_performance = {}
        all_stats = [S1_stats, S2_stats, S3_stats, S4_stats, S5_stats, 
                    M1_stats, M2_stats, M3_stats, A1_stats, A2_stats]
        
        for stats in all_stats:
            strategy_name = stats['name'].split()[0]  # Get S1, S2, etc.
            if stats['recent_acc']:
                recent_performance[strategy_name] = list(stats['recent_acc'])
        
        # Calculate dynamic confidence scores
        confidence_scores = calculate_strategy_confidence(predictions, recent_performance)
        
        # Multi-level ensemble
        # Level 1: Individual strategy predictions
        ensemble_L1 = np.zeros(10)
        total_confidence = sum(confidence_scores.values())
        
        for name, pred in predictions.items():
            weight = confidence_scores[name] / total_confidence if total_confidence > 0 else 1.0/len(predictions)
            ensemble_L1 += weight * pred
        
        # Level 2: Meta-ensemble with different aggregation methods
        ensemble_methods = []
        
        # Method 1: Weighted average (from Level 1)
        ensemble_methods.append(ensemble_L1)
        
        # Method 2: Median ensemble
        pred_matrix = np.array([pred for pred in predictions.values()])
        ensemble_median = np.median(pred_matrix, axis=0)
        ensemble_methods.append(ensemble_median)
        
        # Method 3: Max ensemble (for high-confidence picks)
        ensemble_max = np.max(pred_matrix, axis=0)
        ensemble_methods.append(ensemble_max)
        
        # Method 4: Geometric mean ensemble
        pred_matrix_pos = pred_matrix + 1e-8  # Avoid zeros
        ensemble_geom = np.power(np.prod(pred_matrix_pos, axis=0), 1.0/len(predictions))
        ensemble_methods.append(ensemble_geom)
        
        # Combine ensemble methods with adaptive weights
        method_weights = [0.5, 0.2, 0.15, 0.15]  # Favor weighted average
        
        final_ensemble = np.zeros(10)
        for method, weight in zip(ensemble_methods, method_weights):
            final_ensemble += weight * _norm(method)
        
        # Final normalization and sharpening
        final_ensemble = _norm(final_ensemble)
        final_ensemble = _advanced_softmax(final_ensemble, temperature=0.85, sharpening=True)
        
        return final_ensemble, predictions, confidence_scores
    except Exception as e:
        print("⚠️ Enhanced ensemble error: {}".format(e))
        return _norm([1] * 10), {}, {}

# ======== DATA LOADING (keep existing function) ========
def _parse_date_flexible(s):
    if not s:
        return None
    try:
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
        
        try:
            dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
            return dt.to_pydatetime().date() if pd.notna(dt) else None
        except:
            return None
    except Exception as e:
        print("⚠️ Date parsing error: {}".format(e))
        return None

def load_google_sheets_data():
    """Enhanced data loading (keeping original logic but with better error handling)"""
    global digits, rounds_hist, dates_hist
    
    try:
        old_n = len(digits)
        digits, rounds_hist, dates_hist = [], [], []
        
        print("📊 Loading Google Sheets data...")
        r = requests.get(CSV_URL, timeout=30)
        r.raise_for_status()
        
        content = r.text.strip()
        if not content:
            print("❌ Empty CSV content")
            return False
        
        lines = content.split('\n')
        if len(lines) < 2:
            print("❌ Insufficient CSV data")
            return False
        
        # Find headers
        headers = None
        header_line = 0
        for i, line in enumerate(lines[:5]):
            if not line.strip():
                continue
            cols = [c.strip().strip('"').strip("'") for c in line.split(',')]
            if any(h.lower() in ['digit', 'number', 'result'] for h in cols if h):
                headers = cols
                header_line = i
                break
        
        if headers is None:
            print("❌ Could not find valid headers")
            return False
        
        print("✅ Found headers: {}".format(headers))
        
        # Find column indices
        digit_names = ['digit', 'number', 'result', 'winning', 'win']
        round_names = ['round', 'game', 'period']
        date_names = ['date', 'time', 'timestamp', 'day']
        
        idx_digit = None
        idx_round = None
        idx_date = None
        
        for i, h in enumerate(headers):
            if not h:
                continue
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
        
        print("✅ Column indices - Digit: {}, Round: {}, Date: {}".format(idx_digit, idx_round, idx_date))
        
        # Process data
        valid_count = 0
        for line_num, line in enumerate(lines[header_line + 1:], start=header_line + 2):
            if not line.strip():
                continue
            
            try:
                cols = [c.strip().strip('"').strip("'") for c in line.split(',')]
                
                if len(cols) <= idx_digit:
                    continue
                
                digit_str = cols[idx_digit].strip()
                if not digit_str or not digit_str.replace('.', '').replace('-', '').isdigit():
                    continue
                
                val = int(float(digit_str))  # Handle float strings
                if not (0 <= val <= 9):
                    continue
                
                digits.append(val)
                valid_count += 1
                
                # Process round
                if idx_round is not None and len(cols) > idx_round and cols[idx_round].strip():
                    try:
                        round_str = cols[idx_round].strip()
                        if round_str.replace('.', '').isdigit():
                            round_val = int(float(round_str))
                            rounds_hist.append(round_val)
                        else:
                            rounds_hist.append(None)
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
                print("⚠️ Error processing line {}: {}".format(line_num, e))
                continue
        
        print("✅ Loaded {} valid records (Δ {})".format(len(digits), len(digits)-old_n))
        
        if len(digits) == 0:
            print("❌ No valid data found")
            return False
        
        # Clear caches on new data
        if len(digits) > old_n:
            feature_cache.clear()
            pattern_cache.clear()
            model_cache.clear()
        
        print("📈 Latest digits: {}".format(digits[-10:] if len(digits) >= 10 else digits))
        return True
        
    except requests.exceptions.RequestException as e:
        print("❌ Network error loading data: {}".format(e))
        return False
    except Exception as e:
        print("❌ Error loading data: {}".format(e))
        return False

# ======== PERFORMANCE TRACKING ========
def update_strategy_performance(actual_digit, predictions, confidence_scores):
    """Update performance metrics for all strategies"""
    try:
        if not (0 <= actual_digit <= 9):
            return
        
        strategy_map = {
            'S1': S1_stats, 'S2': S2_stats, 'S3': S3_stats, 'S4': S4_stats, 'S5': S5_stats,
            'M1': M1_stats, 'M2': M2_stats, 'M3': M3_stats, 'A1': A1_stats, 'A2': A2_stats
        }
        
        for strategy_name, pred in predictions.items():
            if strategy_name in strategy_map:
                stats = strategy_map[strategy_name]
                
                # Get top prediction
                top_pred = np.argmax(pred)
                top_3_preds = np.argsort(pred)[-3:]
                
                # Update counters
                stats['total'] += 1
                if top_pred == actual_digit:
                    stats['ok'] += 1
                
                # Top-3 accuracy
                if actual_digit in top_3_preds:
                    stats['top3_acc'] = (stats.get('top3_acc', 0) * (stats['total'] - 1) + 1) / stats['total']
                else:
                    stats['top3_acc'] = (stats.get('top3_acc', 0) * (stats['total'] - 1)) / stats['total']
                
                # Update accuracy
                stats['acc'] = (stats['ok'] / stats['total']) * 100
                
                # Update recent accuracy
                recent_correct = 1 if top_pred == actual_digit else 0
                stats['recent_acc'].append(recent_correct)
                
                # Update best/worst accuracy
                if stats['acc'] > stats['best_acc']:
                    stats['best_acc'] = stats['acc']
                if stats['acc'] < stats['worst_acc']:
                    stats['worst_acc'] = stats['acc']
                
                # Store prediction confidence
                prediction_confidence = pred[actual_digit]  # Confidence for actual digit
                stats['confidence_trend'].append(prediction_confidence)
                
                # Store entropy score
                entropy_score = entropy(pred)
                stats['entropy_scores'].append(entropy_score)
                
                # Store full prediction for analysis
                stats['prediction_history'].append({
                    'prediction': list(pred),
                    'actual': actual_digit,
                    'correct': recent_correct,
                    'confidence': prediction_confidence
                })
        
    except Exception as e:
        print("⚠️ Performance update error: {}".format(e))

# ======== TELEGRAM BOT HANDLERS ========
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    try:
        welcome_text = """🎯 **Enhanced Kolkata FF Prediction Bot**

🤖 **Commands:**
/predict or /p - Get next digit prediction
/stats - View detailed strategy statistics  
/reload - Reload data from sheets
/analysis - Get deep analysis of recent patterns
/confidence - View confidence metrics
/help - Show this help

🔮 **Enhanced Features:**
• 10 Advanced ML prediction strategies
• Neural Network ensemble learning
• Dynamic confidence weighting  
• Multi-scale pattern recognition
• Quantum-inspired algorithms
• Real-time performance tracking

💡 **Latest Updates:**
• Enhanced feature engineering
• Improved ensemble methods
• Better accuracy tracking
• Advanced pattern detection

⚠️ **Disclaimer:** Predictions are for entertainment only!"""
        
        bot.reply_to(message, welcome_text, parse_mode='Markdown')
    except Exception as e:
        print("⚠️ Welcome error: {}".format(e))

@bot.message_handler(commands=['predict', 'p'])
def predict_handler(message):
    try:
        if len(digits) < 50:
            bot.reply_to(message, "❌ Insufficient data for prediction. Loading...")
            if load_google_sheets_data():
                bot.reply_to(message, "✅ Data loaded. Try /predict again.")
            else:
                bot.reply_to(message, "❌ Failed to load data.")
            return
        
        # Get enhanced predictions
        ensemble, individual, confidence_scores = get_enhanced_ensemble_prediction()
        
        # Find top predictions with enhanced analysis
        sorted_indices = np.argsort(ensemble)[::-1]
        
        # Calculate prediction strength
        max_prob = ensemble[sorted_indices[0]]
        second_prob = ensemble[sorted_indices[1]] if len(sorted_indices) > 1 else 0
        strength = max_prob - second_prob
        
        # Determine confidence level
        if max_prob > 0.25 and strength > 0.08:
            confidence_level = "🔥 Very High"
            confidence_color = "🟢"
        elif max_prob > 0.18 and strength > 0.05:
            confidence_level = "⭐ High" 
            confidence_color = "🟡"
        elif max_prob > 0.12:
            confidence_level = "💫 Medium"
            confidence_color = "🟠"
        else:
            confidence_level = "📊 Low"
            confidence_color = "🔴"
        
        # Format response
        response = "🎯 **ENHANCED KOLKATA FF PREDICTION**\n\n"
        response += "{} **Confidence Level:** {}\n\n".format(confidence_color, confidence_level)
        
        response += "🏆 **Top 3 Predictions:**\n"
        for i in range(min(3, len(sorted_indices))):
            idx = sorted_indices[i]
            prob = ensemble[idx] * 100
            
            if i == 0:
                indicator = "🥇"
            elif i == 1:
                indicator = "🥈"
            else:
                indicator = "🥉"
            
            response += "{} **{}** - {:.1f}%\n".format(indicator, idx, prob)
        
        response += "\n📊 **Detailed Probabilities:**\n"
        for d in range(10):
            prob_percent = ensemble[d] * 100
            
            if prob_percent >= 15:
                indicator = "🔥"
            elif prob_percent >= 12:
                indicator = "⭐"
            elif prob_percent >= 10:
                indicator = "💫"
            elif prob_percent >= 8:
                indicator = "📈"
            else:
                indicator = "📊"
            
            response += "{} {}: {:.1f}%\n".format(indicator, d, prob_percent)
        
        # Add strategy consensus
        top_digit = sorted_indices[0]
        agreement_count = sum(1 for pred in individual.values() if np.argmax(pred) == top_digit)
        total_strategies = len(individual)
        consensus_percent = (agreement_count / total_strategies) * 100
        
        response += "\n🤝 **Strategy Consensus:** {:.0f}% ({}/{})".format(
            consensus_percent, agreement_count, total_strategies)
        
        # Add data info
        response += "\n\n📈 **Data:** {} records".format(len(digits))
        response += "\n🕐 **Time:** {}".format(datetime.now().strftime("%H:%M:%S"))
        
        # Add prediction strength indicator
        if strength > 0.12:
            response += "\n💪 **Prediction Strength:** Very Strong"
        elif strength > 0.08:
            response += "\n💪 **Prediction Strength:** Strong"
        elif strength > 0.05:
            response += "\n💪 **Prediction Strength:** Moderate"
        else:
            response += "\n💪 **Prediction Strength:** Weak"
        
        bot.reply_to(message, response, parse_mode='Markdown')
        
    except Exception as e:
        print("⚠️ Enhanced predict error: {}".format(e))
        bot.reply_to(message, "❌ Prediction failed. Try again.")

@bot.message_handler(commands=['stats'])
def stats_handler(message):
    try:
        all_stats = [S1_stats, S2_stats, S3_stats, S4_stats, S5_stats, 
                    M1_stats, M2_stats, M3_stats, A1_stats, A2_stats]
        
        response = "📊 **ENHANCED STRATEGY STATISTICS**\n\n"
        
        # Sort by accuracy
        sorted_stats = sorted(all_stats, key=lambda x: x['acc'], reverse=True)
        
        for i, stats in enumerate(sorted_stats):
            if stats['total'] > 0:
                acc = stats['acc']
                
                if i == 0:
                    status = "👑"  # Best performer
                elif acc > 25:
                    status = "🔥"
                elif acc > 20:
                    status = "⭐"
                elif acc > 15:
                    status = "💫"
                else:
                    status = "📊"
                
                response += "{} **{}**\n".format(status, stats['name'])
                response += "   📈 Accuracy: {:.1f}% ({}/{})\n".format(acc, stats['ok'], stats['total'])
                
                if stats.get('top3_acc', 0) > 0:
                    response += "   🎯 Top-3 Acc: {:.1f}%\n".format(stats['top3_acc'] * 100)
                
                if stats.get('best_acc', 0) > 0:
                    response += "   🏆 Best: {:.1f}%\n".format(stats['best_acc'])
                
                # Recent performance
                if stats['recent_acc']:
                    recent_10 = list(stats['recent_acc'])[-10:]
                    recent_acc = (sum(recent_10) / len(recent_10)) * 100
                    trend = "📈" if recent_acc > acc else "📉" if recent_acc < acc else "➡️"
                    response += "   {} Recent: {:.1f}%\n".format(trend, recent_acc)
                
                response += "\n"
        
        # Overall statistics
        total_predictions = sum(s['total'] for s in all_stats)
        total_correct = sum(s['ok'] for s in all_stats)
        overall_accuracy = (total_correct / total_predictions * 100) if total_predictions > 0 else 0
        
        response += "📊 **OVERALL PERFORMANCE**\n"
        response += "🎯 Combined Accuracy: {:.1f}%\n".format(overall_accuracy)
        response += "📈 Total Predictions: {}\n".format(total_predictions)
        response += "🎲 Total Records: {}\n".format(len(digits))
        response += "🕐 Last Update: {}".format(datetime.now().strftime("%H:%M:%S"))
        
        bot.reply_to(message, response, parse_mode='Markdown')
    except Exception as e:
        print("⚠️ Enhanced stats error: {}".format(e))
        bot.reply_to(message, "❌ Stats retrieval failed.")

@bot.message_handler(commands=['analysis'])
def analysis_handler(message):
    try:
        if len(digits) < 50:
            bot.reply_to(message, "❌ Insufficient data for analysis.")
            return
        
        response = "🔍 **DEEP PATTERN ANALYSIS**\n\n"
        
        # Recent pattern analysis
        recent_20 = digits[-20:] if len(digits) >= 20 else digits
        recent_counts = Counter(recent_20)
        
        response += "📊 **Recent 20 Results:**\n"
        most_frequent = recent_counts.most_common(3)
        for digit, count in most_frequent:
            percentage = (count / len(recent_20)) * 100
            response += "• {}: {} times ({:.1f}%)\n".format(digit, count, percentage)
        
        # Gap analysis
        response += "\n⏳ **Current Gaps:**\n"
        gaps = {}
        for d in range(10):
            gap = 999
            for i in range(len(digits) - 1, -1, -1):
                if digits[i] == d:
                    gap = len(digits) - 1 - i
                    break
            gaps[d] = gap
        
        # Show digits with longest gaps
        sorted_gaps = sorted(gaps.items(), key=lambda x: x[1], reverse=True)[:5]
        for digit, gap in sorted_gaps:
            urgency = "🔥" if gap > 20 else "⚠️" if gap > 15 else "📊"
            response += "{} Digit {}: {} draws ago\n".format(urgency, digit, gap)
        
        # Entropy analysis
        entropy_recent = _calculate_entropy(recent_20)
        max_entropy = math.log2(10)
        entropy_percent = (entropy_recent / max_entropy) * 100
        
        response += "\n🌀 **Randomness Level:** {:.1f}%\n".format(entropy_percent)
        if entropy_percent > 85:
            response += "   Very random - hard to predict\n"
        elif entropy_percent > 70:
            response += "   Moderately random\n"
        else:
            response += "   Some patterns detectable\n"
        
        # Streak analysis
        if len(digits) >= 2:
            current_digit = digits[-1]
            streak = 1
            for i in range(len(digits) - 2, -1, -1):
                if digits[i] == current_digit:
                    streak += 1
                else:
                    break
            
            response += "\n🔄 **Current Streak:** Digit {} appeared {} time(s)\n".format(current_digit, streak)
            
            if streak >= 3:
                response += "   ⚠️ Long streak - may break soon\n"
        
        # Hot/Cold analysis
        recent_50 = digits[-50:] if len(digits) >= 50 else digits
        expected_freq = len(recent_50) / 10
        
        hot_digits = []
        cold_digits = []
        
        for d in range(10):
            count = recent_50.count(d)
            if count > expected_freq * 1.3:
                hot_digits.append((d, count))
            elif count < expected_freq * 0.7:
                cold_digits.append((d, count))
        
        if hot_digits:
            response += "\n🔥 **Hot Digits:** "
            response += ", ".join(str(d) for d, _ in hot_digits[:3])
        
        if cold_digits:
            response += "\n🧊 **Cold Digits:** "
            response += ", ".join(str(d) for d, _ in cold_digits[:3])
        
        response += "\n\n🕐 **Analysis Time:** {}".format(datetime.now().strftime("%H:%M:%S"))
        
        bot.reply_to(message, response, parse_mode='Markdown')
        
    except Exception as e:
        print("⚠️ Analysis error: {}".format(e))
        bot.reply_to(message, "❌ Analysis failed.")

@bot.message_handler(commands=['confidence'])
def confidence_handler(message):
    try:
        all_stats = [S1_stats, S2_stats, S3_stats, S4_stats, S5_stats, 
                    M1_stats, M2_stats, M3_stats, A1_stats, A2_stats]
        
        response = "🎯 **CONFIDENCE METRICS**\n\n"
        
        for stats in all_stats:
            if stats['total'] > 0 and stats['confidence_trend']:
                avg_confidence = np.mean(stats['confidence_trend'])
                strategy_name = stats['name'].split()[0]  # Get S1, S2, etc.
                
                confidence_level = "🔥" if avg_confidence > 0.15 else "⭐" if avg_confidence > 0.12 else "💫"
                
                response += "{} **{}:** {:.1f}%\n".format(
                    confidence_level, strategy_name, avg_confidence * 100)
        
        # Show ensemble confidence if available
        if len(digits) >= 50:
            try:
                ensemble, _, confidence_scores = get_enhanced_ensemble_prediction()
                max_ensemble_prob = np.max(ensemble)
                
                response += "\n🎲 **Current Ensemble Confidence:** {:.1f}%\n".format(max_ensemble_prob * 100)
                
                if max_ensemble_prob > 0.2:
                    response += "   Status: Very Confident 🔥\n"
                elif max_ensemble_prob > 0.15:
                    response += "   Status: Confident ⭐\n"
                elif max_ensemble_prob > 0.12:
                    response += "   Status: Moderate 💫\n"
                else:
                    response += "   Status: Low Confidence 📊\n"
                    
            except:
                pass
        
        response += "\n🕐 **Updated:** {}".format(datetime.now().strftime("%H:%M:%S"))
        
        bot.reply_to(message, response, parse_mode='Markdown')
        
    except Exception as e:
        print("⚠️ Confidence error: {}".format(e))
        bot.reply_to(message, "❌ Confidence metrics failed.")

@bot.message_handler(commands=['reload'])
def reload_handler(message):
    try:
        bot.reply_to(message, "🔄 Reloading enhanced data...")
        success = load_google_sheets_data()
        
        if success:
            # Clear model cache to retrain with new data
            global S1_model, S2_model, S3_model
            S1_model, S2_model, S3_model = None, None, None
            model_cache.clear()
            
            bot.reply_to(message, "✅ Data reloaded! {} records available. Models will retrain on next prediction.".format(len(digits)))
        else:
            bot.reply_to(message, "❌ Failed to reload data.")
    except Exception as e:
        print("⚠️ Reload error: {}".format(e))
        bot.reply_to(message, "❌ Reload failed.")

# ======== MAIN EXECUTION ========
def run_flask():
    """Run Flask server"""
    try:
        port = int(os.getenv('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
    except Exception as e:
        print("⚠️ Flask error: {}".format(e))

def main():
    """Enhanced main execution function"""
    print("🚀 Starting Enhanced Kolkata FF Bot...")
    
    # Load learning data
    load_learning_data()
    
    # Initial data load
    print("📊 Loading initial data...")
    if load_google_sheets_data():
        print("✅ Initial data loaded: {} records".format(len(digits)))
        
        # Pre-train models if enough data
        if len(digits) >= 150:
            print("🤖 Pre-training models...")
            threading.Thread(target=lambda: [train_advanced_randomforest(), 
                                            train_enhanced_gradientboosting(),
                                            train_neural_network_ensemble()], daemon=True).start()
    else:
        print("⚠️ Initial data load failed")
    
    # Start Flask in separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Periodic data updates
    def periodic_update():
        while True:
            try:
                time.sleep(1800)  # Update every 30 minutes
                if load_google_sheets_data():
                    save_learning_data()
                    print("🔄 Periodic update completed")
            except Exception as e:
                print("⚠️ Periodic update error: {}".format(e))
    
    update_thread = threading.Thread(target=periodic_update, daemon=True)
    update_thread.start()
    
    print("🤖 Enhanced Bot started successfully!")
    print("📡 Telegram bot polling...")
    
    # Start bot polling with enhanced error handling
    try:
        while True:
            try:
                bot.infinity_polling(timeout=10, long_polling_timeout=5)
            except Exception as e:
                print("⚠️ Bot polling error: {}".format(e))
                time.sleep(5)  # Brief pause before restarting
    except KeyboardInterrupt:
        print("🛑 Bot stopped by user")
    finally:
        save_learning_data()
        print("✅ Learning data saved on exit")

if __name__ == "__main__":
    main()