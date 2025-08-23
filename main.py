# ===================== Enhanced Kolkata FF Bot with Advanced Strategies - FIXED =====================
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

# Advanced ML imports with error handling
try:
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
    ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML libraries not available: {e}")
    ML_AVAILABLE = False

import hashlib

# ======== CONFIG ========
BOT_TOKEN = os.getenv("BOT_TOKEN", "8306210029:AAHl7sxAEEq0FT750MAThHrAioYyAbRI1oI")
ADMIN_CHAT_ID = None
SPREADSHEET_ID = "10wI8T-NzqYsq6L73kPZ_bibuv2dw7xhQAmOr0msvk1A"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=csv"

# ======== GLOBALS ========
bot = telebot.TeleBot(BOT_TOKEN)
app = Flask(__name__)

@app.route("/")
def home():
    return "Kolkata FF Bot is running!"

@app.route("/healthz")
def _healthz():
    return "ok"

# Enhanced data structures with error handling
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
    "A1": 0.11, "A2": 0.10, "A3": 0.10
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
    except Exception as e:
        print(f"⚠️ Write error: {e}")
        try:
            if 'tmp_path' in locals():
                os.remove(tmp_path)
        except:
            pass

def save_learning_data():
    try:
        data = {
            "S1": dict(S1_stats), "S2": dict(S2_stats), "S4": dict(S4_stats), "S5": dict(S5_stats),
            "M1": dict(M1_stats), "M2": dict(M2_stats), "M3": dict(M3_stats),
            "A1": dict(A1_stats), "A2": dict(A2_stats), "A3": dict(A3_stats),
            "last_update": datetime.now().isoformat(),
            "data_size": len(digits)
        }
        # Add autobet data if available
        if 'AUTO' in globals():
            data["autobet"] = AUTO.save_state()
        
        _atomic_write_json(learning_storage_file, data)
        print("✅ Learning data saved")
    except Exception as e:
        print(f"⚠️ Error saving learning data: {e}")

def load_learning_data():
    try:
        if os.path.exists(learning_storage_file) and os.path.getsize(learning_storage_file) > 0:
            try:
                with open(learning_storage_file, "r") as f:
                    data = json.load(f)
            except Exception:
                print("⚠️ learning_data.json corrupt — resetting")
                data = {}
        else:
            data = {}
        
        # Load strategy stats
        for name, store in [
            ("S1", S1_stats), ("S2", S2_stats), ("S4", S4_stats), ("S5", S5_stats),
            ("M1", M1_stats), ("M2", M2_stats), ("M3", M3_stats),
            ("A1", A1_stats), ("A2", A2_stats), ("A3", A3_stats)
        ]:
            if isinstance(data.get(name), dict):
                store.update(data[name])
        
        # Load autobet if available
        if "autobet" in data and 'AUTO' in globals():
            AUTO.load_state(data["autobet"])
        
        print("✅ Learning state loaded")
        return True
    except Exception as e:
        print(f"⚠️ Error loading learning data: {e}")
        return False

# ======== ENHANCED UTILS ========
def _safe_array(v, default_size=10):
    """Safely convert to numpy array with error handling"""
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
        
        # Handle NaN and inf values
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Ensure positive values
        arr = np.abs(arr)
        
        return arr
    except Exception as e:
        print(f"⚠️ Array conversion error: {e}")
        return np.ones(default_size) / default_size

def _norm(v):
    """Safe normalization function"""
    try:
        arr = _safe_array(v, 10)
        arr[arr < 0] = 0.0
        s = arr.sum()
        if s > 0:
            return arr / s
        else:
            return np.ones(10) / 10.0
    except Exception as e:
        print(f"⚠️ Normalization error: {e}")
        return np.ones(10) / 10.0

def _softmax(x, t=1.0):
    """Safe softmax function"""
    try:
        if isinstance(x, (int, float)):
            return np.array([1.0])
        
        a = _safe_array(x, 10)
        if len(a) == 0:
            return np.ones(10) / 10.0
            
        a = a / max(t, 1e-9)
        a = a - a.max()  # Numerical stability
        e = np.exp(a)
        s = e.sum()
        
        if s > 0:
            result = e / s
        else:
            result = np.ones(len(a)) / len(a)
        
        return result
    except Exception as e:
        print(f"⚠️ Softmax error: {e}")
        return np.ones(10) / 10.0

def _advanced_softmax(x, temperature=1.0, sharpening=False):
    """Advanced softmax with error handling"""
    try:
        a = _safe_array(x, 10)
        if len(a) == 0:
            return np.ones(10) / 10.0
            
        if sharpening:
            mean_val = np.mean(a)
            std_val = np.std(a)
            if std_val > 0:
                a = np.where(a > mean_val + std_val, a * 1.3, a)
        
        return _softmax(a, temperature)
    except Exception as e:
        print(f"⚠️ Advanced softmax error: {e}")
        return np.ones(10) / 10.0

def _parse_date_flexible(s: str):
    """Enhanced date parsing with error handling"""
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
        
        # Fallback
        try:
            dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
            return dt.to_pydatetime().date() if pd.notna(dt) else None
        except:
            return None
    except Exception as e:
        print(f"⚠️ Date parsing error: {e}")
        return None

def load_google_sheets_data():
    """Enhanced data loading with comprehensive error handling"""
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
        
        print(f"✅ Found headers: {headers}")
        
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
        
        print(f"✅ Column indices - Digit: {idx_digit}, Round: {idx_round}, Date: {idx_date}")
        
        # Process data
        valid_count = 0
        for line_num, line in enumerate(lines[header_line + 1:], start=header_line + 2):
            if not line.strip():
                continue
            
            try:
                cols = [c.strip().strip('"').strip("'") for c in line.split(',')]
                
                # Process digit
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
                print(f"⚠️ Error processing line {line_num}: {e}")
                continue
        
        print(f"✅ Loaded {len(digits)} valid records (Δ {len(digits)-old_n})")
        
        if len(digits) == 0:
            print("❌ No valid data found")
            return False
        
        # Clear caches on new data
        if len(digits) > old_n:
            feature_cache.clear()
            pattern_cache.clear()
        
        print(f"📈 Latest digits: {digits[-10:] if len(digits) >= 10 else digits}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error loading data: {e}")
        return False
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

# Initialize stats
S1_stats = create_enhanced_stats('RF Advanced')
S2_stats = create_enhanced_stats('GB Advanced')
S4_stats = create_enhanced_stats('SelfTopK Enhanced')
S5_stats = create_enhanced_stats('HighConf Adaptive')
M1_stats = create_enhanced_stats('BayesDirichlet Pro')
M2_stats = create_enhanced_stats('HazardGap Pro')
M3_stats = create_enhanced_stats('Residue Pro')
A1_stats = create_enhanced_stats('Spectral Analysis')
A2_stats = create_enhanced_stats('Markov Chain Deep')
A3_stats = create_enhanced_stats('Quantum Inspired')

# ======== SAFE FEATURE EXTRACTION ========
def _make_features(series, rounds=None, dates=None):
    """Safe feature extraction with error handling"""
    try:
        if len(series) < 10:
            return None
        
        cache_key = f"orig_{len(series)}_{hash(str(series[-20:]) if len(series) >= 20 else str(series))}"
        if cache_key in feature_cache:
            return feature_cache[cache_key]
        
        feats = []
        
        # Last 10 digits
        last_10 = series[-10:]
        feats.extend(last_10)
        
        # Frequency analysis - last 20
        window_20 = series[-20:] if len(series) >= 20 else series
        for d in range(10):
            count = window_20.count(d)
            freq = count / len(window_20)
            feats.append(freq)
        
        # Frequency analysis - last 50
        window_50 = series[-50:] if len(series) >= 50 else series
        for d in range(10):
            count = window_50.count(d)
            freq = count / len(window_50)
            feats.append(freq)
        
        # Gap analysis
        for d in range(10):
            gap = 999
            for i in range(len(series) - 1, -1, -1):
                if series[i] == d:
                    gap = len(series) - 1 - i
                    break
            feats.append(min(gap, 100))  # Cap gap at 100
        
        # EWMA counts with safe computation
        try:
            acc = np.zeros(10)
            w = 1.0
            for v in reversed(series):
                if 0 <= v <= 9:  # Safety check
                    acc[v] += w
                w *= 0.97
                if w < 1e-6:
                    break
            
            total = acc.sum()
            if total > 0:
                acc = acc / total
            else:
                acc = np.ones(10) / 10.0
            feats.extend(list(acc))
        except Exception as e:
            print(f"⚠️ EWMA error: {e}")
            feats.extend([0.1] * 10)
        
        # Streak calculation
        streak = 1
        if len(series) >= 2:
            last_digit = series[-1]
            for i in range(len(series) - 2, -1, -1):
                if series[i] == last_digit:
                    streak += 1
                else:
                    break
        feats.append(min(streak, 20))  # Cap streak at 20
        
        # Weekday features
        if dates and any(d is not None for d in dates):
            try:
                wd = None
                for i in range(len(dates) - 1, -1, -1):
                    if dates[i] is not None:
                        wd = pd.Timestamp(dates[i]).weekday()
                        break
                for w in range(7):
                    feats.append(1 if wd == w else 0)
            except Exception as e:
                print(f"⚠️ Date feature error: {e}")
                feats.extend([0] * 7)
        else:
            feats.extend([0] * 7)
        
        # Round features
        if rounds and any(r is not None for r in rounds):
            try:
                rcur = None
                for i in range(len(rounds) - 1, -1, -1):
                    if rounds[i] is not None:
                        rcur = rounds[i]
                        break
                for rv in range(1, 9):
                    feats.append(1 if rcur == rv else 0)
            except Exception as e:
                print(f"⚠️ Round feature error: {e}")
                feats.extend([0] * 8)
        else:
            feats.extend([0] * 8)
        
        # Validate features
        feats = [float(f) if isinstance(f, (int, float, np.number)) and not np.isnan(f) else 0.0 for f in feats]
        
        feature_cache[cache_key] = feats
        return feats
        
    except Exception as e:
        print(f"⚠️ Feature extraction error: {e}")
        return None

def _build_xy(series, rounds=None, dates=None):
    """Safe XY building with error handling"""
    try:
        if len(series) < 50:
            return [], []
        
        X, y = [], []
        start_idx = max(30, len(series) // 4)  # Adaptive start
        
        for i in range(start_idx, len(series) - 1):
            hist = series[:i]
            feats = _make_features(hist, rounds[:i] if rounds else None, dates[:i] if dates else None)
            if feats is None:
                continue
            
            X.append(feats)
            y.append(series[i])
        
        print(f"✅ Built training data: {len(X)} samples")
        return X, y
    except Exception as e:
        print(f"⚠️ XY building error: {e}")
        return [], []

# ======== SAFE MODELS ========
S1_model, S2_model = None, None

def train_randomforest_advanced():
    """Safe Random Forest training"""
    try:
        if not ML_AVAILABLE:
            print("⚠️ ML libraries not available")
            return None
            
        if len(digits) < 80:
            print("⚠️ Insufficient data for RF training")
            return None
        
        X, y = _build_xy(digits, rounds_hist, dates_hist)
        if len(X) < 50:
            print("⚠️ Insufficient training samples")
            return None
        
        print(f"🤖 Training Random Forest with {len(X)} samples...")
        
        # Safe train-test split
        try:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=0.2, random_state=42, 
                stratify=y if len(set(y)) > 1 else None
            )
        except:
            # Fallback without stratification
            X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Safe model training
        model = RandomForestClassifier(
            n_estimators=200,  # Reduced for stability
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            bootstrap=True,
            random_state=42,
            n_jobs=1  # Single thread for stability
        )
        
        model.fit(X_tr, y_tr)
        
        # Validate
        train_score = model.score(X_tr, y_tr)
        val_score = model.score(X_va, y_va)
        
        print(f"✅ RF trained - Train: {train_score:.3f}, Val: {val_score:.3f}")
        return model
        
    except Exception as e:
        print(f"❌ RF training error: {e}")
        return None

def train_gradientboosting_advanced():
    """Safe Gradient Boosting training"""
    try:
        if not ML_AVAILABLE:
            return None
            
        if len(digits) < 80:
            return None
        
        X, y = _build_xy(digits, rounds_hist, dates_hist)
        if len(X) < 50:
            return None
        
        print(f"🤖 Training Gradient Boosting...")
        
        model = GradientBoostingClassifier(
            n_estimators=100,  # Reduced for stability
            learning_rate=0.1,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        model.fit(X, y)
        score = model.score(X, y)
        
        print(f"✅ GB trained - Score: {score:.3f}")
        return model
        
    except Exception as e:
        print(f"❌ GB training error: {e}")
        return None

# ======== SAFE STRATEGY FUNCTIONS ========
def predict_S1():
    """Safe Random Forest prediction"""
    try:
        global S1_model
        if S1_model is None or len(digits) < 50:
            S1_model = train_randomforest_advanced()
        
        if S1_model is None:
            return _norm([1] * 10)
        
        feats = _make_features(digits, rounds_hist, dates_hist)
        if feats is None:
            return _norm([1] * 10)
        
        try:
            probs = S1_model.predict_proba([feats])[0]
            return _norm(probs)
        except Exception as e:
            print(f"⚠️ S1 prediction error: {e}")
            # Retrain on error
            S1_model = train_randomforest_advanced()
            if S1_model:
                try:
                    probs = S1_model.predict_proba([feats])[0]
                    return _norm(probs)
                except:
                    pass
            return _norm([1] * 10)
    except Exception as e:
        print(f"⚠️ S1 error: {e}")
        return _norm([1] * 10)

def predict_S2():
    """Safe Gradient Boosting prediction"""
    try:
        global S2_model
        if S2_model is None or len(digits) < 50:
            S2_model = train_gradientboosting_advanced()
        
        if S2_model is None:
            return _norm([1] * 10)
        
        feats = _make_features(digits, rounds_hist, dates_hist)
        if feats is None:
            return _norm([1] * 10)
        
        try:
            probs = S2_model.predict_proba([feats])[0]
            return _norm(probs)
        except Exception as e:
            print(f"⚠️ S2 prediction error: {e}")
            S2_model = train_gradientboosting_advanced()
            if S2_model:
                try:
                    probs = S2_model.predict_proba([feats])[0]
                    return _norm(probs)
                except:
                    pass
            return _norm([1] * 10)
    except Exception as e:
        print(f"⚠️ S2 error: {e}")
        return _norm([1] * 10)

def predict_S4():
    """Safe Self-TopK prediction"""
    try:
        if len(digits) < 20:
            return _norm([1] * 10)
        
        recent = digits[-50:] if len(digits) >= 50 else digits
        counts = [recent.count(d) for d in range(10)]
        
        # Self-referential adjustment
        last_digit = digits[-1]
        if 0 <= last_digit <= 9:
            counts[last_digit] *= 1.2
        
        return _norm(counts)
    except Exception as e:
        print(f"⚠️ S4 error: {e}")
        return _norm([1] * 10)

def predict_S5():
    """Safe High Confidence prediction"""
    try:
        if len(digits) < 30:
            return _norm([1] * 10)
        
        # Simple gap-based prediction
        gaps = {}
        for d in range(10):
            gap = 999
            for i in range(len(digits) - 1, -1, -1):
                if digits[i] == d:
                    gap = len(digits) - 1 - i
                    break
            gaps[d] = gap
        
        # Inverse gap probability
        inv_gaps = [1.0 / (gaps[d] + 1) for d in range(10)]
        return _norm(inv_gaps)
    except Exception as e:
        print(f"⚠️ S5 error: {e}")
        return _norm([1] * 10)

def predict_M1():
    """Safe Bayesian prediction"""
    try:
        if len(digits) < 20:
            return _norm([1] * 10)
        
        # Simple Dirichlet-like approach
        recent = digits[-30:] if len(digits) >= 30 else digits
        alpha = np.ones(10)  # Prior
        
        for d in recent:
            if 0 <= d <= 9:
                alpha[d] += 1
        
        return _norm(alpha)
    except Exception as e:
        print(f"⚠️ M1 error: {e}")
        return _norm([1] * 10)

def predict_M2():
    """Safe Hazard Gap prediction"""
    try:
        if len(digits) < 20:
            return _norm([1] * 10)
        
        hazards = []
        for d in range(10):
            gaps = []
            last_pos = None
            for i, val in enumerate(digits):
                if val == d:
                    if last_pos is not None:
                        gaps.append(i - last_pos)
                    last_pos = i
            
            if gaps:
                avg_gap = np.mean(gaps)
                current_gap = len(digits) - 1 - last_pos if last_pos is not None else avg_gap
                hazard = max(0, current_gap / max(avg_gap, 1))
            else:
                hazard = 1.0
            hazards.append(hazard)
        
        return _norm(hazards)
    except Exception as e:
        print(f"⚠