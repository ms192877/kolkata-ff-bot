# ===================== Enhanced Kolkata FF Bot - Railway Optimized =====================
import telebot
import os
import re
import time
import json
import threading
import requests
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from flask import Flask
from collections import Counter, deque, defaultdict
import warnings
warnings.filterwarnings("ignore")

# ML imports with proper error handling for Railway
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    from scipy.stats import entropy
    import joblib
    ML_AVAILABLE = True
    print("✅ ML libraries loaded successfully")
except ImportError as e:
    print(f"⚠️ ML libraries not available: {e}")
    ML_AVAILABLE = False

import hashlib
import math

# ======== CONFIG ========
BOT_TOKEN = os.getenv("BOT_TOKEN", "8306210029:AAHl7sxAEEq0FT750MAThHrAioYyAbRI1oI")
SPREADSHEET_ID = "10wI8T-NzqYsq6L73kPZ_bibuv2dw7xhQAmOr0msvk1A"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=csv"

# ======== FLASK APP ========
app = Flask(__name__)

@app.route("/")
def home():
    return "🎯 Enhanced Kolkata FF Bot is Running! 🤖"

@app.route("/health")
def health():
    return {"status": "healthy", "data_size": len(digits), "timestamp": datetime.now().isoformat()}

# ======== GLOBALS ========
bot = telebot.TeleBot(BOT_TOKEN)
digits, rounds_hist, dates_hist = [], [], []
model_cache = {}
feature_cache = {}

# Enhanced stats structure
def create_stats(name):
    return {
        'name': name, 'total': 0, 'ok': 0, 'acc': 0.0, 
        'recent_acc': deque(maxlen=50), 'best_acc': 0.0,
        'confidence_scores': deque(maxlen=30)
    }

# Initialize strategy stats
S1_stats = create_stats('Advanced Random Forest')
S2_stats = create_stats('Enhanced Gradient Boosting') 
S3_stats = create_stats('Neural Network')
S4_stats = create_stats('Pattern Recognition')
S5_stats = create_stats('Adaptive Confidence')
M1_stats = create_stats('Bayesian Inference')
M2_stats = create_stats('Markov Chain')
M3_stats = create_stats('Spectral Analysis')

# Ensemble weights
ENSEMBLE_WEIGHTS = {
    "S1": 0.15, "S2": 0.13, "S3": 0.12, "S4": 0.12, "S5": 0.10,
    "M1": 0.13, "M2": 0.10, "M3": 0.15
}

# ======== UTILITY FUNCTIONS ========
def _safe_array(v, default_size=10):
    try:
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                return np.ones(default_size) / default_size
            arr = np.array(v, dtype=float)
        else:
            return np.ones(default_size) / default_size
        
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        return np.abs(arr)
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

def _softmax(x, temperature=1.0):
    try:
        a = _safe_array(x, 10)
        if len(a) == 0:
            return np.ones(10) / 10.0
        
        a = a / max(temperature, 1e-9)
        a = a - a.max()
        e = np.exp(a)
        s = e.sum()
        
        return e / s if s > 0 else np.ones(len(a)) / len(a)
    except:
        return np.ones(10) / 10.0

# ======== FEATURE ENGINEERING ========
def _calculate_entropy(sequence):
    try:
        if not sequence:
            return 0
        counts = Counter(sequence)
        total = len(sequence)
        return -sum((count/total) * math.log2(count/total) for count in counts.values())
    except:
        return 0

def _make_features(series, rounds=None, dates=None):
    try:
        if len(series) < 30:
            return None
        
        # Cache key for performance
        cache_key = f"feat_{len(series)}_{hash(str(series[-20:]))}"
        if cache_key in feature_cache:
            return feature_cache[cache_key]
        
        features = []
        
        # Last N digits
        for n in [5, 10]:
            if len(series) >= n:
                features.extend(series[-n:])
            else:
                features.extend(series + [0] * (n - len(series)))
        
        # Frequency analysis
        recent_30 = series[-30:] if len(series) >= 30 else series
        for d in range(10):
            count = recent_30.count(d)
            features.append(count / len(recent_30))
        
        # Statistical features
        features.extend([
            np.mean(recent_30),
            np.std(recent_30),
            _calculate_entropy(recent_30)
        ])
        
        # Gap analysis
        for d in range(10):
            gap = 999
            for i in range(len(series) - 1, -1, -1):
                if series[i] == d:
                    gap = len(series) - 1 - i
                    break
            features.append(min(gap, 100))
        
        # Sequential patterns
        last_digit = series[-1] if series else 0
        for i in range(10):
            features.append(1 if last_digit == i else 0)
        
        # Clean features
        features = [float(f) if isinstance(f, (int, float, np.number)) and 
                   not (np.isnan(f) or np.isinf(f)) else 0.0 for f in features]
        
        feature_cache[cache_key] = features
        return features
        
    except Exception as e:
        print(f"⚠️ Feature extraction error: {e}")
        return None

def _build_xy(series, rounds=None, dates=None):
    try:
        if len(series) < 100:
            return [], []
        
        X, y = [], []
        start_idx = max(50, len(series) // 4)
        
        for i in range(start_idx, len(series) - 1):
            hist = series[:i]
            feats = _make_features(hist, rounds[:i] if rounds else None, 
                                  dates[:i] if dates else None)
            if feats is None:
                continue
            
            X.append(feats)
            y.append(series[i])
        
        print(f"✅ Built training data: {len(X)} samples")
        return X, y
    except Exception as e:
        print(f"⚠️ XY building error: {e}")
        return [], []

# ======== ML MODELS ========
S1_model, S2_model, S3_model = None, None, None
scalers = {}

def train_random_forest():
    try:
        if not ML_AVAILABLE or len(digits) < 150:
            return None
        
        X, y = _build_xy(digits, rounds_hist, dates_hist)
        if len(X) < 100:
            return None
        
        print("🤖 Training Random Forest...")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=1
        )
        
        model.fit(X_scaled, y)
        scalers['S1'] = scaler
        
        print("✅ Random Forest trained")
        return model
        
    except Exception as e:
        print(f"❌ Random Forest training error: {e}")
        return None

def train_gradient_boosting():
    try:
        if not ML_AVAILABLE or len(digits) < 150:
            return None
        
        X, y = _build_xy(digits, rounds_hist, dates_hist)
        if len(X) < 100:
            return None
        
        print("🤖 Training Gradient Boosting...")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        
        model.fit(X_scaled, y)
        scalers['S2'] = scaler
        
        print("✅ Gradient Boosting trained")
        return model
        
    except Exception as e:
        print(f"❌ Gradient Boosting training error: {e}")
        return None

def train_neural_network():
    try:
        if not ML_AVAILABLE or len(digits) < 200:
            return None
        
        X, y = _build_xy(digits, rounds_hist, dates_hist)
        if len(X) < 150:
            return None
        
        print("🤖 Training Neural Network...")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.01,
            max_iter=300,
            random_state=42
        )
        
        model.fit(X_scaled, y)
        scalers['S3'] = scaler
        
        print("✅ Neural Network trained")
        return model
        
    except Exception as e:
        print(f"❌ Neural Network training error: {e}")
        return None

# ======== PREDICTION STRATEGIES ========
def predict_S1():
    """Random Forest prediction"""
    try:
        global S1_model
        if S1_model is None or len(digits) < 150:
            S1_model = train_random_forest()
        
        if S1_model is None or 'S1' not in scalers:
            return _norm([1] * 10)
        
        feats = _make_features(digits, rounds_hist, dates_hist)
        if feats is None:
            return _norm([1] * 10)
        
        scaler = scalers['S1']
        feats_scaled = scaler.transform([feats])
        probs = S1_model.predict_proba(feats_scaled)[0]
        return _softmax(probs)
    except:
        return _norm([1] * 10)

def predict_S2():
    """Gradient Boosting prediction"""
    try:
        global S2_model
        if S2_model is None or len(digits) < 150:
            S2_model = train_gradient_boosting()
        
        if S2_model is None or 'S2' not in scalers:
            return _norm([1] * 10)
        
        feats = _make_features(digits, rounds_hist, dates_hist)
        if feats is None:
            return _norm([1] * 10)
        
        scaler = scalers['S2']
        feats_scaled = scaler.transform([feats])
        probs = S2_model.predict_proba(feats_scaled)[0]
        return _softmax(probs)
    except:
        return _norm([1] * 10)

def predict_S3():
    """Neural Network prediction"""
    try:
        global S3_model
        if S3_model is None or len(digits) < 200:
            S3_model = train_neural_network()
        
        if S3_model is None or 'S3' not in scalers:
            return _norm([1] * 10)
        
        feats = _make_features(digits, rounds_hist, dates_hist)
        if feats is None:
            return _norm([1] * 10)
        
        scaler = scalers['S3']
        feats_scaled = scaler.transform([feats])
        probs = S3_model.predict_proba(feats_scaled)[0]
        return _softmax(probs)
    except:
        return _norm([1] * 10)

def predict_S4():
    """Pattern Recognition"""
    try:
        if len(digits) < 50:
            return _norm([1] * 10)
        
        patterns = np.zeros(10)
        
        # Recent frequency
        recent_20 = digits[-20:] if len(digits) >= 20 else digits
        for d in range(10):
            count = recent_20.count(d)
            patterns[d] += count * 0.4
        
        # Gap analysis
        for d in range(10):
            gap = 999
            for i in range(len(digits) - 1, -1, -1):
                if digits[i] == d:
                    gap = len(digits) - 1 - i
                    break
            
            if gap > 15:
                patterns[d] += 0.4
            elif gap > 10:
                patterns[d] += 0.2
        
        return _norm(patterns)
    except:
        return _norm([1] * 10)

def predict_S5():
    """Adaptive High Confidence"""
    try:
        if len(digits) < 30:
            return _norm([1] * 10)
        
        confidence_scores = np.zeros(10)
        
        # Multi-window analysis
        windows = [10, 20, 30]
        weights = [0.5, 0.3, 0.2]
        
        for window_size, weight in zip(windows, weights):
            if len(digits) >= window_size:
                window_data = digits[-window_size:]
                
                for d in range(10):
                    count = window_data.count(d)
                    freq = count / len(window_data)
                    expected_freq = 0.1
                    
                    if freq > expected_freq:
                        confidence_scores[d] += (freq - expected_freq) * weight * 5
        
        return _softmax(confidence_scores)
    except:
        return _norm([1] * 10)

def predict_M1():
    """Bayesian Inference"""
    try:
        if len(digits) < 30:
            return _norm([1] * 10)
        
        # Bayesian approach with priors
        prior_alpha = np.ones(10) * 0.5
        
        time_scales = [10, 25, 50]
        scale_weights = [0.5, 0.3, 0.2]
        
        posterior_alpha = prior_alpha.copy()
        
        for scale, weight in zip(time_scales, scale_weights):
            if len(digits) >= scale:
                window_data = digits[-scale:]
                
                for d in range(10):
                    count = window_data.count(d)
                    posterior_alpha[d] += count * weight
        
        expected_probs = posterior_alpha / posterior_alpha.sum()
        return _softmax(expected_probs)
    except:
        return _norm([1] * 10)

def predict_M2():
    """Markov Chain"""
    try:
        if len(digits) < 50:
            return _norm([1] * 10)
        
        # First order Markov chain
        transitions = np.zeros((10, 10))
        for i in range(len(digits) - 1):
            curr, next_d = digits[i], digits[i + 1]
            if 0 <= curr <= 9 and 0 <= next_d <= 9:
                transitions[curr][next_d] += 1
        
        # Normalize with smoothing
        for i in range(10):
            transitions[i] = (transitions[i] + 0.1) / (transitions[i].sum() + 1.0)
        
        if 0 <= digits[-1] <= 9:
            pred = transitions[digits[-1]]
            return _norm(pred)
        
        return _norm([1] * 10)
    except:
        return _norm([1] * 10)

def predict_M3():
    """Spectral Analysis"""
    try:
        if len(digits) < 60:
            return _norm([1] * 10)
        
        spectral_scores = np.zeros(10)
        
        for d in range(10):
            positions = [i for i, val in enumerate(digits) if val == d]
            
            if len(positions) < 3:
                spectral_scores[d] = 0.1
                continue
            
            # Analyze intervals
            intervals = np.diff(positions)
            
            if len(intervals) > 0:
                mean_interval = np.mean(intervals)
                current_gap = len(digits) - 1 - positions[-1] if positions else 50
                
                # Probability based on gap similarity to historical patterns
                if abs(current_gap - mean_interval) < 5:
                    spectral_scores[d] = 0.8
                else:
                    spectral_scores[d] = 0.2
        
        return _softmax(spectral_scores)
    except:
        return _norm([1] * 10)

# ======== ENSEMBLE PREDICTION ========
def get_ensemble_prediction():
    """Get ensemble prediction from all strategies"""
    try:
        predictions = {}
        
        strategies = {
            'S1': predict_S1, 'S2': predict_S2, 'S3': predict_S3,
            'S4': predict_S4, 'S5': predict_S5, 'M1': predict_M1,
            'M2': predict_M2, 'M3': predict_M3
        }
        
        for name, func in strategies.items():
            try:
                pred = func()
                predictions[name] = pred
            except Exception as e:
                print(f"⚠️ Strategy {name} failed: {e}")
                predictions[name] = _norm([1] * 10)
        
        # Ensemble combination
        ensemble = np.zeros(10)
        total_weight = sum(ENSEMBLE_WEIGHTS.values())
        
        for name, pred in predictions.items():
            weight = ENSEMBLE_WEIGHTS.get(name, 0.1) / total_weight
            ensemble += weight * pred
        
        ensemble = _norm(ensemble)
        return ensemble, predictions
    except Exception as e:
        print(f"⚠️ Ensemble error: {e}")
        return _norm([1] * 10), {}

# ======== DATA LOADING ========
def _parse_date_flexible(s):
    if not s:
        return None
    try:
        s = s.strip()
        patterns = [
            (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"),
            (r"^\d{2}/\d{2}/\d{4}$", "%d/%m/%Y"),
            (r"^\d{2}-\d{2}-\d{4}$", "%d-%m-%Y"),
        ]
        
        for pattern, fmt in patterns:
            if re.match(pattern, s):
                try:
                    return pd.to_datetime(s, format=fmt, errors="coerce").date()
                except:
                    continue
        
        return pd.to_datetime(s, dayfirst=True, errors="coerce").date()
    except:
        return None

def load_google_sheets_data():
    """Load data from Google Sheets"""
    global digits, rounds_hist, dates_hist
    
    try:
        old_n = len(digits)
        digits, rounds_hist, dates_hist = [], [], []
        
        print("📊 Loading Google Sheets data...")
        r = requests.get(CSV_URL, timeout=30)
        r.raise_for_status()
        
        content = r.text.strip()
        if not content:
            return False
        
        lines = content.split('\n')
        if len(lines) < 2:
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
            return False
        
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
            return False
        
        # Process data
        for line in lines[header_line + 1:]:
            if not line.strip():
                continue
            
            try:
                cols = [c.strip().strip('"').strip("'") for c in line.split(',')]
                
                if len(cols) <= idx_digit:
                    continue
                
                digit_str = cols[idx_digit].strip()
                if not digit_str or not digit_str.replace('.', '').replace('-', '').isdigit():
                    continue
                
                val = int(float(digit_str))
                if not (0 <= val <= 9):
                    continue
                
                digits.append(val)
                
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
                    
            except:
                continue
        
        print(f"✅ Loaded {len(digits)} records (Δ {len(digits)-old_n})")
        
        if len(digits) == 0:
            return False
        
        # Clear caches on new data
        if len(digits) > old_n:
            feature_cache.clear()
            model_cache.clear()
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

# ======== TELEGRAM BOT HANDLERS ========
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    try:
        welcome_text = """🎯 **Enhanced Kolkata FF Prediction Bot**

🤖 **Commands:**
/predict or /p - Get next digit prediction
/stats - View strategy statistics  
/reload - Reload data from sheets
/analysis - Get pattern analysis
/help - Show this help

🔮 **Features:**
• 8 Advanced prediction strategies
• Machine Learning models
• Pattern recognition
• Statistical analysis
• Real-time data updates

⚠️ **Disclaimer:** For entertainment only!"""
        
        bot.reply_to(message, welcome_text, parse_mode='Markdown')
    except Exception as e:
        print(f"⚠️ Welcome error: {e}")

@bot.message_handler(commands=['predict', 'p'])
def predict_handler(message):
    try:
        if len(digits) < 50:
            bot.reply_to(message, "❌ Insufficient data. Loading...")
            if load_google_sheets_data():
                bot.reply_to(message, "✅ Data loaded. Try /predict again.")
            else:
                bot.reply_to(message, "❌ Failed to load data.")
            return
        
        # Get predictions
        ensemble, individual = get_ensemble_prediction()
        
        # Find top predictions
        sorted_indices = np.argsort(ensemble)[::-1]
        
        # Calculate confidence
        max_prob = ensemble[sorted_indices[0]]
        second_prob = ensemble[sorted_indices[1]] if len(sorted_indices) > 1 else 0
        strength = max_prob - second_prob
        
        if max_prob > 0.2 and strength > 0.08:
            confidence_level = "🔥 Very High"
        elif max_prob > 0.15 and strength > 0.05:
            confidence_level = "⭐ High"
        elif max_prob > 0.12:
            confidence_level = "💫 Medium"
        else:
            confidence_level = "📊 Low"
        
        # Format response
        response = "🎯 **KOLKATA FF PREDICTION**\n\n"
        response += f"🎲 **Confidence:** {confidence_level}\n\n"
        
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
            
            response += f"{indicator} **{idx}** - {prob:.1f}%\n"
        
        response += "\n📊 **All Probabilities:**\n"
        for d in range(10):
            prob_percent = ensemble[d] * 100
            indicator = "🔥" if prob_percent >= 15 else "⭐" if prob_percent >= 12 else "📊"
            response += f"{indicator} {d}: {prob_percent:.1f}%\n"
        
        # Add consensus
        top_digit = sorted_indices[0]
        agreement_count = sum(1 for pred in individual.values() if np.argmax(pred) == top_digit)
        consensus_percent = (agreement_count / len(individual)) * 100
        
        response += f"\n🤝 **Consensus:** {consensus_percent:.0f}% ({agreement_count}/{len(individual)})"
        response += f"\n📈 **Data:** {len(digits)} records"
        response += f"\n🕐 **Time:** {datetime.now().strftime('%H:%M:%S')}"
        
        bot.reply_to(message, response, parse_mode='Markdown')
        
    except Exception as e:
        print(f"⚠️ Predict error: {e}")
        bot.reply_to(message, "❌ Prediction failed. Try again.")

@bot.message_handler(commands=['stats'])
def stats_handler(message):
    try:
        all_stats = [S1_stats, S2_stats, S3_stats, S4_stats, S5_stats, M1_stats, M2_stats, M3_stats]
        
        response = "📊 **STRATEGY STATISTICS**\n\n"
        
        for stats in sorted(all_stats, key=lambda x: x['acc'], reverse=True):
            if stats['total'] > 0:
                acc = stats['acc']
                status = "🔥" if acc > 25 else "⭐" if acc > 20 else "💫" if acc > 15 else "📊"
                
                response += f"{status} **{stats['name']}**\n"
                response += f"   📈 Accuracy: {acc:.1f}% ({stats['ok']}/{stats['total']})\n"
                
                if stats['recent_acc']:
                    recent_10 = list(stats['recent_acc'])[-10:]
                    recent_acc = (sum(recent_10) / len(recent_10)) * 100
                    trend = "📈" if recent_acc > acc else "📉" if recent_acc < acc else "➡️"
                    response += f"   {trend} Recent: {recent_acc:.1f}%\n"
                
                response += "\n"
        
        # Overall statistics
        total_predictions = sum(s['total'] for s in all_stats)
        total_correct = sum(s['ok'] for s in all_stats)
        overall_accuracy = (total_correct / total_predictions * 100) if total_predictions > 0 else 0
        
        response += "📊 **OVERALL PERFORMANCE**\n"
        response += f"🎯 Combined Accuracy: {overall_accuracy:.1f}%\n"
        response += f"📈 Total Predictions: {total_predictions}\n"
        response += f"🎲 Total Records: {len(digits)}\n"
        response += f"🕐 Last Update: {datetime.now().strftime('%H:%M:%S')}"
        
        bot.reply_to(message, response, parse_mode='Markdown')
    except Exception as e:
        print(f"⚠️ Stats error: {e}")
        bot.reply_to(message, "❌ Stats retrieval failed.")

@bot.message_handler(commands=['analysis'])
def analysis_handler(message):
    try:
        if len(digits) < 50:
            bot.reply_to(message, "❌ Insufficient data for analysis.")
            return
        
        response = "🔍 **PATTERN ANALYSIS**\n\n"
        
        # Recent pattern analysis
        recent_20 = digits[-20:] if len(digits) >= 20 else digits
        recent_counts = Counter(recent_20)
        
        response += "📊 **Recent 20 Results:**\n"
        most_frequent = recent_counts.most_common(3)
        for digit, count in most_frequent:
            percentage = (count / len(recent_20)) * 100
            response += f"• {digit}: {count} times ({percentage:.1f}%)\n"
        
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
            response += f"{urgency} Digit {digit}: {gap} draws ago\n"
        
        # Entropy analysis
        entropy_recent = _calculate_entropy(recent_20)
        max_entropy = math.log2(10)
        entropy_percent = (entropy_recent / max_entropy) * 100
        
        response += f"\n🌀 **Randomness Level:** {entropy_percent:.1f}%\n"
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
            
            response += f"\n🔄 **Current Streak:** Digit {current_digit} appeared {streak} time(s)\n"
            
            if streak >= 3:
                response += "   ⚠️ Long streak - may break soon\n"
        
        response += f"\n🕐 **Analysis Time:** {datetime.now().strftime('%H:%M:%S')}"
        
        bot.reply_to(message, response, parse_mode='Markdown')
        
    except Exception as e:
        print(f"⚠️ Analysis error: {e}")
        bot.reply_to(message, "❌ Analysis failed.")

@bot.message_handler(commands=['reload'])
def reload_handler(message):
    try:
        bot.reply_to(message, "🔄 Reloading data...")
        success = load_google_sheets_data()
        
        if success:
            # Clear model cache to retrain
            global S1_model, S2_model, S3_model
            S1_model, S2_model, S3_model = None, None, None
            model_cache.clear()
            
            bot.reply_to(message, f"✅ Data reloaded! {len(digits)} records available.")
        else:
            bot.reply_to(message, "❌ Failed to reload data.")
    except Exception as e:
        print(f"⚠️ Reload error: {e}")
        bot.reply_to(message, "❌ Reload failed.")

# Handle any text message for basic prediction
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        text = message.text.lower()
        if any(word in text for word in ['predict', 'number', 'next', 'digit']):
            predict_handler(message)
        else:
            bot.reply_to(message, "Use /predict for predictions or /help for commands!")
    except Exception as e:
        print(f"⚠️ Message handler error: {e}")

# ======== PERFORMANCE TRACKING ========
def update_performance(actual_digit, predictions):
    """Update performance for all strategies"""
    try:
        if not (0 <= actual_digit <= 9):
            return
        
        strategy_map = {
            'S1': S1_stats, 'S2': S2_stats, 'S3': S3_stats, 'S4': S4_stats,
            'S5': S5_stats, 'M1': M1_stats, 'M2': M2_stats, 'M3': M3_stats
        }
        
        for strategy_name, pred in predictions.items():
            if strategy_name in strategy_map:
                stats = strategy_map[strategy_name]
                
                top_pred = np.argmax(pred)
                
                stats['total'] += 1
                if top_pred == actual_digit:
                    stats['ok'] += 1
                
                stats['acc'] = (stats['ok'] / stats['total']) * 100
                
                recent_correct = 1 if top_pred == actual_digit else 0
                stats['recent_acc'].append(recent_correct)
                
                if stats['acc'] > stats['best_acc']:
                    stats['best_acc'] = stats['acc']
                
                # Store confidence
                confidence = pred[actual_digit]
                stats['confidence_scores'].append(confidence)
        
    except Exception as e:
        print(f"⚠️ Performance update error: {e}")

# ======== MAIN EXECUTION ========
def run_flask():
    """Run Flask server"""
    try:
        port = int(os.getenv('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
    except Exception as e:
        print(f"⚠️ Flask error: {e}")

def main():
    """Main execution function"""
    print("🚀 Starting Enhanced Kolkata FF Bot...")
    
    # Initial data load
    print("📊 Loading initial data...")
    if load_google_sheets_data():
        print(f"✅ Initial data loaded: {len(digits)} records")
        
        # Pre-train models if enough data
        if len(digits) >= 150 and ML_AVAILABLE:
            print("🤖 Pre-training models...")
            def pretrain():
                train_random_forest()
                train_gradient_boosting()
                train_neural_network()
            
            threading.Thread(target=pretrain, daemon=True).start()
    else:
        print("⚠️ Initial data load failed")
    
    # Start Flask in separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("🌐 Flask server started")
    
    # Periodic data updates
    def periodic_update():
        while True:
            try:
                time.sleep(1800)  # Update every 30 minutes
                if load_google_sheets_data():
                    print("🔄 Periodic update completed")
            except Exception as e:
                print(f"⚠️ Periodic update error: {e}")
    
    update_thread = threading.Thread(target=periodic_update, daemon=True)
    update_thread.start()
    print("⏰ Periodic updates started")
    
    print("🤖 Bot started successfully!")
    print("📡 Starting Telegram bot polling...")
    
    # Start bot polling with error handling
    try:
        bot.infinity_polling(timeout=10, long_polling_timeout=5)
    except Exception as e:
        print(f"❌ Bot polling error: {e}")
        # Try to restart polling
        time.sleep(5)
        try:
            bot.infinity_polling(timeout=10, long_polling_timeout=5)
        except Exception as e2:
            print(f"❌ Failed to restart bot: {e2}")

if __name__ == "__main__":
    main()