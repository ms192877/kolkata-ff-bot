# ===================== main.py (Strategies + Auto Bet Planner + Clean UI) =====================
import telebot, os, re, time, json, threading, requests
import numpy as np
import pandas as pd
import threading
from datetime import datetime, date
from flask import Flask
from collections import Counter
from dataclasses import dataclass, field

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

# ======== CONFIG (kept as-is) ========
BOT_TOKEN = "8306210029:AAHl7sxAEEq0FT750MAThHrAioYyAbRI1oI"
ADMIN_CHAT_ID = None  # auto-filled from first /start if None
SPREADSHEET_ID = "10wI8T-NzqYsq6L73kPZ_bibuv2dw7xhQAmOr0msvk1A"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=csv"

# ======== GLOBALS ========
bot = telebot.TeleBot(BOT_TOKEN)
app = Flask('')


@app.route("/healthz")
def _healthz():
    return "ok"


digits, rounds_hist, dates_hist = [], [], []

# knobs
CALIBRATE_PROBS = False
TRAIN_RATIO = 0.9
S5_DEFAULT_THRESHOLD = 0.24

# ======== STORAGE (JSON-safe) ========
learning_storage_file = "learning_data.json"


def _atomic_write_json(path: str, obj: dict):
    import tempfile
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def save_learning_data():
    data = {
        "S1": S1_stats,
        "S2": S2_stats,
        "S4": S4_stats,
        "S5": S5_stats,
        "M1": M1_stats,
        "M2": M2_stats,
        "M3": M3_stats,
        "autobet": AUTO.save_state(),
        "last_update": datetime.now().isoformat(),
    }
    try:
        _atomic_write_json(learning_storage_file, data)
    except Exception as e:
        print(f"❌ Error saving learning data: {e}")


def load_learning_data():
    try:
        if os.path.exists(learning_storage_file) and os.path.getsize(
                learning_storage_file) > 0:
            try:
                with open(learning_storage_file, "r") as f:
                    data = json.load(f)
            except Exception:
                print("⚠️ learning_data.json corrupt — resetting this run.")
                data = {}
        else:
            data = {}
        for name, store in [
            ("S1", S1_stats),
            ("S2", S2_stats),
            ("S4", S4_stats),
            ("S5", S5_stats),
            ("M1", M1_stats),
            ("M2", M2_stats),
            ("M3", M3_stats),
        ]:
            if isinstance(data.get(name), dict): store.update(data[name])
        if "autobet" in data: AUTO.load_state(data["autobet"])
        print("✅ Learning + AutoBet state loaded")
    except Exception as e:
        print(f"❌ Error loading learning data: {e}")


# ======== UTILS ========
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


def _parse_date_flexible(s: str):
    if not s: return None
    s = s.strip()
    ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    DMY_S = re.compile(r"^\d{2}/\d{2}/\d{4}$")
    DMY_D = re.compile(r"^\d{2}-\d{2}-\d{4}$")
    MDY_S = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
    try:
        if ISO.match(s):
            dt = pd.to_datetime(s,
                                format="%Y-%m-%d",
                                dayfirst=False,
                                errors="coerce")
        elif DMY_S.match(s):
            dt = pd.to_datetime(s,
                                format="%d/%m/%Y",
                                dayfirst=True,
                                errors="coerce")
        elif DMY_D.match(s):
            dt = pd.to_datetime(s,
                                format="%d-%m-%Y",
                                dayfirst=True,
                                errors="coerce")
        elif MDY_S.match(s):
            dt = pd.to_datetime(s,
                                format="%m/%d/%Y",
                                dayfirst=False,
                                errors="coerce")
        else:
            dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        return dt.to_pydatetime().date() if pd.notna(dt) else None
    except:
        return None


def load_google_sheets_data():
    global digits, rounds_hist, dates_hist
    old_n = len(digits)
    digits, rounds_hist, dates_hist = [], [], []
    try:
        r = requests.get(CSV_URL, timeout=12)
        r.raise_for_status()
        lines = r.text.strip().split('\n')
        if not lines: return False
        headers = [c.strip().strip('"') for c in lines[0].split(',')]
        idx_digit = headers.index('Digit') if 'Digit' in headers else None
        idx_round = headers.index('Round') if 'Round' in headers else None
        idx_date = headers.index('Date') if 'Date' in headers else None
        for ln in lines[1:]:
            if not ln.strip(): continue
            cols = [c.strip().strip('"') for c in ln.split(',')]
            if idx_digit is None or len(
                    cols) <= idx_digit or not cols[idx_digit].isdigit():
                continue
            val = int(cols[idx_digit])
            if 0 <= val <= 9: digits.append(val)
            else: continue
            # round
            if idx_round is not None and len(cols) > idx_round:
                try:
                    rounds_hist.append(int(cols[idx_round]))
                except:
                    rounds_hist.append(None)
            else:
                rounds_hist.append(None)
            # date
            if idx_date is not None and len(
                    cols) > idx_date and cols[idx_date]:
                dates_hist.append(_parse_date_flexible(cols[idx_date]))
            else:
                dates_hist.append(None)
        print(f"✅ Loaded {len(digits)} rows (Δ {len(digits)-old_n})")
        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False


# ======== STATS HOLDERS ========
S1_stats = {'name': 'RF', 'total': 0, 'ok': 0, 'acc': 0.0, 'conf': []}
S2_stats = {'name': 'GB', 'total': 0, 'ok': 0, 'acc': 0.0, 'conf': []}
S4_stats = {'name': 'SelfTopK', 'total': 0, 'ok': 0, 'acc': 0.0, 'conf': []}
S5_stats = {'name': 'HighConf', 'total': 0, 'ok': 0, 'acc': 0.0, 'conf': []}
M1_stats = {
    'name': 'BayesDirichlet',
    'total': 0,
    'ok': 0,
    'acc': 0.0,
    'conf': []
}
M2_stats = {'name': 'HazardGap', 'total': 0, 'ok': 0, 'acc': 0.0, 'conf': []}
M3_stats = {'name': 'Residue', 'total': 0, 'ok': 0, 'acc': 0.0, 'conf': []}


# ======== FEATURES ========
def _make_features(series, rounds=None, dates=None):
    if len(series) < 10: return None
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
    # ewma counts
    acc = np.zeros(10)
    w = 1.0
    for v in reversed(series):
        acc[v] += w
        w *= 0.97
        if w < 1e-6: break
    acc = acc / acc.sum() if acc.sum() > 0 else np.ones(10) / 10.0
    feats.extend(list(acc))
    # streak
    streak = 1
    for i in range(len(series) - 2, -1, -1):
        if series[i] == series[-1]: streak += 1
        else: break
    feats.append(streak)
    # weekday one-hot
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
    # round one-hot (1..8)
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
    return feats


def _build_xy(series, rounds=None, dates=None):
    X, y = [], []
    for i in range(30, len(series) - 1):
        hist = series[:i]
        feats = _make_features(hist, rounds[:i] if rounds else None,
                               dates[:i] if dates else None)
        if feats is None: continue
        X.append(feats)
        y.append(series[i])
    return X, y


# ======== MODELS ========
S1_model, S2_model = None, None


def train_randomforest_full():
    if len(digits) < 80: return None
    X, y = _build_xy(digits, rounds_hist, dates_hist)
    if len(X) < 50: return None
    X_tr, X_va, y_tr, y_va = train_test_split(X,
                                              y,
                                              test_size=0.2,
                                              random_state=42,
                                              stratify=y)
    base = RandomForestClassifier(n_estimators=260,
                                  max_depth=18,
                                  min_samples_split=3,
                                  min_samples_leaf=1,
                                  class_weight='balanced',
                                  random_state=42,
                                  n_jobs=-1)
    model = CalibratedClassifierCV(
        base,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        method='sigmoid') if CALIBRATE_PROBS else base
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_va, model.predict(X_va))
    print(f"✅ RF val acc: {acc*100:.1f}%")
    return model


def train_gradientboosting_full():
    if len(digits) < 80: return None
    X, y = _build_xy(digits, rounds_hist, dates_hist)
    if len(X) < 50: return None
    X_tr, X_va, y_tr, y_va = train_test_split(X,
                                              y,
                                              test_size=0.2,
                                              random_state=42,
                                              stratify=y)
    base = GradientBoostingClassifier(n_estimators=300,
                                      learning_rate=0.06,
                                      max_depth=3,
                                      random_state=42)
    model = CalibratedClassifierCV(
        base,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        method='sigmoid') if CALIBRATE_PROBS else base
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_va, model.predict(X_va))
    print(f"✅ GB val acc: {acc*100:.1f}%")
    return model


def S1_predict():
    global S1_model
    if len(digits) < 20: return 0, 0.2, np.ones(10) / 10.0
    if S1_model is None: S1_model = train_randomforest_full()
    f = _make_features(digits, rounds_hist, dates_hist)
    if S1_model is None or f is None: return 0, 0.2, np.ones(10) / 10.0
    if hasattr(S1_model, "predict_proba"):
        pro = S1_model.predict_proba([f])[0]
        p = int(np.argmax(pro))
        c = float(pro[p])
        return p, min(0.95, c), pro
    p = int(S1_model.predict([f])[0])
    return p, 0.5, np.ones(10) / 10.0


def S2_predict():
    global S2_model
    if len(digits) < 20: return 0, 0.2, np.ones(10) / 10.0
    if S2_model is None: S2_model = train_gradientboosting_full()
    f = _make_features(digits, rounds_hist, dates_hist)
    if S2_model is None or f is None: return 0, 0.2, np.ones(10) / 10.0
    if hasattr(S2_model, "predict_proba"):
        pro = S2_model.predict_proba([f])[0]
        p = int(np.argmax(pro))
        c = float(pro[p])
        return p, min(0.95, c), pro
    p = int(S2_model.predict([f])[0])
    return p, 0.5, np.ones(10) / 10.0


# ----- S4 Self-Learner Top-k -----
class SelfLearner:

    def __init__(self):
        self.delay_gaps = {d: [] for d in range(10)}

    def train(self, series):
        self.delay_gaps = {d: [] for d in range(10)}
        last = {d: None for d in range(10)}
        for i, v in enumerate(series):
            if last[v] is not None: self.delay_gaps[v].append(i - last[v])
            last[v] = i

    def predict_topk(self, series, k=3):
        last = {d: None for d in range(10)}
        for i, v in enumerate(series):
            last[v] = i
        L = len(series)
        scores = np.zeros(10)
        for d in range(10):
            gaps = self.delay_gaps.get(d, [])
            if not gaps:
                scores[d] = 1.0
                continue
            arr = np.array(gaps)
            cur = (L - 1 - last[d]) if last[d] is not None else int(arr.mean())
            cur = max(1, int(cur))
            ge = (arr >= cur).sum()
            eq = (arr == cur).sum()
            scores[d] = (eq + 0.25) / (ge + 2.5)
        probs = _norm(scores)
        order = np.argsort(-probs).tolist()
        return order[:k], probs


SELF = SelfLearner()


def S4_predict():
    if len(digits) < 10: return [0, 1], [0.2, 0.2], np.ones(10) / 10.0
    topk, probs = SELF.predict_topk(digits, k=3)
    confs = [float(probs[p]) for p in topk]
    return topk, confs, probs


# ----- S5 High-Confidence -----
def S5_predict():
    if len(digits) < 10: return 0, 0.2
    topk, probs = SELF.predict_topk(digits, k=1)
    p = int(topk[0])
    c = float(probs[p])
    if c >= S5_DEFAULT_THRESHOLD: return p, c
    fb = max(set(digits[-20:]),
             key=digits[-20:].count) if len(digits) >= 20 else digits[-1]
    return int(fb), max(0.28, c)


# ======== NEW TOP-3: M1/M2/M3 ========
@dataclass
class StrategyBayesDirichlet:
    prior_scale: float = 1.0
    recent_window: int = 30
    sample_count: int = 200
    alpha_global: np.ndarray = field(default_factory=lambda: np.ones(10))

    def train(self, h):
        cnt = Counter(h)
        self.alpha_global = np.array(
            [self.prior_scale + cnt.get(d, 0) for d in range(10)], float)

    def predict(self, h):
        if not h:
            pr = np.ones(10) / 10.0
            return 0, 0.1, pr
        r = h[-self.recent_window:] if len(h) >= self.recent_window else h
        cnt_r = Counter(r)
        alpha = self.alpha_global + np.array(
            [cnt_r.get(d, 0) for d in range(10)], float)
        wins = np.zeros(10)
        for _ in range(self.sample_count):
            s = np.random.dirichlet(alpha)
            wins[np.argmax(s)] += 1
        pr = wins / wins.sum() if wins.sum() > 0 else np.ones(10) / 10.0
        p = int(np.argmax(pr))
        return p, float(pr[p]), pr


@dataclass
class StrategyHazardGap:
    alpha: float = 1.0
    delay_gaps: dict = field(
        default_factory=lambda: {d: []
                                 for d in range(10)})

    def train(self, h):
        self.delay_gaps = {d: [] for d in range(10)}
        last = {d: None for d in range(10)}
        for i, v in enumerate(h):
            if last[v] is not None: self.delay_gaps[v].append(i - last[v])
            last[v] = i

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
            if not gaps:
                scores[d] = 1.0
                continue
            arr = np.array(gaps)
            cur = (L - 1 - last[d]) if last[d] is not None else int(arr.mean())
            cur = max(1, int(cur))
            ge = (arr >= cur).sum()
            eq = (arr == cur).sum()
            scores[d] = (eq + self.alpha * 0.25) / (ge + self.alpha * 2.5)
        pr = _norm(scores)
        p = int(np.argmax(pr))
        return p, float(pr[p]), pr


@dataclass
class StrategyResidueBalancer:
    target_even: float = 0.5
    target_mod3: list = field(default_factory=lambda: [1 / 3] * 3)
    target_mod5: list = field(default_factory=lambda: [0.2] * 5)
    temp: float = 0.6

    def train(self, h):
        pass

    def predict(self, h):
        if not h:
            pr = np.ones(10) / 10.0
            return 0, 0.1, pr
        n = len(h)
        even = sum(1 for v in h if v % 2 == 0) / n
        cnt3 = [0, 0, 0]
        cnt5 = [0] * 5
        for v in h:
            cnt3[v % 3] += 1
            cnt5[v % 5] += 1
        mod3 = [c / n for c in cnt3]
        mod5 = [c / n for c in cnt5]
        desir = np.zeros(10)
        for d in range(10):
            new_even = (even * n + (1 if d % 2 == 0 else 0)) / (n + 1)
            sc_even = -abs(new_even - self.target_even)
            new3 = [c * n for c in mod3]
            new3[d % 3] += 1
            new3 = [c / (n + 1) for c in new3]
            sc_m3 = -sum(abs(new3[r] - self.target_mod3[r]) for r in range(3))
            new5 = [c * n for c in mod5]
            new5[d % 5] += 1
            new5 = [c / (n + 1) for c in new5]
            sc_m5 = -sum(abs(new5[r] - self.target_mod5[r]) for r in range(5))
            desir[d] = 0.4 * sc_even + 0.35 * sc_m3 + 0.25 * sc_m5
        pr = _softmax(desir, t=self.temp)
        p = int(np.argmax(pr))
        return p, float(pr[p]), pr


M1, M2, M3 = StrategyBayesDirichlet(), StrategyHazardGap(
), StrategyResidueBalancer()


# ======== EVAL / TRAIN ========
def _bump(stats, pred, actual, conf):
    stats['total'] += 1
    if pred == actual: stats['ok'] += 1
    stats['acc'] = (stats['ok'] / stats['total'] *
                    100.0) if stats['total'] else 0.0
    stats['conf'].append(conf)
    if len(stats['conf']) > 200: stats['conf'] = stats['conf'][-200:]


def calculate_accuracy_leakproof():
    global S1_model, S2_model
    if len(digits) < 80: return
    split = int(len(digits) * TRAIN_RATIO)
    train_d, test_d = digits[:split], digits[split:]

    S1_model = S2_model = None
    for s in (S1_stats, S2_stats, S4_stats, S5_stats, M1_stats, M2_stats,
              M3_stats):
        s.update({'total': 0, 'ok': 0, 'acc': 0.0, 'conf': []})

    SELF.train(train_d)
    M1.train(train_d)
    M2.train(train_d)
    M3.train(train_d)
    S1_model = train_randomforest_full()
    S2_model = train_gradientboosting_full()

    for i in range(len(test_d)):
        hist = train_d + test_d[:i]
        actual = test_d[i]
        p, c, _ = S1_predict()
        _bump(S1_stats, p, actual, c)
        p, c, _ = S2_predict()
        _bump(S2_stats, p, actual, c)
        topk, confs, _ = S4_predict()
        S4_stats['total'] += 1
        if actual in topk[:3]: S4_stats['ok'] += 1
        S4_stats['acc'] = (S4_stats['ok'] / S4_stats['total'] * 100.0)
        p, c = S5_predict()
        _bump(S5_stats, p, actual, c)
        p, c, _ = M1.predict(hist)
        _bump(M1_stats, p, actual, c)
        p, c, _ = M2.predict(hist)
        _bump(M2_stats, p, actual, c)
        p, c, _ = M3.predict(hist)
        _bump(M3_stats, p, actual, c)


def train_all_full():
    SELF.train(digits)
    M1.train(digits)
    M2.train(digits)
    M3.train(digits)
    global S1_model, S2_model
    S1_model = train_randomforest_full()
    S2_model = train_gradientboosting_full()
    save_learning_data()


# ======== ENSEMBLE ========
def ensemble_predict(history):
    _, _, pr1 = S1_predict()
    _, _, pr2 = S2_predict()
    _, _, pr4 = S4_predict()
    p5, c5 = S5_predict()
    pr5 = np.ones(10) / 10.0
    pr5[p5] = min(0.97, max(c5, 0.35))
    _, _, pm1 = M1.predict(history)
    _, _, pm2 = M2.predict(history)
    _, _, pm3 = M3.predict(history)
    w = {
        "S1": 0.18,
        "S2": 0.16,
        "S4": 0.10,
        "S5": 0.08,
        "M1": 0.20,
        "M2": 0.14,
        "M3": 0.14
    }
    mix = _norm(w["S1"] * pr1 + w["S2"] * pr2 + w["S4"] * pr4 + w["S5"] * pr5 +
                w["M1"] * pm1 + w["M2"] * pm2 + w["M3"] * pm3)
    pred = int(np.argmax(mix))
    conf = float(mix[pred])
    top3 = np.argsort(-mix)[:3].tolist()
    return pred, conf, top3, mix


# ======== AUTO BET PLANNER ========
class AutoBetPlanner:

    def __init__(self):
        # defaults as requested
        self.capital = 3800
        self.bet_per_digit = 10
        self.daily_risk_frac = 0.10  # 10% capital as daily risk cap
        self.daily_target = 400  # ~₹400 profit
        self.daily_stop_loss = None  # if None, computed from risk_frac
        self.risk_threshold = 0.20  # if round risk > 20% of daily risk OR conf<40% => SKIP
        self.enabled = False
        self.chat_id = None
        self._reset_day_state()

    def _reset_day_state(self):
        self.day = date.today()
        self.day_bet_total = 0
        self.day_win_total = 0
        self.day_net = 0
        self.round_log = []
        self.last_seen_n = 0

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
        }

    def load_state(self, d):
        try:
            self.capital = d.get("capital", self.capital)
            self.bet_per_digit = d.get("bet_per_digit", self.bet_per_digit)
            self.daily_risk_frac = d.get("daily_risk_frac",
                                         self.daily_risk_frac)
            self.daily_target = d.get("daily_target", self.daily_target)
            self.daily_stop_loss = d.get("daily_stop_loss",
                                         self.daily_stop_loss)
            self.risk_threshold = d.get("risk_threshold", self.risk_threshold)
            self.enabled = d.get("enabled", self.enabled)
        except Exception as e:
            print("AutoBet load_state err:", e)

    # --- core decision ---
    def select_digits(self):
        # use ensemble + strategy consensus to choose up to 3 digits
        _, _, sorted3, mix = ensemble_predict(digits)
        # consensus count (how many strategies pick this as top1)
        picks = []
        s1, _c, _p = S1_predict()[:2]
        s2, _c, _p = S2_predict()[:2]
        s5, _c = S5_predict()
        m1, _c, _p = M1.predict(digits)
        m2, _c, _p = M2.predict(digits)
        m3, _c, _p = M3.predict(digits)
        top_votes = Counter([s1, s2, s5, m1, m2, m3])
        # rule: keep digits with votes >=2 or ensemble prob >= 0.18
        for d in sorted3:
            if top_votes[d] >= 2 or mix[d] >= 0.18:
                picks.append(d)
        # enforce min1 max3
        if len(picks) == 0:
            picks = [sorted3[0]]
        if len(picks) > 3:
            # keep top-3 by ensemble prob
            picks = sorted(picks, key=lambda x: -mix[x])[:3]
        conf = float(sum(mix[d] for d in picks) / len(picks))
        return picks, conf, mix

    def risk_score(self, num_digits, conf):
        # cost risk against daily risk budget
        cost = num_digits * self.bet_per_digit
        risk_budget = self.daily_risk_budget()
        # normalize: how much of budget this bet consumes
        frac = cost / max(1.0, risk_budget)
        # lower conf -> higher risk penalty
        conf_penalty = max(
            0.0, 0.4 - conf) / 0.4  # 0 if conf>=0.4; up to 1 if conf==0
        return frac + 0.5 * conf_penalty, cost, risk_budget

    def recommend_text(self, picks, conf, risk_sc, cost):
        risk_tag = "Low ✅" if risk_sc < self.risk_threshold else "High ⚠️"
        rec = "PLAY" if risk_sc < self.risk_threshold and conf >= 0.40 else "SKIP ❌"
        return risk_tag, rec

    def handle_new_data(self):
        # call on auto_refresh when new rows added
        self._ensure_today()
        if not self.enabled or self.chat_id is None: return
        n = len(digits)
        if n <= self.last_seen_n: return
        # one new result likely means next round to predict
        self.last_seen_n = n

        # stop conditions
        if self.day_net >= self.daily_target:
            bot.send_message(
                self.chat_id,
                f"🛑 *Target Reached* — Net +₹{self.day_net}.\nAutoBet paused for today.",
                parse_mode='HTML')
            self.enabled = False
            save_learning_data()
            return
        if -self.day_net >= self.daily_risk_budget():
            bot.send_message(
                self.chat_id,
                f"🛑 *Stop-Loss Hit* — Net -₹{abs(self.day_net)}.\nAutoBet paused for today.",
                parse_mode='HTML')
            self.enabled = False
            save_learning_data()
            return

        # make suggestion for next round
        picks, conf, mix = self.select_digits()
        risk_sc, cost, budget = self.risk_score(len(picks), conf)
        risk_tag, rec = self.recommend_text(picks, conf, risk_sc, cost)

        msg = (
            f"📌 *Next Round Suggestion*\n"
            f"🎯 Digits: *{', '.join(map(str,picks))}*\n"
            f"💵 Bet: ₹{self.bet_per_digit} each → *₹{cost}*\n"
            f"Risk: {risk_tag}  |  Confidence: *{conf*100:.1f}%*\n\n"
            f"👉 *Recommended:* {rec}\n"
            f"— Budget today: ₹{budget:.0f} | Used: ₹{self.day_bet_total}\n")
        bot.send_message(self.chat_id, msg, parse_mode='HTML')

    def record_result(self, played_digits, actual_digit):
        # optional hook if you log wins (not auto since we don't place real bets)
        if not played_digits: return
        cost = len(played_digits) * self.bet_per_digit
        win = 0
        if actual_digit in played_digits:
            # Kolkata FF style 1:9 payout assumption for single digit
            # if played multiple digits, payout remains per winning digit
            win = 90  # ₹10 -> ₹90 net win; you can scale by bet_per_digit
            win = int((self.bet_per_digit / 10.0) * 90)
        self.day_bet_total += cost
        self.day_win_total += win
        self.day_net = self.day_win_total - self.day_bet_total
        self.round_log.append({
            "played": played_digits,
            "actual": actual_digit,
            "cost": cost,
            "win": win
        })


AUTO = AutoBetPlanner()


# ======== WEB MINI STATUS ========
@app.route('/')
def home():
    return f"""
    <h1>🤖 Kolkata FF — Auto Bet Planner</h1>
    <p>N={len(digits)} | Capital: ₹{AUTO.capital} | Bet/Digit: ₹{AUTO.bet_per_digit}</p>
    <p>🎯 Target: ₹{AUTO.daily_target} | Stop-loss: ₹{AUTO.daily_risk_budget():.0f}</p>
    <p>📈 Acc: RF {S1_stats['acc']:.1f}% | GB {S2_stats['acc']:.1f}% | S4 {S4_stats['acc']:.1f}% | S5 {S5_stats['acc']:.1f}% | M1 {M1_stats['acc']:.1f}% | M2 {M2_stats['acc']:.1f}% | M3 {M3_stats['acc']:.1f}%</p>
    <p>🟢 AutoBet: {"ON" if AUTO.enabled else "OFF"} | Risk-th: {AUTO.risk_threshold:.2f}</p>
    """


# ======== TELEGRAM UI ========
WELCOME = """🤖 <b>Kolkata FF Auto Bet Planner Ready!</b>

💰 <b>Capital:</b> ₹{cap}
🎯 <b>Daily Target:</b> ₹{tgt} | <b>Stop-loss:</b> ~₹{sl}
📊 <b>Bet/Digit:</b> ₹{bet} | <b>Max 3 digits/round</b>
⚖️ <b>Risk-threshold:</b> {risk:.2f}

Use:
• /autobet on — start auto suggestions
• /autobet off — stop
• /betsummary — see today summary
• /set_capital 3800
• /set_bet 10
• /set_risk 0.2
• /refresh — reload sheet & re-evaluate
• /accuracy — current accuracies
"""


@bot.message_handler(commands=['start'])
def start_cmd(m):
    global ADMIN_CHAT_ID
    if ADMIN_CHAT_ID is None: ADMIN_CHAT_ID = m.chat.id
    AUTO.chat_id = m.chat.id
    msg = WELCOME.format(cap=AUTO.capital,
                         tgt=AUTO.daily_target,
                         sl=int(AUTO.daily_risk_budget()),
                         bet=AUTO.bet_per_digit,
                         risk=AUTO.risk_threshold)
    bot.reply_to(m, msg, parse_mode='HTML')


@bot.message_handler(commands=['predict'])
def predict_cmd(m):

    if len(digits) < 5:
        bot.reply_to(m, "❌ Not enough data.")
        return
    # === Round headline (added) ===
    try:
        if 'rounds_hist' in globals() and rounds_hist:
            _valid_rounds = [r for r in rounds_hist if r is not None]
            _next_round = (max(_valid_rounds) + 1) if _valid_rounds else '?'
        else:
            _next_round = '?'
    except Exception:
        _next_round = '?'
    _round_headline = f"📢 *Round {_next_round} Prediction* 📢\n\n"

    if 'rounds_hist' in globals() and rounds_hist:
        _valid_rounds = [r for r in rounds_hist if r is not None]
        _next_round = (max(_valid_rounds) + 1) if _valid_rounds else '?'
    else:
        _next_round = '?'
    if len(digits) < 5:
        bot.reply_to(m, "❌ Not enough data.")
        return
    p1, c1, _ = S1_predict()
    p2, c2, _ = S2_predict()
    topk, confs, _ = S4_predict()
    p5, c5 = S5_predict()
    pm1, cm1, _ = M1.predict(digits)
    pm2, cm2, _ = M2.predict(digits)
    pm3, cm3, _ = M3.predict(digits)
    pe, ce, top3e, _ = ensemble_predict(digits)

    s4_text = "".join([
        f"  {i+1}. *{p}* ({c*100:.1f}%)\n"
        for i, (p, c) in enumerate(zip(topk, confs))
    ])
    msg = _round_headline + f"""🎯 *Predictions*)

    S1 RF → *{p1}* ({c1*100:.1f}%)
    S2 GB → *{p2}* ({c2*100:.1f}%)

    S4 Self-Topk:
    {s4_text}S5 High-Conf → *{p5}* ({c5*100:.1f}%)

    M1 Bayes → *{pm1}* ({cm1*100:.1f}%)
    M2 Hazard → *{pm2}* ({cm2*100:.1f}%)
    M3 Residue → *{pm3}* ({cm3*100:.1f}%)

    🏆 Ensemble → *{pe}* | Conf: {ce*100:.1f}% 
    Top-3: {top3e}
    ⏰ {datetime.now().strftime('%H:%M:%S')} | N={len(digits)}
    """
    bot.reply_to(m, msg, parse_mode='HTML')


@bot.message_handler(commands=['accuracy'])
def acc_cmd(m):
    msg = f"""📊 *Accuracies (Leak-Proof 90/10)*

S1 RF: {S1_stats['acc']:.2f}%
S2 GB: {S2_stats['acc']:.2f}%
S4 Self-Topk: {S4_stats['acc']:.2f}%
S5 High-Conf: {S5_stats['acc']:.2f}%

M1 Bayes: {M1_stats['acc']:.2f}%
M2 Hazard: {M2_stats['acc']:.2f}%
M3 Residue: {M3_stats['acc']:.2f}%
"""
    bot.reply_to(m, msg, parse_mode='HTML')


@bot.message_handler(commands=['refresh'])
def refresh_cmd(m):
    ok = load_google_sheets_data()
    if ok:
        SELF.train(digits)
        M1.train(digits)
        M2.train(digits)
        M3.train(digits)
        calculate_accuracy_leakproof()
        save_learning_data()
        bot.reply_to(
            m, f"✅ Data refreshed. N={len(digits)}\n"
            f"S1:{S1_stats['acc']:.2f}% S2:{S2_stats['acc']:.2f}% S4:{S4_stats['acc']:.2f}% S5:{S5_stats['acc']:.2f}% "
            f"M1:{M1_stats['acc']:.2f}% M2:{M2_stats['acc']:.2f}% M3:{M3_stats['acc']:.2f}%",
            parse_mode='HTML')
        # also nudge AutoBet to suggest next
        AUTO.handle_new_data()
    else:
        bot.reply_to(m, "❌ Failed to refresh data.")


@bot.message_handler(commands=['autobet'])
def autobet_cmd(m):
    parts = m.text.strip().split()
    if len(parts) < 2 or parts[1] not in ("on", "off"):
        bot.reply_to(m, "Usage: /autobet on | off")
        return
    AUTO.chat_id = m.chat.id
    if parts[1] == "on":
        AUTO.enabled = True
        AUTO._ensure_today()
        bot.reply_to(
            m,
            f"🟢 AutoBet *ON* — Target ₹{AUTO.daily_target}, Stop-loss ~₹{int(AUTO.daily_risk_budget())}\n"
            f"Bet/Digit ₹{AUTO.bet_per_digit}, Risk-th {AUTO.risk_threshold:.2f}",
            parse_mode='HTML')
        # immediate suggestion
        AUTO.handle_new_data()
    else:
        AUTO.enabled = False
        bot.reply_to(m, "🔴 AutoBet *OFF*", parse_mode='HTML')
    save_learning_data()


@bot.message_handler(commands=['betsummary'])
def betsummary_cmd(m):
    AUTO._ensure_today()
    msg = (
        f"📊 *Daily Summary*\n"
        f"Total Bets: ₹{AUTO.day_bet_total}\n"
        f"Wins: ₹{AUTO.day_win_total}\n"
        f"Net: {'+' if AUTO.day_net>=0 else ''}₹{AUTO.day_net}\n"
        f"Target Hit: {'Yes' if AUTO.day_net>=AUTO.daily_target else 'No'}\n"
        f"Stop-loss Triggered: {'Yes' if -AUTO.day_net>=AUTO.daily_risk_budget() else 'No'}"
    )
    bot.reply_to(m, msg, parse_mode='HTML')


@bot.message_handler(commands=['set_capital'])
def set_capital_cmd(m):
    try:
        val = float(m.text.strip().split()[1])
        AUTO.capital = max(100.0, val)
        bot.reply_to(
            m,
            f"✅ Capital set to ₹{int(AUTO.capital)}\nStop-loss budget ~₹{int(AUTO.daily_risk_budget())}"
        )
        save_learning_data()
    except Exception:
        bot.reply_to(m, "Usage: /set_capital 3800")


@bot.message_handler(commands=['set_bet'])
def set_bet_cmd(m):
    try:
        val = float(m.text.strip().split()[1])
        AUTO.bet_per_digit = max(1.0, val)
        bot.reply_to(m, f"✅ Bet per digit set to ₹{int(AUTO.bet_per_digit)}")
        save_learning_data()
    except Exception:
        bot.reply_to(m, "Usage: /set_bet 10")


@bot.message_handler(commands=['set_risk'])
def set_risk_cmd(m):
    try:
        val = float(m.text.strip().split()[1])
        AUTO.risk_threshold = min(0.8, max(0.05, val))
        bot.reply_to(
            m, f"✅ Risk-threshold set to {AUTO.risk_threshold:.2f}\n"
            f"(lower = stricter SKIP; higher = more PLAY)")
        save_learning_data()
    except Exception:
        bot.reply_to(m, "Usage: /set_risk 0.2")


# ======== BACKGROUND TASKS ========
def auto_refresh():
    while True:
        time.sleep(300)  # 5 min
        old = len(digits)
        if load_google_sheets_data():
            if len(digits) > old:
                # retrain light modules
                SELF.train(digits)
                M1.train(digits)
                M2.train(digits)
                M3.train(digits)
                calculate_accuracy_leakproof()
                save_learning_data()
                print(f"[Auto] New data: {len(digits)} (+{len(digits)-old})")
                AUTO.handle_new_data()
            else:
                print(f"[Auto] Checked: {len(digits)}")


def keep_alive():

    def run():
        app.run(host="0.0.0.0", port=3000)

    t = threading.Thread(target=run)
    t.daemon = True
    t.start()


def self_ping():
    url = os.getenv('REPL_URL') or (f"https://{os.getenv('REPL_SLUG')}.{os.getenv('REPL_OWNER')}.repl.co" if os.getenv('REPL_SLUG') and os.getenv('REPL_OWNER') else None)
    if not url:
        print('⚠️ Set REPL_URL in Secrets or ensure REPL_SLUG/REPL_OWNER are available')
        return
    while True:
        try:
            requests.get(url + "/healthz", timeout=10)
            print(f"🔄 Self-ping {url}/healthz")
        except Exception as e:
            print("Ping error:", e)
        time.sleep(300)


def initialize():
    print("🚀 Starting Accuracy-Boosted Bot (with Auto Bet Planner)")
    load_learning_data()
    load_google_sheets_data()
    SELF.train(digits)
    M1.train(digits)
    M2.train(digits)
    M3.train(digits)
    calculate_accuracy_leakproof()
    keep_alive()
    threading.Thread(target=self_ping, daemon=True).start()
    threading.Thread(target=auto_refresh, daemon=True).start()


def run_bot():
    while True:
        try:
            print("🤖 Bot polling started...")
            bot.polling(none_stop=True, timeout=10)
        except Exception as e:
            print(f"❌ Bot crashed: {e}")
            print("🔄 Restarting in 5 seconds...")
            time.sleep(5)


if __name__ == "__main__":
    initialize()
    run_bot()

