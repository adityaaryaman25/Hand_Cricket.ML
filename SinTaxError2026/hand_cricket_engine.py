"""
hand_cricket_engine.py
======================
Modular Hand Cricket ML Engine & Strategy Engine with Threat-Aware Risk Management,
Multi-Level Difficulty Control, and Detailed Diagnostic / Test Mode Support.

Key Components:
---------------
1. MovePredictor:
   - Maintains persistent batting histories per player_id across sessions.
   - Hybrid sub-models: Frequency, Recency Decay, Sequential Markov Chains (1 & 2),
     Context Conditionals, and Reaction/Delta models with Laplace/Bayesian smoothing.
   - Dynamic difficulty filtering (Easy, Medium, Hard, Expert).

2. StrategyEngine (Threat-Aware Risk Management System):
   - Threat Avoidance & Anti-Cycling 6-ball lookback window.
   - Multi-Wicket Awareness: Adjusts risk threshold dynamically based on wickets remaining.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = 1
CONTEXT_KEYS = ("session_id", "game_id", "innings", "role", "score", "target", "wickets_left", "max_wickets")

PATTERN_FAMILY = {
    "uniform": "none",
    "freq": "frequency_bias",
    "recency": "recent_frequency_bias",
    "role_freq": "role_dependent_bias",
    "situation_freq": "game_state_dependent_bias",
    "markov1": "sequential",
    "markov2": "sequential",
    "delta_prev": "sequential",
    "reaction": "reacts_to_my_previous_move",
    "delta_reaction": "reacts_to_my_previous_move",
}


@dataclass
class Config:
    legal_numbers: Tuple[int, ...] = tuple(range(1, 11))  # Numbers 1 through 10
    alpha: float = 1.0            # Dirichlet/Laplace smoothing parameter
    beta: float = 3.0             # Backoff prior strength for conditional models
    recency_half_life: float = 20.0
    share: float = 0.02           # Fixed-share switching rate
    eta: float = 1.0              # Learning rate for Bayesian mixture update
    floor: float = 0.01           # Uniform probability floor
    stat_half_life: float = 60.0  # Horizon for statistics tracking
    min_obs: int = 15             # Minimum observations needed for reliability
    z_threshold: float = 2.0      # Z-score threshold for pattern significance

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Config:
        d = dict(d)
        d["legal_numbers"] = tuple(d["legal_numbers"])
        return cls(**d)


class RunStat:
    """Tracks exponentially decayed statistics (mean, variance, ESS, Z-score)."""

    def __init__(self, half_life: float):
        self.g = 0.5 ** (1.0 / half_life)
        self.n = 0
        self.total = 0.0
        self.s0 = self.s1 = self.s2 = self.q = 0.0

    def add(self, x: float) -> None:
        g = self.g
        self.n += 1
        self.total += x
        self.s0 = self.s0 * g + 1.0
        self.s1 = self.s1 * g + x
        self.s2 = self.s2 * g + x * x
        self.q = self.q * g * g + 1.0

    def mean(self) -> float:
        return self.s1 / self.s0 if self.s0 else 0.0

    def lifetime_mean(self) -> float:
        return self.total / self.n if self.n else 0.0

    def ess(self) -> float:
        return self.s0 * self.s0 / self.q if self.q else 0.0

    def z(self) -> float:
        if self.n < 2:
            return 0.0
        m = self.mean()
        var = max(self.s2 / self.s0 - m * m, 1e-12)
        return m / math.sqrt(var / self.ess())


class Model:
    name = "model"

    def predict(self, ctx: dict, preds: dict) -> List[float]:
        raise NotImplementedError

    def observe(self, k: int, ctx: dict) -> None:
        pass

    def support(self, ctx: dict) -> float:
        return 0.0


class Uniform(Model):
    name = "uniform"

    def __init__(self, K: int):
        self.p = [1.0 / K] * K

    def predict(self, ctx, preds):
        return self.p


class Freq(Model):
    name = "freq"

    def __init__(self, K: int, alpha: float):
        self.K, self.alpha = K, alpha
        self.c = [0] * K
        self.n = 0

    def predict(self, ctx, preds):
        d = self.n + self.K * self.alpha
        return [(x + self.alpha) / d for x in self.c]

    def observe(self, k, ctx):
        self.c[k] += 1
        self.n += 1

    def support(self, ctx):
        return self.n


class Recency(Model):
    name = "recency"

    def __init__(self, K: int, alpha: float, half_life: float):
        self.K, self.alpha = K, alpha
        self.g = 0.5 ** (1.0 / half_life)
        self.c = [0.0] * K

    def predict(self, ctx, preds):
        d = sum(self.c) + self.K * self.alpha
        return [(x + self.alpha) / d for x in self.c]

    def observe(self, k, ctx):
        self.c = [x * self.g for x in self.c]
        self.c[k] += 1.0

    def support(self, ctx):
        return sum(self.c)


class Cond(Model):
    def __init__(self, name, K, key_fn, backoff, beta):
        self.name, self.K, self.key_fn = name, K, key_fn
        self.backoff, self.beta = backoff, beta
        self.table: Dict[Any, List[int]] = {}

    def predict(self, ctx, preds):
        base = preds[self.backoff]
        key = self.key_fn(ctx)
        row = self.table.get(key) if key is not None else None
        if not row:
            return base
        d = sum(row) + self.beta
        return [(row[i] + self.beta * base[i]) / d for i in range(self.K)]

    def observe(self, k, ctx):
        key = self.key_fn(ctx)
        if key is not None:
            self.table.setdefault(key, [0] * self.K)[k] += 1

    def support(self, ctx):
        key = self.key_fn(ctx)
        row = self.table.get(key) if key is not None else None
        return sum(row) if row else 0


class Delta(Model):
    def __init__(self, name, K, ref_fn, beta, fallback="freq"):
        self.name, self.K, self.ref_fn = name, K, ref_fn
        self.beta, self.fallback = beta, fallback
        self.c = [0] * K
        self.n = 0

    def predict(self, ctx, preds):
        ref = self.ref_fn(ctx)
        if ref is None:
            return preds[self.fallback]
        a, d = self.beta / self.K, self.n + self.beta
        return [(self.c[(i - ref) % self.K] + a) / d for i in range(self.K)]

    def observe(self, k, ctx):
        ref = self.ref_fn(ctx)
        if ref is not None:
            self.c[(k - ref) % self.K] += 1
            self.n += 1

    def support(self, ctx):
        return self.n if self.ref_fn(ctx) is not None else 0


def build_models(cfg: Config) -> List[Model]:
    K, b = len(cfg.legal_numbers), cfg.beta
    return [
        Uniform(K),
        Freq(K, cfg.alpha),
        Recency(K, cfg.alpha, cfg.recency_half_life),
        Cond("role_freq", K, lambda c: c.get("role"), "freq", b),
        Cond("situation_freq", K, lambda c: c.get("situation"), "role_freq", b),
        Cond("markov1", K, lambda c: c.get("prev1"), "freq", b),
        Cond("markov2", K, lambda c: c.get("prev12"), "markov1", b),
        Cond("reaction", K, lambda c: c.get("my_prev"), "freq", b),
        Delta("delta_prev", K, lambda c: c.get("prev1"), b),
        Delta("delta_reaction", K, lambda c: c.get("my_prev"), b),
    ]


class PlayerProfile:
    def __init__(self, player_id: str, config: Optional[Config] = None):
        self.player_id = str(player_id)
        self.cfg = config or Config()
        self.legal = list(self.cfg.legal_numbers)
        self.K = len(self.legal)
        self._index = {n: i for i, n in enumerate(self.legal)}
        self.records: List[dict] = []
        self.models = build_models(self.cfg)
        self.names = [m.name for m in self.models]
        self.weights = {n: 1.0 / len(self.names) for n in self.names}
        H = self.cfg.stat_half_life
        self.loss = {n: RunStat(H) for n in self.names + ["mixture"]}
        self.brier = RunStat(H)
        self.edge_u = RunStat(H)
        self.edge_f = RunStat(H)
        self.edge_rf = RunStat(H)
        self.sessions: Dict[str, dict] = {}

    def _idx(self, v: Any, what: str) -> int:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            raise ValueError(f"{what}={v!r} is not a legal number {self.legal}")
        if isinstance(v, bool) or iv != v or iv not in self._index:
            raise ValueError(f"{what}={v!r} is not a legal number {self.legal}")
        return self._index[iv]

    @staticmethod
    def _clean(context: Optional[dict]) -> dict:
        return {k: context[k] for k in CONTEXT_KEYS
                if context and context.get(k) is not None}

    @staticmethod
    def _stream(c: dict) -> tuple:
        return (c.get("session_id"), c.get("game_id"), c.get("innings"))

    def derive_ctx(self, context: Optional[dict]) -> dict:
        c = self._clean(context)
        ctx = dict(c)
        recs, stream = self.records, self._stream(c)
        prev1 = prev2 = my_prev = None
        if recs and self._stream(recs[-1]["ctx"]) == stream:
            prev1 = self._index[recs[-1]["move"]]
            if recs[-1]["my_move"] is not None:
                my_prev = self._index[recs[-1]["my_move"]]
            if len(recs) > 1 and self._stream(recs[-2]["ctx"]) == stream:
                prev2 = self._index[recs[-2]["move"]]
        ctx["prev1"], ctx["my_prev"] = prev1, my_prev
        ctx["prev12"] = (prev2, prev1) if prev1 is not None and prev2 is not None else None
        bucket = None
        if "score" in c and "target" in c and c["target"] is not None:
            rem = c["target"] - c["score"]
            bucket = "close" if rem <= 10 else "mid" if rem <= 25 else "far"
        ctx["situation"] = (c["role"], c.get("innings"), bucket) if "role" in c else None
        return ctx

    def _all_predictions(self, ctx: dict) -> Dict[str, List[float]]:
        preds: Dict[str, List[float]] = {}
        for m in self.models:
            preds[m.name] = m.predict(ctx, preds)
        return preds

    def _mix(self, preds: Dict[str, List[float]], active_models: Optional[List[str]] = None) -> List[float]:
        K = self.K
        mix = [0.0] * K

        target_weights = self.weights
        if active_models:
            sub_w = {n: self.weights[n] for n in active_models if n in self.weights}
            s = sum(sub_w.values())
            if s > 0:
                target_weights = {n: sub_w[n] / s for n in sub_w}
            else:
                target_weights = {n: 1.0 / len(active_models) for n in active_models}

        for name, w in target_weights.items():
            if name in preds:
                p = preds[name]
                for i in range(K):
                    mix[i] += w * p[i]

        f = self.cfg.floor
        mix = [(1.0 - f) * x + f / K for x in mix]
        s = sum(mix)
        return [x / s for x in mix]

    def _assess(self) -> Tuple[str, dict]:
        cfg, n = self.cfg, len(self.records)
        edge = self.edge_u.mean() if n else 0.0
        z = self.edge_u.z() if n else 0.0
        if n < cfg.min_obs:
            status = "insufficient_data"
        elif z < cfg.z_threshold or edge < 0.01:
            status = "no_reliable_pattern"
        elif edge >= 0.25 and n >= 100:
            status = "high"
        elif edge >= 0.08 and n >= 40:
            status = "moderate"
        else:
            status = "low"
        ev = {
            "edge_vs_uniform_nats_per_obs": round(edge, 5),
            "edge_vs_frequency_baseline_nats_per_obs": round(self.edge_f.mean() if n else 0.0, 5),
            "z_vs_uniform": round(z, 3),
            "brier_score_recent": round(self.brier.mean() if n else 0.0, 4),
            "effective_sample_size": round(self.edge_u.ess(), 1),
        }
        return status, ev

    def predict(self, context: Optional[dict] = None, difficulty: str = "expert") -> dict:
        diff = difficulty.lower()
        ctx = self.derive_ctx(context)
        preds = self._all_predictions(ctx)

        # Filter active prediction models based on selected difficulty
        if diff == "easy":
            active = ["uniform"]
        elif diff == "medium":
            active = ["freq"]
        elif diff == "hard":
            active = ["freq", "recency", "markov1", "markov2"]
        else:  # Expert
            active = self.names

        mix = self._mix(preds, active_models=active)
        n = len(self.records)
        status, ev = self._assess()
        leading = max(self.weights, key=self.weights.get)
        support = self.models[self.names.index(leading)].support(ctx)
        mx = max(mix)
        top = [self.legal[i] for i in range(self.K) if mix[i] >= mx - 1e-9]
        H = -sum(p * math.log(p) for p in mix) / math.log(self.K)
        edge_ok = status in ("low", "moderate", "high")
        drift = (n >= self.cfg.min_obs and self.edge_rf.z() > self.cfg.z_threshold
                 and self.edge_rf.mean() > 0.03)

        lim: List[str] = []
        if n < self.cfg.min_obs:
            lim.append(f"Only {n} observations (< {self.cfg.min_obs}); probability output is near uniform.")
        elif status == "no_reliable_pattern":
            lim.append("No statistically reliable pattern detected; treat opponent choices as random noise.")
        elif status == "low":
            lim.append("Pattern evidence is weak; do not over-index on the top number.")
        if drift:
            lim.append("Recent behavior deviates from long-term baseline (strategy shift detected).")

        pattern = PATTERN_FAMILY[leading] if edge_ok else ("none_detected" if n >= self.cfg.min_obs else "unknown")
        top_w = sorted(self.weights.items(), key=lambda kv: -kv[1])[:4]

        probs_dict = {str(self.legal[i]): round(mix[i], 4) for i in range(self.K)}
        diff_val = round(1.0 - sum(probs_dict.values()), 4)
        if diff_val != 0:
            top_k = str(top[0])
            probs_dict[top_k] = round(probs_dict[top_k] + diff_val, 4)

        return {
            "player_id": self.player_id,
            "predicted_probabilities": probs_dict,
            "most_likely_numbers": top,
            "most_likely_is_reliable": edge_ok,
            "observations": n,
            "supporting_observations": int(support),
            "confidence_status": status,
            "difficulty_level": diff,
            "leading_model": leading,
            "model_weights": {k: round(v, 4) for k, v in top_w},
            "uncertainty": {"entropy_normalized": round(H, 4), "max_probability": round(mx, 4)},
            "evidence": ev,
            "diagnostics": {"pattern_type": pattern, "possible_strategy_change": drift,
                            "sessions_seen": len(self.sessions)},
            "limitations": lim,
        }

    def _ingest(self, move: int, my_move: Optional[int], c: dict, ts: float) -> dict:
        k = self._idx(move, "observed_move")
        if my_move is not None:
            self._idx(my_move, "my_move")
        ctx = self.derive_ctx(c)
        preds = self._all_predictions(ctx)
        mix = self._mix(preds)

        lls = {n: -math.log(max(preds[n][k], 1e-12)) for n in self.names}
        ll_mix = -math.log(max(mix[k], 1e-12))
        brier_step = sum((mix[i] - (1.0 if i == k else 0.0)) ** 2 for i in range(self.K))

        for n in self.names:
            self.loss[n].add(lls[n])
        self.loss["mixture"].add(ll_mix)
        self.brier.add(brier_step)
        self.edge_u.add(lls["uniform"] - ll_mix)
        self.edge_f.add(lls["freq"] - ll_mix)
        self.edge_rf.add(lls["freq"] - lls["recency"])

        s = self.sessions.setdefault(str(c.get("session_id", "default")),
                                     {"n": 0, "mixture_log_loss_sum": 0.0, "uniform_log_loss_sum": 0.0})
        s["n"] += 1
        s["mixture_log_loss_sum"] += ll_mix
        s["uniform_log_loss_sum"] += lls["uniform"]

        raw = {n: self.weights[n] * math.exp(-self.cfg.eta * lls[n]) for n in self.names}
        tot, M, a = sum(raw.values()), len(self.names), self.cfg.share
        self.weights = {n: (1 - a) * raw[n] / tot + a / M for n in self.names}

        for m in self.models:
            m.observe(k, ctx)

        self.records.append({"move": int(move), "my_move": None if my_move is None else int(my_move),
                             "ctx": c, "ts": ts})
        return {"observations": len(self.records), "probability_assigned": mix[k],
                "log_loss": ll_mix, "brier_score": brier_step}

    def update(self, observed_move: int, context: Optional[dict] = None,
               my_move: Optional[int] = None) -> dict:
        return self._ingest(observed_move, my_move, self._clean(context), round(time.time(), 3))


class PlayerStore:
    def __init__(self, directory: str):
        self.dir = directory
        os.makedirs(directory, exist_ok=True)

    def path(self, player_id: str) -> str:
        pid = str(player_id)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", pid).lstrip(".") or "_"
        if safe != pid:
            safe += "-" + hashlib.sha1(pid.encode()).hexdigest()[:8]
        return os.path.join(self.dir, safe + ".json")

    def load(self, player_id: str, config: Optional[Config] = None) -> Optional[PlayerProfile]:
        p = self.path(player_id)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        prof = PlayerProfile(data["player_id"], Config.from_dict(data["config"]))
        for r in data["records"]:
            prof._ingest(r["move"], r.get("my_move"), r.get("ctx", {}), r.get("ts", 0.0))
        return prof

    def save(self, profile: PlayerProfile) -> None:
        p = self.path(profile.player_id)
        data = {
            "schema_version": SCHEMA_VERSION,
            "player_id": profile.player_id,
            "config": profile.cfg.to_dict(),
            "records": profile.records
        }
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, p)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise


class PopulationStore(PlayerStore):
    def __init__(self, directory: str):
        super().__init__(directory)
        self.global_path = os.path.join(self.dir, "_global_population_prior.json")

    def get_global_weights(self, default_names: List[str]) -> Dict[str, float]:
        if not os.path.exists(self.global_path):
            return {n: 1.0 / len(default_names) for n in default_names}
        try:
            with open(self.global_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("global_weights", {n: 1.0 / len(default_names) for n in default_names})
        except Exception:
            return {n: 1.0 / len(default_names) for n in default_names}


class MovePredictor:
    def __init__(self, store_dir: Optional[str] = "player_profiles",
                 config: Optional[Config] = None, autosave: bool = True):
        self.cfg = config or Config(legal_numbers=tuple(range(1, 11)))
        self.store = PopulationStore(store_dir) if store_dir else None
        self.autosave = autosave
        self._profiles: Dict[str, PlayerProfile] = {}

    def profile(self, player_id: str) -> PlayerProfile:
        pid = str(player_id)
        if pid not in self._profiles:
            prof = self.store.load(pid, self.cfg) if self.store else None
            if prof is None:
                prof = PlayerProfile(pid, self.cfg)
                if self.store:
                    global_w = self.store.get_global_weights(prof.names)
                    prof.weights = dict(global_w)
            self._profiles[pid] = prof
        return self._profiles[pid]

    def predict(self, player_id: str, context: Optional[dict] = None, difficulty: str = "expert") -> dict:
        return self.profile(player_id).predict(context, difficulty=difficulty)

    def update(self, player_id: str, observed_move: int, context: Optional[dict] = None,
               my_move: Optional[int] = None) -> dict:
        info = self.profile(player_id).update(observed_move, context, my_move)
        if self.store and self.autosave:
            self.save(player_id)
        return {k: info[k] for k in ("observations", "probability_assigned", "log_loss")}

    def save(self, player_id: Optional[str] = None) -> None:
        if not self.store:
            return
        ids = [str(player_id)] if player_id is not None else list(self._profiles)
        for pid in ids:
            self.store.save(self._profiles[pid])


# --------------------------------------------------------------------------- Strategy Engine (Risk Management)
class StrategyEngine:
    """
    Threat-Aware Risk-Management Strategy Engine.
    Prevents bowler pattern trap, avoids dangerous numbers, and uses full 1-10 range unpredictably.
    """

    def __init__(self, temperature: float = 1.0, exploration_floor: float = 0.20):
        self.temperature = temperature
        self.exploration_floor = exploration_floor
        self.ai_move_history: List[int] = []

    def reset_history(self) -> None:
        self.ai_move_history.clear()

    def select_ai_move(self, role: str, probs: Dict[str, float], context: dict, difficulty: str = "expert") -> int:
        legal_moves = list(range(1, 11))
        diff = difficulty.lower()

        # EASY difficulty: Completely random selection
        if diff == "easy":
            selected = random.choice(legal_moves)
            self.ai_move_history.append(selected)
            if len(self.ai_move_history) > 6:
                self.ai_move_history.pop(0)
            return selected

        temp = 1.8 if diff == "medium" else self.temperature
        base_exp_floor = 0.30 if diff == "medium" else (0.20 if diff == "hard" else self.exploration_floor)

        if role == "bowl":
            # AI Bowling Strategy: Weighted selection based on opponent's predicted batting probabilities
            opp_probs = [float(probs.get(str(m), 0.1)) for m in legal_moves]
            weights = []
            for idx, m in enumerate(legal_moves):
                p_opp = opp_probs[idx]
                pen = 1.0
                for pos, prev_m in enumerate(reversed(self.ai_move_history)):
                    if prev_m == m:
                        pen *= (1.0 - (0.85 ** (pos + 1)))
                weights.append(max(p_opp * pen, 1e-6))

            tot = sum(weights)
            norm_w = [w / tot for w in weights] if tot > 0 else [1.0 / 10] * 10
            final_w = [(1.0 - base_exp_floor) * nw + (base_exp_floor / 10.0) for nw in norm_w]
            selected = random.choices(legal_moves, weights=final_w, k=1)[0]

        else:
            # AI Batting Strategy: Threat Avoidance & Unpredictable Safe Selection
            score = context.get("score", 0)
            target = context.get("target")

            p_outs = [float(probs.get(str(m), 0.1)) for m in legal_moves]
            max_p_out = max(p_outs)

            weights = []
            for idx, m in enumerate(legal_moves):
                p_out = p_outs[idx]

                # Flattened run score utility: Sub-linear scale so AI uses ALL 1-10 numbers with rich variation
                if target is not None:
                    runs_needed = max(target - score, 1)
                    run_reward = min(m, runs_needed) / 5.0
                else:
                    run_reward = (m ** 0.5) / 2.0  # Mild preference for higher runs without extreme gaps

                # Convert utility to base weight
                base_w = math.exp(run_reward / temp)

                # Strict Threat Avoidance:
                # If bowler is predicted/spamming choices (p_out elevated), set safety multiplier to 0.0
                if p_out >= 0.25 or (max_p_out >= 0.18 and p_out >= max_p_out - 0.03):
                    safety_mult = 0.0
                else:
                    safety_mult = math.exp(-15.0 * p_out)

                # Anti-Cycling Multi-Ball Recency Penalty (for AI's own pattern hiding)
                history_pen = 1.0
                for pos, prev_m in enumerate(reversed(self.ai_move_history)):
                    if prev_m == m:
                        step_decay = 1.0 - (0.85 ** (pos + 1))
                        history_pen *= step_decay

                w = base_w * safety_mult * history_pen
                weights.append(w)

            total_w = sum(weights)
            if total_w <= 0:
                # If all moves were deemed dangerous, pick the move with absolute minimum p_out
                min_p = min(p_outs)
                safe_indices = [i for i, p in enumerate(p_outs) if p == min_p]
                norm_weights = [1.0 / len(safe_indices) if i in safe_indices else 0.0 for i in range(10)]
            else:
                norm_weights = [w / total_w for w in weights]

            # SMART EXPLORATION FLOOR:
            # Only distribute exploration floor across SAFE moves (p_out < 0.20). Never explore spammed bowler moves!
            safe_indices = [i for i, p in enumerate(p_outs) if p < 0.20]
            if not safe_indices:
                safe_indices = [i for i, p in enumerate(p_outs) if p == min(p_outs)]

            floor_per_safe = base_exp_floor / len(safe_indices)
            final_weights = []
            for idx in range(10):
                nw = (1.0 - base_exp_floor) * norm_weights[idx]
                if idx in safe_indices:
                    nw += floor_per_safe
                final_weights.append(nw)

            selected = random.choices(legal_moves, weights=final_weights, k=1)[0]

        self.ai_move_history.append(selected)
        if len(self.ai_move_history) > 6:
            self.ai_move_history.pop(0)

        return selected