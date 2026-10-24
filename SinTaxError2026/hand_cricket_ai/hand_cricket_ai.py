#!/usr/bin/env python3
"""
Adaptive Hand Cricket AI — standard-library-only prototype.

Run:
    python hand_cricket_ai.py

Optional:
    python hand_cricket_ai.py --generate-synthetic
    python hand_cricket_ai.py --reset-profile

Game rules:
- Human calls odd/even and both choose a toss number (1..10).
- The caller wins the toss if the sum has the called parity; otherwise the other player wins.
- Batter and bowler each choose 1..10. Equal numbers = wicket; otherwise batter's number is added.
- The session counts HUMAN gameplay number selections (batting or bowling), not toss picks.
- The session stops at the requested number of human gameplay selections, even mid-match.
- Learned profile/history persist in JSON/CSV between runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
PROFILE_PATH = ROOT / "hand_cricket_profile.json"
MOVES_PATH = ROOT / "hand_cricket_moves.csv"
MATCHES_PATH = ROOT / "hand_cricket_matches.csv"
SYNTHETIC_PATH = ROOT / "synthetic_patterned_data.csv"
NUMBERS = list(range(1, 11))
ROLES = ("batting", "bowling", "toss")
ALPHA = 1.0


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def safe_int(prompt, low, high):
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
            if low <= value <= high:
                return value
        except ValueError:
            pass
        print(f"Enter a whole number from {low} to {high}.")


def safe_choice(prompt, choices):
    choices = [c.lower() for c in choices]
    while True:
        value = input(prompt).strip().lower()
        if value in choices:
            return value
        print("Choose: " + ", ".join(choices))


class AdaptivePlayerModel:
    """Online model of the HUMAN's number choices, split by role."""
    def __init__(self):
        self.counts = {role: Counter() for role in ROLES}
        self.transitions = {role: defaultdict(Counter) for role in ROLES}
        self.recent = {role: deque(maxlen=60) for role in ROLES}
        self.last = {role: None for role in ROLES}
        self.observations = Counter()
        # Match-level feedback for AI actions. This is weak credit assignment:
        # all actions in a completed match receive that match's result.
        self.feedback = {role: {n: Counter() for n in NUMBERS} for role in ("batting", "bowling")}

    def observe(self, role, number):
        if role not in ROLES or number not in NUMBERS:
            return
        previous = self.last[role]
        self.counts[role][number] += 1
        if previous is not None:
            self.transitions[role][previous][number] += 1
        self.recent[role].append(number)
        self.last[role] = number
        self.observations[role] += 1

    def probabilities(self, role, previous=None):
        """Blend smoothed long-run frequency, recent frequency and transitions."""
        alpha = ALPHA
        long_total = sum(self.counts[role].values())
        long_probs = {
            n: (self.counts[role][n] + alpha) / (long_total + 10 * alpha)
            for n in NUMBERS
        }

        recent_counts = Counter(self.recent[role])
        recent_total = len(self.recent[role])
        recent_probs = {
            n: (recent_counts[n] + alpha) / (recent_total + 10 * alpha)
            for n in NUMBERS
        }

        # A transition distribution is trusted only after enough examples.
        row = self.transitions[role].get(previous, Counter()) if previous is not None else Counter()
        row_total = sum(row.values())
        if row_total >= 4:
            transition_probs = {n: (row[n] + alpha) / (row_total + 10 * alpha) for n in NUMBERS}
            probs = {n: 0.45 * transition_probs[n] + 0.35 * recent_probs[n] + 0.20 * long_probs[n]
                     for n in NUMBERS}
        else:
            probs = {n: 0.60 * recent_probs[n] + 0.40 * long_probs[n] for n in NUMBERS}

        total = sum(probs.values())
        return {n: probs[n] / total for n in NUMBERS}

    def best_prediction(self, role, previous=None):
        probs = self.probabilities(role, previous)
        best_p = max(probs.values())
        tied = [n for n, p in probs.items() if abs(p - best_p) < 1e-12]
        return random.choice(tied), probs

    def save(self, path=PROFILE_PATH):
        payload = {
            "version": 1,
            "saved_at": now_iso(),
            "counts": {r: {str(n): self.counts[r][n] for n in NUMBERS} for r in ROLES},
            "transitions": {
                r: {str(prev): {str(n): c for n, c in self.transitions[r][prev].items()}
                    for prev in self.transitions[r]}
                for r in ROLES
            },
            "recent": {r: list(self.recent[r]) for r in ROLES},
            "last": self.last,
            "observations": dict(self.observations),
            "feedback": {r: {str(n): dict(self.feedback[r][n]) for n in NUMBERS}
                         for r in ("batting", "bowling")},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path=PROFILE_PATH):
        model = cls()
        if not path.exists():
            return model
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for role in ROLES:
                for n, c in data.get("counts", {}).get(role, {}).items():
                    model.counts[role][int(n)] = int(c)
                for prev, row in data.get("transitions", {}).get(role, {}).items():
                    model.transitions[role][int(prev)] = Counter({int(n): int(c) for n, c in row.items()})
                model.recent[role].extend(int(n) for n in data.get("recent", {}).get(role, [])[-60:])
                last = data.get("last", {}).get(role)
                model.last[role] = int(last) if last is not None else (model.recent[role][-1] if model.recent[role] else None)
            model.observations.update(data.get("observations", {}))
            for role in ("batting", "bowling"):
                for n, stats in data.get("feedback", {}).get(role, {}).items():
                    model.feedback[role][int(n)].update(stats)
        except (ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
            print(f"Warning: profile could not be loaded ({exc}); starting with a fresh model.")
        return model


def choose_batting_number(model, target, current_score, wickets_left=1):
    # Human is bowling: estimate their bowling-number distribution.
    probs = model.probabilities("bowling", model.last["bowling"])
    # Expected immediate runs. Tiny survival bonus avoids brittle ties and is
    # deliberately small; this baseline is not a full optimal cricket solver.
    utilities = {}
    for n in NUMBERS:
        dismissal_p = probs[n]
        expected_runs = n * (1.0 - dismissal_p)
        urgency = 1.0
        if target is not None:
            needed = max(0, target - current_score)
            urgency = 1.0 + min(0.30, needed / max(10, 10 * (target or 1)))
        stats = model.feedback["batting"][n]
        games = stats["win"] + stats["loss"] + stats["draw"]
        historical_win_rate = (stats["win"] + 0.5 * stats["draw"] + 1.0) / (games + 2.0)
        # Small tie-break-sized adjustment: match W/L is noisy evidence for any one move.
        utilities[n] = expected_runs * urgency + 0.15 * (historical_win_rate - 0.5)
    best = max(utilities.values())
    candidates = [n for n, u in utilities.items() if abs(u - best) < 1e-9]
    return random.choice(candidates), probs, utilities


def choose_bowling_number(model):
    # Human is batting: match their most likely number.
    probs = model.probabilities("batting", model.last["batting"])
    # Use predicted dismissal chance first; past match outcomes are only a tiny tie-break.
    utilities = {}
    for n in NUMBERS:
        stats = model.feedback["bowling"][n]
        games = stats["win"] + stats["loss"] + stats["draw"]
        historical_win_rate = (stats["win"] + 0.5 * stats["draw"] + 1.0) / (games + 2.0)
        utilities[n] = probs[n] + 0.01 * (historical_win_rate - 0.5)
    best = max(utilities.values())
    candidates = [n for n, u in utilities.items() if abs(u - best) < 1e-12]
    return random.choice(candidates), probs


def append_csv(path, fields, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


MOVE_FIELDS = [
    "timestamp", "session_move", "match_id", "innings", "batter", "human_role",
    "human_number", "ai_number", "runs", "wicket", "human_score", "ai_score",
    "target", "human_predicted_probability", "ai_action_probability",
]
MATCH_FIELDS = ["timestamp", "match_id", "winner", "human_score", "ai_score", "human_won_toss", "toss_choice"]


def generate_synthetic_data(path=SYNTHETIC_PATH, rows=3000, seed=41):
    """
    Generate DEMONSTRATION data with deliberately planted patterns.
    These are synthetic benchmark sequences, not real human data.
    """
    rng = random.Random(seed)
    patterns = ["repeat", "cycle", "after_7", "avoid_previous", "mostly_mid", "uniform"]
    fields = ["player_id", "step", "role", "previous_number", "number", "pattern"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        per_player = rows // len(patterns)
        for p_idx, pattern in enumerate(patterns):
            previous = rng.randint(1, 10)
            cycle = [2, 7, 4, 9, 1, 6, 3, 8, 5, 10]
            for step in range(per_player):
                role = "batting" if step % 2 == 0 else "bowling"
                if pattern == "repeat":
                    n = previous if rng.random() < 0.68 else rng.randint(1, 10)
                elif pattern == "cycle":
                    n = cycle[step % len(cycle)] if rng.random() < 0.78 else rng.randint(1, 10)
                elif pattern == "after_7":
                    n = 2 if previous == 7 and rng.random() < 0.75 else rng.randint(1, 10)
                elif pattern == "avoid_previous":
                    options = [x for x in NUMBERS if x != previous]
                    n = rng.choice(options) if rng.random() < 0.85 else previous
                elif pattern == "mostly_mid":
                    n = rng.choice([4, 5, 6, 7]) if rng.random() < 0.72 else rng.randint(1, 10)
                else:
                    n = rng.randint(1, 10)
                writer.writerow({
                    "player_id": f"synthetic_{p_idx+1}", "step": step + 1, "role": role,
                    "previous_number": previous, "number": n, "pattern": pattern
                })
                previous = n
    print(f"Generated {path} with {per_player * len(patterns)} synthetic rows.")
    print("Use this to test pattern-detection code; do NOT treat it as your personal history.")


def show_distribution(model, role):
    probs = model.probabilities(role, model.last[role])
    print(f"\nLearned human {role} distribution ({model.observations[role]} observations):")
    for n in NUMBERS:
        bar = "█" * int(probs[n] * 40)
        print(f" {n:2}: {probs[n]:5.1%} {bar}")


def play_session(move_limit):
    model = AdaptivePlayerModel.load()
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_moves = 0
    match_id = 0
    human_match_wins = 0
    ai_match_wins = 0
    draws = 0
    total_matches = 0
    total_human_runs = 0
    total_ai_runs = 0

    print("\nAdaptive Hand Cricket AI")
    print(f"Target: {move_limit} human gameplay number selections")
    print(f"Existing learned observations: batting={model.observations['batting']}, "
          f"bowling={model.observations['bowling']}")
    print("Your move history and model profile will be saved locally.")
    print("Enter q at any gameplay prompt to stop early; progress is saved.\n")

    while session_moves < move_limit:
        match_id += 1
        toss_choice = safe_choice("Toss: call odd or even? [odd/even]: ", ["odd", "even"])
        human_toss_number = safe_int("Your toss number (1-10): ", 1, 10)
        ai_toss_number = random.randint(1, 10)
        toss_sum = human_toss_number + ai_toss_number
        human_won_toss = ((toss_sum % 2 == 1 and toss_choice == "odd") or
                          (toss_sum % 2 == 0 and toss_choice == "even"))
        print(f"AI toss number: {ai_toss_number}; sum={toss_sum}.")
        if human_won_toss:
            toss_winner = "human"
            choice = safe_choice("You won the toss. Choose bat or bowl: ", ["bat", "bowl"])
        else:
            toss_winner = "ai"
            choice = random.choice(["bat", "bowl"])
            print(f"AI won the toss and chooses to {choice}.")
        human_bats_first = (choice == "bat") if toss_winner == "human" else (choice == "bowl")

        human_score = 0
        ai_score = 0
        ai_actions = {"batting": [], "bowling": []}
        target = None
        innings = 1
        match_finished = False
        human_role = "batting" if human_bats_first else "bowling"
        print(f"\n--- Match {match_id}: you {'bat' if human_bats_first else 'bowl'} first ---")

        while session_moves < move_limit and not match_finished:
            # First innings: whichever side bats first; second innings chases target.
            human_batting = (human_role == "batting")
            if human_batting:
                ai_number, ai_probs = choose_bowling_number(model)
                ai_actions["bowling"].append(ai_number)
                raw = input(f"Move {session_moves+1}/{move_limit} | YOU BAT. Choose 1-10: ").strip().lower()
                if raw == "q":
                    model.save()
                    print("\nStopped early. Progress saved.")
                    return
                try:
                    human_number = int(raw)
                    if human_number not in NUMBERS:
                        raise ValueError
                except ValueError:
                    print("Enter a number from 1 to 10, or q to save and quit.")
                    continue
                pred_prob = ai_probs[human_number]
                model.observe("batting", human_number)
                session_moves += 1
                wicket = human_number == ai_number
                runs = 0 if wicket else human_number
                human_score += runs
                print(f"AI bowls {ai_number}. " + ("OUT!" if wicket else f"+{runs} runs."))
                append_csv(MOVES_PATH, MOVE_FIELDS, {
                    "timestamp": now_iso(), "session_move": session_moves, "match_id": f"{session_id}_{match_id}",
                    "innings": innings, "batter": "human", "human_role": "batting",
                    "human_number": human_number, "ai_number": ai_number, "runs": runs,
                    "wicket": int(wicket), "human_score": human_score, "ai_score": ai_score,
                    "target": target if target is not None else "", "human_predicted_probability": round(pred_prob, 6),
                    "ai_action_probability": round(ai_probs[ai_number], 6),
                })
                if wicket:
                    if target is None:
                        target = human_score + 1
                        human_role = "bowling"
                        innings = 2
                        print(f"Innings over. Your total: {human_score}; AI target: {target}.")
                    else:
                        match_finished = True
                if target is not None and innings == 2 and ai_score >= target:
                    match_finished = True
            else:
                ai_number, probs, utilities = choose_batting_number(model, target, ai_score)
                ai_actions["batting"].append(ai_number)
                raw = input(f"Move {session_moves+1}/{move_limit} | YOU BOWL. Choose 1-10: ").strip().lower()
                if raw == "q":
                    model.save()
                    print("\nStopped early. Progress saved.")
                    return
                try:
                    human_number = int(raw)
                    if human_number not in NUMBERS:
                        raise ValueError
                except ValueError:
                    print("Enter a number from 1 to 10, or q to save and quit.")
                    continue
                pred_prob = probs[human_number]
                model.observe("bowling", human_number)
                session_moves += 1
                wicket = human_number == ai_number
                runs = 0 if wicket else ai_number
                ai_score += runs
                print(f"AI bats {ai_number}. " + ("AI OUT!" if wicket else f"AI scores {runs}."))
                append_csv(MOVES_PATH, MOVE_FIELDS, {
                    "timestamp": now_iso(), "session_move": session_moves, "match_id": f"{session_id}_{match_id}",
                    "innings": innings, "batter": "ai", "human_role": "bowling",
                    "human_number": human_number, "ai_number": ai_number, "runs": runs,
                    "wicket": int(wicket), "human_score": human_score, "ai_score": ai_score,
                    "target": target if target is not None else "", "human_predicted_probability": round(pred_prob, 6),
                    "ai_action_probability": round(probs[ai_number], 6),
                })
                if wicket:
                    if target is None:
                        target = ai_score + 1
                        human_role = "batting"
                        innings = 2
                        print(f"AI innings over. AI total: {ai_score}; your target: {target}.")
                    else:
                        match_finished = True
                if target is not None and innings == 2 and human_score >= target:
                    match_finished = True

            # Handle chase victory after a scoring move.
            if target is not None:
                if innings == 2:
                    if human_role == "batting" and human_score >= target:
                        match_finished = True
                    if human_role == "bowling" and ai_score >= target:
                        match_finished = True

            # Switch roles after first innings ended (target has just been set).
            if target is not None and innings == 2 and not match_finished:
                # human_role was already set to the chasing side above.
                pass

            # If innings one ended by wicket, role switch has happened. If the
            # target was reached, stop. Otherwise continue until the chase wicket.
            if session_moves >= move_limit:
                break

        # A match is only scored as W/L if both innings actually concluded.
        # If the move limit ends mid-match, retain moves but don't fabricate a result.
        completed = (target is not None and match_finished)
        if completed:
            total_matches += 1
            if human_score > ai_score:
                winner = "human"
                human_match_wins += 1
            elif ai_score > human_score:
                winner = "ai"
                ai_match_wins += 1
            else:
                winner = "draw"
                draws += 1
            # Feed the completed match outcome back into the actions used in it.
            # This is intentionally a small influence because match-level credit is noisy.
            outcome_key = "win" if winner == "ai" else ("loss" if winner == "human" else "draw")
            for ai_role, used_numbers in ai_actions.items():
                for used_number in used_numbers:
                    model.feedback[ai_role][used_number][outcome_key] += 1
            append_csv(MATCHES_PATH, MATCH_FIELDS, {
                "timestamp": now_iso(), "match_id": f"{session_id}_{match_id}", "winner": winner,
                "human_score": human_score, "ai_score": ai_score,
                "human_won_toss": int(human_won_toss), "toss_choice": toss_choice,
            })
            print(f"\nMatch result: {winner.upper()} | You {human_score} - AI {ai_score}")
            print(f"Session results: {human_match_wins} human wins, {ai_match_wins} AI wins, {draws} draws.")
        else:
            print(f"\nMove limit reached during match {match_id}; partial match saved as moves, not W/L.")
        total_human_runs += human_score
        total_ai_runs += ai_score
        model.save()
        if session_moves >= move_limit:
            break
        print(f"\nStarting next match automatically ({session_moves}/{move_limit} moves used).")

    print("\n=== SESSION SUMMARY ===")
    print(f"Human selections processed: {session_moves}/{move_limit}")
    print(f"Completed matches: {total_matches} | Human wins: {human_match_wins} | AI wins: {ai_match_wins} | Draws: {draws}")
    if total_matches:
        print(f"AI win rate among completed matches: {100*ai_match_wins/total_matches:.1f}%")
    show_distribution(model, "batting")
    show_distribution(model, "bowling")
    model.save()
    print(f"\nSaved profile: {PROFILE_PATH}")
    print(f"Saved move log: {MOVES_PATH}")
    print(f"Saved completed-match log: {MATCHES_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Adaptive hand-cricket opponent")
    parser.add_argument("--generate-synthetic", action="store_true",
                        help="generate a separate synthetic pattern benchmark CSV and exit")
    parser.add_argument("--reset-profile", action="store_true",
                        help="delete the learned JSON profile before playing")
    args = parser.parse_args()

    if args.generate_synthetic:
        generate_synthetic_data()
        return
    if args.reset_profile and PROFILE_PATH.exists():
        PROFILE_PATH.unlink()
        print("Deleted learned profile. Existing CSV logs were left untouched.")

    print("How many human gameplay number selections should the session process?")
    print("Recommended: 1,500. Valid range: 100-2,000.")
    move_limit = safe_int("Total moves [100-2000]: ", 100, 2000)
    play_session(move_limit)


if __name__ == "__main__":
    main()
