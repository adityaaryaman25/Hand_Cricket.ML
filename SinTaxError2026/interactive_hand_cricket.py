"""
interactive_hand_cricket.py
===========================
Interactive Hand Cricket CLI Test Suite with Test / Debug Mode & Configurable Wickets.

Modes:
------
1. Normal Gameplay Mode: Clean CLI output, standard 1-wicket rules, hidden probabilities.
2. Test / Debug Mode:
   - Configurable Wickets per Innings (1 to 10 wickets).
   - Real-time probability distributions P(1)...P(10) displayed before every ball.
   - Live sub-model weights, leading prediction model, confidence status, and uncertainty metrics.
   - Comprehensive learning snapshot & diagnostic progress tracking.
"""

import os
import random
import sys
import time
from hand_cricket_engine import MovePredictor, Config, PopulationStore, StrategyEngine


def get_valid_int(prompt: str, min_val: int, max_val: int) -> int:
    """Helper to ensure valid integer input within range."""
    while True:
        try:
            val = int(input(prompt).strip())
            if min_val <= val <= max_val:
                return val
            print(f"Invalid input. Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")


def select_game_mode() -> tuple[bool, int]:
    """Prompts for Game Mode (Normal vs Test/Debug Mode) and Wickets per Innings."""
    print("\nSelect Mode:")
    print("  1. Normal Mode (Clean gameplay, 1 Wicket, hidden probabilities)")
    print("  2. Test / Debug Mode (Live probabilities, custom wickets, agent progress tracking)")
    choice = get_valid_int("Choose Mode (1-2): ", 1, 2)

    is_test_mode = (choice == 2)
    if is_test_mode:
        max_wickets = get_valid_int("Enter Wickets per Innings for Test Mode (1-10): ", 1, 10)
        print(f"-> TEST MODE ACTIVATED: {max_wickets} Wicket(s) per Innings.\n")
    else:
        max_wickets = 1
        print("-> NORMAL MODE ACTIVATED: Standard 1 Wicket.\n")

    return is_test_mode, max_wickets


def select_difficulty() -> str:
    """Prompts player to select AI Difficulty level."""
    print("Select Difficulty Level:")
    print("  1. Easy   (Mostly random bowling & batting)")
    print("  2. Medium (Frequency-based prediction)")
    print("  3. Hard   (Frequency + Markov Chains + Recency weighting)")
    print("  4. Expert (Full adaptive ML + Anti-cycling risk management)")
    choice = get_valid_int("Choose level (1-4): ", 1, 4)

    mapping = {1: "easy", 2: "medium", 3: "hard", 4: "expert"}
    selected = mapping[choice]
    print(f"Difficulty set to: {selected.upper()}\n")
    return selected


def run_toss(player_id: str) -> str:
    """Executes Odd/Even toss."""
    print("--- TOSS ---")
    choice = ""
    while choice not in ("o", "e", "odd", "even"):
        choice = input("Choose Odd or Even (o/e): ").strip().lower()

    user_is_odd = choice.startswith("o")
    user_num = get_valid_int("Enter your toss number (1-10): ", 1, 10)
    ai_num = random.randint(1, 10)
    toss_sum = user_num + ai_num
    is_sum_odd = (toss_sum % 2 != 0)

    print(f"You played: {user_num} | AI played: {ai_num} | Sum = {toss_sum} ({'Odd' if is_sum_odd else 'Even'})")

    user_won = (is_sum_odd == user_is_odd)

    if user_won:
        print(f"Congratulations {player_id}, you WON the toss!")
        pick = ""
        while pick not in ("bat", "bowl", "1", "2"):
            pick = input("Choose (1) Batting or (2) Bowling: ").strip().lower()
        if pick in ("1", "bat"):
            print("You chose to BAT first.\n")
            return "bat"
        else:
            print("You chose to BOWL first.\n")
            return "bowl"
    else:
        print("AI WON the toss!")
        ai_pick = "bat" if random.random() < 0.5 else "bowl"
        if ai_pick == "bat":
            print("AI chose to BAT first (You will BOWL first).\n")
            return "bowl"
        else:
            print("AI chose to BOWL first (You will BAT first).\n")
            return "bat"


def print_debug_dashboard(pred_out: dict, ball_count: int, wickets_lost: int, max_wickets: int, role: str):
    """Renders a real-time ML diagnostic dashboard in Test Mode."""
    probs = pred_out["predicted_probabilities"]
    conf = pred_out["confidence_status"]
    leading = pred_out["leading_model"]
    obs = pred_out["observations"]
    top_w = pred_out["model_weights"]
    entropy = pred_out["uncertainty"]["entropy_normalized"]

    print("\n" + "┌" + "─" * 68 + "┐")
    print(f"│ 🧪 [TEST MODE DIAGNOSTICS] Ball #{ball_count} | Wickets: {wickets_lost}/{max_wickets} | Opponent Role: {role.upper():<5} │")
    print("├" + "─" * 68 + "┤")
    
    # Print Probability Grid in 2 neat rows (1-5 and 6-10)
    p_str1 = "  ".join([f"{num}:{probs[str(num)]:.3f}" for num in range(1, 6)])
    p_str2 = "  ".join([f"{num if num < 10 else '10'}:{probs[str(num)]:.3f}" for num in range(6, 11)])
    print(f"│ Probabilities (1-5) : {p_str1:<45} │")
    print(f"│ Probabilities (6-10): {p_str2:<45} │")
    print("├" + "─" * 68 + "┤")
    print(f"│ Leading Sub-Model   : {leading:<20} | Confidence: {conf:<15} │")
    print(f"│ Total Obs Logged    : {obs:<20} | Normalized Entropy: {entropy:<9.4f} │")
    
    w_str = ", ".join([f"{k}:{v*100:.1f}%" for k, v in list(top_w.items())[:3]])
    print(f"│ Top Model Weights   : {w_str:<45} │")
    print("└" + "─" * 68 + "┘")


def play_single_game(game_number: int, player_id: str, predictor: MovePredictor,
                     strategy: StrategyEngine, difficulty: str, test_mode: bool,
                     max_wickets: int, session_id: str) -> bool:
    """Plays a 1-game session of Hand Cricket with configurable wickets and Test Mode diagnostics."""
    print("=" * 70)
    mode_label = f"TEST MODE ({max_wickets} Wicket{'s' if max_wickets > 1 else ''})" if test_mode else "NORMAL MODE"
    print(f"GAME {game_number} START [{mode_label} | Difficulty: {difficulty.upper()}]")
    print("=" * 70)

    strategy.reset_history()
    human_role_inn1 = run_toss(player_id)
    ai_role_inn1 = "bowl" if human_role_inn1 == "bat" else "bat"

    # --- INNINGS 1 ---
    print("=================== INNINGS 1 ===================")
    print(f"Batsman: {'YOU (' + player_id + ')' if human_role_inn1 == 'bat' else 'AI Agent'}")
    print(f"Bowler : {'AI Agent' if human_role_inn1 == 'bat' else 'YOU (' + player_id + ')'}")
    print(f"Wickets: {max_wickets}")
    print("=================================================")

    score_inn1 = 0
    wickets_inn1 = 0
    ball_count = 0

    while wickets_inn1 < max_wickets:
        ball_count += 1
        opp_context = {
            "session_id": session_id,
            "game_id": game_number,
            "innings": 1,
            "role": human_role_inn1,
            "score": score_inn1,
            "target": None,
            "wickets_left": max_wickets - wickets_inn1,
            "max_wickets": max_wickets
        }

        # 1. AI Predicts Human's Next Move BEFORE input
        pred_out = predictor.predict(player_id, opp_context, difficulty=difficulty)
        probs = pred_out["predicted_probabilities"]

        # If Test Mode is enabled, show the ML probabilities dashboard
        if test_mode:
            print_debug_dashboard(pred_out, ball_count, wickets_inn1, max_wickets, human_role_inn1)

        # 2. Strategy Engine selects AI move
        ai_move = strategy.select_ai_move(ai_role_inn1, probs, opp_context, difficulty=difficulty)

        # 3. Get Human input
        human_move = get_valid_int(f"[Ball {ball_count}] Enter your number (1-10): ", 1, 10)

        p_human = probs[str(human_move)]
        print(f" -> You played: {human_move} | AI played: {ai_move}")
        if test_mode:
            print(f"    [P({human_move}) Assigned by AI = {p_human:.4f}]")

        # 4. Post-Ball Update / Online Backprop Learning
        predictor.update(player_id, observed_move=human_move, context=opp_context, my_move=ai_move)

        if human_move == ai_move:
            wickets_inn1 += 1
            print(f"\n>>> OUT! Wicket {wickets_inn1}/{max_wickets} falls on ball {ball_count}!")
            print(f"    Current Score: {score_inn1} runs ({wickets_inn1}/{max_wickets} wkts)\n")
        else:
            runs_this_ball = human_move if human_role_inn1 == "bat" else ai_move
            score_inn1 += runs_this_ball
            scorer = "You" if human_role_inn1 == "bat" else "AI"
            print(f"    {scorer} scored {runs_this_ball} runs. Total: {score_inn1}/{wickets_inn1}")

    target = score_inn1 + 1
    print(f"\n>>> INNINGS 1 END. Final Score: {score_inn1}/{wickets_inn1}")
    print(f">>> TARGET SET: {target} runs needed to win in Innings 2.\n")

    # --- INNINGS 2 ---
    human_role_inn2 = "bowl" if human_role_inn1 == "bat" else "bat"
    ai_role_inn2 = "bowl" if human_role_inn2 == "bat" else "bat"

    print("=================== INNINGS 2 ===================")
    print(f"Batsman: {'YOU (' + player_id + ')' if human_role_inn2 == 'bat' else 'AI Agent'}")
    print(f"Bowler : {'AI Agent' if human_role_inn2 == 'bat' else 'YOU (' + player_id + ')'}")
    print(f"Target : {target} runs ({max_wickets} Wickets)")
    print("=================================================")

    score_inn2 = 0
    wickets_inn2 = 0
    ball_count_2 = 0
    game_winner = None

    while game_winner is None:
        ball_count_2 += 1
        opp_context_2 = {
            "session_id": session_id,
            "game_id": game_number,
            "innings": 2,
            "role": human_role_inn2,
            "score": score_inn2,
            "target": target,
            "wickets_left": max_wickets - wickets_inn2,
            "max_wickets": max_wickets
        }

        # 1. AI Predicts Human's Next Move BEFORE input
        pred_out = predictor.predict(player_id, opp_context_2, difficulty=difficulty)
        probs = pred_out["predicted_probabilities"]

        if test_mode:
            print_debug_dashboard(pred_out, ball_count_2, wickets_inn2, max_wickets, human_role_inn2)

        # 2. Strategy Engine selects AI move
        ai_move = strategy.select_ai_move(ai_role_inn2, probs, opp_context_2, difficulty=difficulty)

        # 3. Get Human input
        human_move = get_valid_int(f"[Innings 2 - Ball {ball_count_2}] Enter your number (1-10): ", 1, 10)

        p_human = probs[str(human_move)]
        print(f" -> You played: {human_move} | AI played: {ai_move}")
        if test_mode:
            print(f"    [P({human_move}) Assigned by AI = {p_human:.4f}]")

        # 4. Post-Ball Update / Online Backprop Learning
        predictor.update(player_id, observed_move=human_move, context=opp_context_2, my_move=ai_move)

        if human_move == ai_move:
            wickets_inn2 += 1
            print(f"\n>>> OUT! Wicket {wickets_inn2}/{max_wickets} falls on ball {ball_count_2}!")
            print(f"    Current Score: {score_inn2}/{wickets_inn2} (Target: {target})\n")
            
            if wickets_inn2 >= max_wickets:
                if score_inn2 >= target:
                    game_winner = "human" if human_role_inn2 == "bat" else "ai"
                else:
                    game_winner = "ai" if human_role_inn2 == "bat" else "human"
                break
        else:
            runs_this_ball = human_move if human_role_inn2 == "bat" else ai_move
            score_inn2 += runs_this_ball
            scorer = "You" if human_role_inn2 == "bat" else "AI"
            print(f"    {scorer} scored {runs_this_ball} runs. Total: {score_inn2}/{wickets_inn2} (Target: {target})")

            if score_inn2 >= target:
                print(f"\n>>> TARGET PASSED! Final Score: {score_inn2}/{wickets_inn2} >= {target}")
                game_winner = "human" if human_role_inn2 == "bat" else "ai"
                break

    print("\n" + "-" * 50)
    if game_winner == "human":
        print(f"🏆 GAME {game_number} WINNER: YOU ({player_id})!")
        return True
    else:
        print(f"🤖 GAME {game_number} WINNER: AI AGENT!")
        return False


def main():
    print("=" * 70)
    print(" INTERACTIVE HAND CRICKET ML TESTING SUITE (Moves 1-10) ")
    print("=" * 70)

    player_id = input("\nEnter Player Name / ID (e.g. rahul): ").strip()
    if not player_id:
        player_id = "test_player"

    num_games = get_valid_int("How many games do you want to play? ", 1, 100)
    test_mode, max_wickets = select_game_mode()
    difficulty = select_difficulty()

    cfg10 = Config(legal_numbers=tuple(range(1, 11)))
    predictor = MovePredictor(store_dir="player_profiles", config=cfg10)
    strategy = StrategyEngine(temperature=0.8, exploration_floor=0.10)

    human_wins = 0
    ai_wins = 0
    session_id = f"cli_session_{int(time.time())}"

    for g in range(1, num_games + 1):
        user_won = play_single_game(g, player_id, predictor, strategy, difficulty,
                                    test_mode, max_wickets, session_id)
        if user_won:
            human_wins += 1
        else:
            ai_wins += 1

        print(f"\n[CURRENT SERIES SCORE] {player_id}: {human_wins} | AI Agent: {ai_wins}\n")

    print("=" * 70)
    print(" FINAL SERIES RESULTS & AGENT PERFORMANCE ")
    print("=" * 70)
    print(f"Player ID         : {player_id}")
    print(f"Game Mode         : {'TEST / DEBUG MODE' if test_mode else 'NORMAL MODE'}")
    print(f"Wickets per Inn   : {max_wickets}")
    print(f"Difficulty Level  : {difficulty.upper()}")
    print(f"Total Games Played: {num_games}")
    print(f"Series Score      : {player_id} {human_wins} - {ai_wins} AI Agent")

    profile = predictor.profile(player_id)
    print(f"\n[AGENT LEARNING & ACCURACY METRICS FOR {player_id.upper()}]")
    print(f"Total Balls Logged Across Matches : {len(profile.records)}")
    top_w = sorted(profile.weights.items(), key=lambda kv: -kv[1])[:4]
    print("Learned Sub-Model Weight Breakdown:")
    for m_name, w in top_w:
        print(f"  - {m_name:<20}: {w * 100:6.2f}% weight")

    status, ev = profile._assess()
    print(f"Pattern Detection Confidence     : {status.upper()}")
    print(f"Information Edge vs Random      : {ev['edge_vs_uniform_nats_per_obs']} nats/obs")
    print(f"Recent Brier Loss Score (0=best) : {ev['brier_score_recent']}")
    print(f"Profile Saved To                 : player_profiles/{player_id}.json")
    print("=" * 70)


if __name__ == "__main__":
    main()