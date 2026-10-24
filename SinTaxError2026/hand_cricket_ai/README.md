# Adaptive Hand Cricket AI

## Files
- `hand_cricket_ai.py`: playable, online-learning CLI prototype.
- Run `python hand_cricket_ai.py --generate-synthetic` to generate `synthetic_patterned_data.csv`.
- `hand_cricket_profile.json`: persistent learned profile, created on first run.
- `hand_cricket_moves.csv`: append-only move log.
- `hand_cricket_matches.csv`: completed match outcomes.

## Run
```bash
python hand_cricket_ai.py
```
Enter 100–2000 human gameplay number selections when prompted. `q` at a gameplay prompt saves progress and quits. The AI starts a new match automatically when a match ends, until the selected move limit is reached.

## Important prototype limitations
- The implementation uses the rules as described: numbers 1–10, equal number means wicket, otherwise the batter's number is added.
- The session limit counts your batting/bowling selections, not toss choices.
- It logs W/L after completed matches; a match cut off by the move limit is logged as partial moves, not assigned a fake W/L.
- The synthetic dataset contains planted patterns and is for testing/benchmarking only. It is not evidence about any real player's subconscious behavior.
- This is a baseline online probability model, not a guarantee of winning. Random choices cannot be predicted reliably.
