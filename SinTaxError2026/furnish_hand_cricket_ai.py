#!/usr/bin/env python3
"""
Adaptive Hand Cricket AI with webcam gesture input and a dashboard-style UI.

Run:
    python hand_cricket_ai.py

Optional:
    python hand_cricket_ai.py --generate-synthetic
    python hand_cricket_ai.py --reset-profile

How you play:
- CHOICES (odd/even toss call, bat/bowl) are made in popups inside the game
  window: click a button or press 1 / 2.
- NUMBERS are shown to the webcam. Whenever a number gesture is detected and
  stable, it is taken as your turn (toss number, batting number, bowling number).
- After a number is accepted, lower your hand out of the camera view; the next
  turn starts once the camera sees no hand. (This lets you show the same
  number twice in a row.)
- Press Q in the game window to save and quit.

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
import random
import sys
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

from hand_cricket_engine import MovePredictor, Config, StrategyEngine

ROOT = Path(__file__).resolve().parent
PROFILE_PATH = ROOT / "hand_cricket_profile.json"
MOVES_PATH = ROOT / "hand_cricket_moves.csv"
MATCHES_PATH = ROOT / "hand_cricket_matches.csv"
SYNTHETIC_PATH = ROOT / "synthetic_patterned_data.csv"
NUMBERS = list(range(1, 11))
ROLES = ("batting", "bowling", "toss")
ALPHA = 1.0

# ---------------- GESTURE CONFIGURATION ----------------
CAMERA_INDEX = 0
HAND_MODEL_PATH = "models/hand_landmarker.task"
CLASSIFIER_PATH = "models/gesture_model.joblib"
CONFIDENCE_THRESHOLD = 0.65
SMOOTHING_WINDOW = 9
MIN_VOTES = 5
HAND_DOWN_FRAMES = 6  # consecutive "no hand" frames needed before the next turn

# ---------------- UI CONFIGURATION ----------------
UI_SCALE = 0.9  # 1.0 = 1200x900 window. Lower it (e.g. 0.8) if it doesn't fit your screen.

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class QuitRequested(Exception):
    """Raised when the player asks to stop (Q in the camera window or 'q' typed)."""


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


def safe_word(prompt, aliases):
    """
    Ask for a typed word. `aliases` maps accepted words to a canonical answer,
    e.g. {"bat": "bat", "bowl": "bowl", "ball": "bowl"}.
    """
    shown = "/".join(dict.fromkeys(aliases.values()))
    while True:
        value = input(prompt).strip().lower()
        if value in ("q", "quit"):
            raise QuitRequested
        if value in aliases:
            return aliases[value]
        print(f"Type one of: {shown}  (or q to quit)")


ODD_EVEN = {"odd": "odd", "even": "even"}
BAT_BOWL = {"bat": "bat", "bowl": "bowl", "ball": "bowl"}


# ======================================================================
# Dashboard UI (pure OpenCV drawing - no camera needed to render it)
# ======================================================================
class CricketUI:
    """
    Draws the whole game screen onto one canvas, inspired by the Figma design:
    header bar, dark score card, a big live-camera YOU card, an AI card with a
    VS badge, a feedback strip and a "Choose your move" row of number tiles.
    Popups (toss, innings change, results) are drawn inside the same window.

    All colours are BGR tuples (OpenCV order).
    """

    W, H = 1200, 880
    SCALE = UI_SCALE

    BG = (16, 12, 8)  # near-black page
    HEADER = (24, 18, 12)
    WHITE = (34, 26, 18)  # dark panel/card fill
    BORDER = (64, 54, 38)
    GREEN = (50, 45, 6)  # deep teal-black score card
    GREEN_LINE = (90, 80, 20)
    GREEN_TEXT = (200, 190, 120)  # muted cyan label
    LIME = (240, 220, 20)  # neon cyan accent
    LIME_TINT = (46, 40, 14)  # selected / "YOU" card fill
    LIME_EDGE = (150, 130, 10)
    INK = (245, 240, 225)  # main text (light)
    DARK = (14, 10, 6)  # dark text used on top of cyan
    MUTED = (150, 140, 110)
    LABEL = (170, 165, 120)  # small caps labels
    RED = (90, 80, 255)
    OK = (120, 230, 90)
    AMBER = (0, 190, 255)
    SALMON = (140, 100, 255)

    CAM_MAXW, CAM_MAXH = 540, 366   # the camera is fitted (not cropped) inside this area

    def __init__(self, cv2, np):
        self.cv2, self.np = cv2, np
        self.FONT = cv2.FONT_HERSHEY_DUPLEX
        self.state = {}
        self.round = None
        self.history = deque(maxlen=6)   # recent balls shown under your number
        self._masks = {}
        self.hover = None          # option index under the mouse in a choice popup
        self.button_rects = []     # (x1, y1, x2, y2) of popup buttons, in unscaled canvas coords

    def hit(self, x, y):
        for i, (x1, y1, x2, y2) in enumerate(self.button_rects):
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i
        return None

    # ---------- drawing helpers ----------
    def text(self, img, s, x, y, scale=0.7, color=None, thick=1, anchor="l", spacing=0):
        cv2 = self.cv2
        color = self.INK if color is None else color
        if spacing:
            widths = [cv2.getTextSize(ch, self.FONT, scale, thick)[0][0] + spacing for ch in s]
            total = sum(widths) - spacing
        else:
            total = cv2.getTextSize(s, self.FONT, scale, thick)[0][0]
        if anchor == "c":
            x -= total // 2
        elif anchor == "r":
            x -= total
        if spacing:
            for ch, w in zip(s, widths):
                cv2.putText(img, ch, (int(x), int(y)), self.FONT, scale, color, thick, cv2.LINE_AA)
                x += w
        else:
            cv2.putText(img, s, (int(x), int(y)), self.FONT, scale, color, thick, cv2.LINE_AA)
        return total

    def width(self, s, scale, thick=1):
        return self.cv2.getTextSize(s, self.FONT, scale, thick)[0][0]

    def fit(self, s, max_w, scale, thick=1):
        while len(s) > 4 and self.width(s, scale, thick) > max_w:
            s = s[:-4] + "..."
        return s

    def rrect(self, img, x, y, w, h, r, fill, border=None, bw=2):
        cv2 = self.cv2

        def draw(x, y, w, h, r, c):
            r = max(0, min(r, w // 2, h // 2))
            x2, y2 = x + w, y + h
            cv2.rectangle(img, (x + r, y), (x2 - r, y2), c, -1)
            cv2.rectangle(img, (x, y + r), (x2, y2 - r), c, -1)
            for cx, cy in ((x + r, y + r), (x2 - r, y + r), (x + r, y2 - r), (x2 - r, y2 - r)):
                cv2.circle(img, (cx, cy), r, c, -1, cv2.LINE_AA)

        if border is not None:
            draw(x, y, w, h, r, border)
            draw(x + bw, y + bw, w - 2 * bw, h - 2 * bw, r - bw, fill)
        else:
            draw(x, y, w, h, r, fill)

    def outline(self, img, x, y, w, h, r, color, t=3):
        cv2 = self.cv2
        x2, y2 = x + w - 1, y + h - 1
        aa = cv2.LINE_AA
        cv2.line(img, (x + r, y), (x2 - r, y), color, t, aa)
        cv2.line(img, (x + r, y2), (x2 - r, y2), color, t, aa)
        cv2.line(img, (x, y + r), (x, y2 - r), color, t, aa)
        cv2.line(img, (x2, y + r), (x2, y2 - r), color, t, aa)
        cv2.ellipse(img, (x + r, y + r), (r, r), 0, 180, 270, color, t, aa)
        cv2.ellipse(img, (x2 - r, y + r), (r, r), 0, 270, 360, color, t, aa)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, t, aa)
        cv2.ellipse(img, (x + r, y2 - r), (r, r), 0, 90, 180, color, t, aa)

    def paste_rounded(self, canvas, img, x, y, r):
        np = self.np
        h, w = img.shape[:2]
        key = (w, h, r)
        mask = self._masks.get(key)
        if mask is None:
            m = np.zeros((h, w), np.uint8)
            self.rrect(m, 0, 0, w - 1, h - 1, r, 255)
            mask = (m.astype(np.float32) / 255.0)[..., None]
            self._masks[key] = mask
        roi = canvas[y:y + h, x:x + w].astype(np.float32)
        canvas[y:y + h, x:x + w] = (img.astype(np.float32) * mask + roi * (1.0 - mask)).astype(np.uint8)

    def tone(self, color):
        """Map the bright BGR colours used by the game loop onto the UI palette."""
        b, g, r = color
        if g > 200 and r < 120 and b < 120:
            return self.OK
        if r > 200 and g < 120 and b < 120:
            return self.RED
        return self.AMBER

    def _icon(self, c, kind, cx, cy):
        cv2, np = self.cv2, self.np
        aa = cv2.LINE_AA
        if kind == "ball":
            cv2.circle(c, (cx, cy), 34, (60, 60, 205), -1, aa)
            cv2.circle(c, (cx, cy), 34, (40, 40, 150), 2, aa)
            cv2.ellipse(c, (cx - 26, cy), (26, 36), 0, -50, 50, (255, 255, 255), 2, aa)
            cv2.ellipse(c, (cx + 26, cy), (26, 36), 0, 130, 230, (255, 255, 255), 2, aa)
        elif kind == "bat":
            t = np.deg2rad(-45.0)
            u = (np.cos(t), np.sin(t))
            n = (-np.sin(t), np.cos(t))

            def poly(s0, s1, w, fill, edge):
                pts = np.array([(cx + u[0] * s + n[0] * d, cy + u[1] * s + n[1] * d)
                                for s, d in ((s0, -w / 2), (s1, -w / 2), (s1, w / 2), (s0, w / 2))],
                               np.int32)
                cv2.fillPoly(c, [pts], fill, aa)
                cv2.polylines(c, [pts], True, edge, 2, aa)

            poly(-46, 8, 24, (150, 195, 228), self.LIME_EDGE)
            poly(8, 42, 9, (150, 140, 110), self.LIME_EDGE)
        elif kind == "coin":
            cv2.circle(c, (cx, cy), 34, self.LIME, -1, aa)
            cv2.circle(c, (cx, cy), 34, self.LIME_EDGE, 3, aa)
            cv2.circle(c, (cx, cy), 25, self.LIME_EDGE, 1, aa)
            self.text(c, "T", cx, cy + 12, 1.1, self.DARK, 2, anchor="c")

    # ---------- state ----------
    def _round_texts(self):
        """(title, subtitle, mood, left_caption, right_caption) for the last ball."""
        r = self.round
        if not r:
            return ("Let's play!", "Show a number from 1 to 10 to the camera.", "neutral", "", "")
        h, a, runs, wicket = r["human"], r["ai"], r["runs"], r["wicket"]
        s = "" if runs == 1 else "s"
        if r["role"] == "toss":
            return (f"Toss: you {h} + AI {a} = {h + a}", "The toss decides who picks bat or bowl.",
                    "neutral", "Your toss number", "AI toss number")
        if r["role"] == "bat":
            if wicket:
                return (f"OUT! The AI bowled {a} too", f"You both showed {h}. Wicket falls.",
                        "bad", "OUT", f"AI bowled {a}")
            if runs >= 6:
                title = f"Great shot! You scored {runs} runs"
            else:
                title = f"Good shot! You scored {runs} run{s}"
            return (title, "Keep the momentum going. Show your next number.", "good",
                    f"+{runs} run{s}", f"AI bowled {a}")
        if wicket:
            return ("Wicket! You got the AI out", f"You both showed {h}.", "good",
                    "Wicket!", f"AI batted {a}")
        return (f"AI scored {runs} run{s}", "Try a different number to stop its next shot.", "bad",
                f"You bowled {h}", f"AI +{runs} run{s}")

    # ---------- sections ----------
    def _header(self, c):
        cv2 = self.cv2
        cv2.rectangle(c, (0, 0), (self.W, 72), self.HEADER, -1)
        cv2.line(c, (0, 72), (self.W, 72), self.BORDER, 2)
        self.rrect(c, 40, 14, 46, 46, 13, self.GREEN)
        cv2.circle(c, (63, 37), 11, self.LIME, -1, cv2.LINE_AA)
        cv2.ellipse(c, (63, 37), (11, 11), 0, 200, 340, self.GREEN, 2, cv2.LINE_AA)
        self.text(c, "Hand Cricket", 104, 48, 0.95, self.INK, 2)

        st = self.state
        done = st.get("role") == "DONE"
        label = ("Session complete" if done else
                 f"Match {st.get('match', '-')} in progress  |  Balls played {st.get('move', 0)}")
        pw = self.width(label, 0.55) + 62
        px = (self.W - pw) // 2
        self.rrect(c, px, 19, pw, 36, 18, (34, 28, 20))
        cv2.circle(c, (px + 25, 37), 7, self.OK if done else self.LIME, -1, cv2.LINE_AA)
        self.text(c, label, px + 44, 43, 0.55, self.LABEL, 1)

        self.rrect(c, 1020, 15, 140, 44, 22, self.HEADER, border=self.BORDER, bw=2)
        self.text(c, "Quit (Q)", 1090, 44, 0.6, self.INK, 1, anchor="c")

    def _score_card(self, c):
        cv2 = self.cv2
        st = self.state
        self.rrect(c, 40, 86, 1120, 100, 28, self.GREEN)

        role = st.get("role", "")
        role_txt = {"BAT": "BATTING", "BOWL": "BOWLING", "TOSS": "TOSS", "DONE": "DONE"}.get(role, "")

        self.text(c, "YOUR SCORE", 74, 112, 0.46, self.GREEN_TEXT, 1, spacing=3)
        w = self.text(c, str(st.get("human", 0)), 72, 172, 2.0, self.INK, 4)
        if role_txt:
            pw = self.width(role_txt, 0.42, 1) + 28
            self.rrect(c, 72 + w + 18, 146, pw, 26, 13, self.LIME)
            self.text(c, role_txt, 72 + w + 18 + pw // 2, 164, 0.42, self.DARK, 1, anchor="c", spacing=2)

        cv2.line(c, (420, 102), (420, 170), self.GREEN_LINE, 1, cv2.LINE_AA)
        self.text(c, "AI SCORE", 452, 112, 0.46, self.GREEN_TEXT, 1, spacing=3)
        self.text(c, str(st.get("ai", 0)), 450, 172, 2.0, self.LIME, 4)

        target = st.get("target")
        stats = [("TARGET", "-" if target is None else str(target), 800),
                 ("BALLS PLAYED", str(st.get("balls", st.get("move", 0))), 935),
                 ("STREAK", str(st.get("streak", 0)), 1068)]
        for label, value, cx in stats:
            self.text(c, label, cx, 112, 0.4, self.GREEN_TEXT, 1, anchor="c", spacing=2)
            self.text(c, value, cx, 168, 1.3, self.INK, 2, anchor="c")

    def _camera_tile(self, frame, landmarks):
        cv2 = self.cv2
        if landmarks is not None:
            h, w, _ = frame.shape
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            for p in pts:
                cv2.circle(frame, p, 4, (240, 220, 20), -1)  # cyan joints
            for a, b in CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)  # white bones
        h, w = frame.shape[:2]
        s = min(self.CAM_MAXW / w, self.CAM_MAXH / h)
        return cv2.resize(frame, (int(w * s), int(h * s)))

    def _cards(self, c, frame, landmarks, headline, color, confidence):
        cv2 = self.cv2
        top, h = 202, 472
        lx, lw = 40, 720
        rx, rw = 820, 340
        left_num = right_num = "-"
        if self.round:
            left_num, right_num = str(self.round["human"]), str(self.round["ai"])
        _, _, mood, left_cap, right_cap = self._round_texts()
        area_y = top + 60

        # ----- YOU card (big camera) -----
        self.rrect(c, lx, top, lw, h, 34, self.LIME_TINT, border=self.LIME_EDGE, bw=2)
        self.text(c, "YOU PLAYED", lx + 26, top + 40, 0.52, self.LABEL, 1, spacing=3)
        self.rrect(c, lx + lw - 96, top + 18, 70, 32, 16, self.LIME)
        self.text(c, "YOU", lx + lw - 61, top + 40, 0.5, self.DARK, 1, anchor="c", spacing=2)
        self.text(c, left_num, lx + 26, top + 118, 2.4, self.INK, 4)

        if self.history:
            self.text(c, "RECENT", lx + 26, top + 164, 0.4, self.LABEL, 1, spacing=3)
            for i, rec in enumerate(reversed(self.history)):
                y = top + 194 + i * 30
                self.text(c, f"{rec['human']} v {rec['ai']}", lx + 26, y, 0.5, self.INK, 1)
                self.text(c, rec["text"], lx + 140, y, 0.5, self.OK if rec["good"] else self.RED, 1, anchor="r")

        tile = self._camera_tile(frame, landmarks)
        th_, tw_ = tile.shape[:2]
        area_x = lx + lw - 24 - self.CAM_MAXW
        bx = area_x + (self.CAM_MAXW - tw_) // 2
        by = area_y + (self.CAM_MAXH - th_) // 2
        self.rrect(c, bx - 6, by - 6, tw_ + 12, th_ + 12, 28, self.WHITE)
        self.paste_rounded(c, tile, bx, by, 22)
        tone = self.tone(color)
        self.outline(c, bx, by, tw_, th_, 22, tone, 3)

        badge = headline + (f"  {confidence:.0%}" if confidence else "")
        badge = self.fit(badge, tw_ - 64, 0.55)
        bw = self.width(badge, 0.55) + 42
        self.rrect(c, bx + 14, by + 14, bw, 32, 16, (8, 6, 4))
        cv2.circle(c, (bx + 31, by + 30), 6, tone, -1, cv2.LINE_AA)
        self.text(c, badge, bx + 45, by + 36, 0.55, self.INK, 1)

        cap_color = self.RED if (self.round and self.round["role"] == "bat" and self.round["wicket"]) else self.INK
        self.text(c, left_cap or "Show your number", area_x + self.CAM_MAXW // 2, top + h - 20,
                  0.62, cap_color, 1, anchor="c")

        # ----- VS badge -----
        vx, vy = (lx + lw + rx) // 2, top + h // 2
        cv2.circle(c, (vx, vy), 29, self.BORDER, -1, cv2.LINE_AA)
        cv2.circle(c, (vx, vy), 27, self.HEADER, -1, cv2.LINE_AA)
        self.text(c, "VS", vx, vy + 8, 0.62, self.INK, 2, anchor="c")

        # ----- AI card -----
        self.rrect(c, rx, top, rw, h, 34, self.WHITE, border=self.BORDER, bw=2)
        self.text(c, "AI PLAYED", rx + 26, top + 40, 0.52, self.LABEL, 1, spacing=3)
        self.rrect(c, rx + rw - 96, top + 18, 70, 32, 16, self.GREEN)
        self.text(c, "AI", rx + rw - 61, top + 40, 0.5, self.LIME, 1, anchor="c", spacing=2)
        self.text(c, right_num, rx + 26, top + 118, 2.4, self.INK, 4)

        aw, ah = 280, 210
        ax = rx + (rw - aw) // 2
        ay = area_y + (self.CAM_MAXH - ah) // 2 + 14
        self.rrect(c, ax - 6, ay - 6, aw + 12, ah + 12, 30, (28, 22, 15), border=self.BORDER, bw=2)
        self.rrect(c, ax, ay, aw, ah, 24, self.WHITE)
        n = self.round["ai"] if self.round else None
        if n is None:
            self.text(c, "?", ax + aw // 2, ay + 130, 4.2, self.BORDER, 7, anchor="c")
        else:
            self.text(c, str(n), ax + aw // 2, ay + 118, 4.8, self.LIME, 8, anchor="c")
            for i in range(10):
                px = ax + (aw - 9 * 20) // 2 + i * 20
                if i < n:
                    cv2.circle(c, (px, ay + 176), 6, self.LIME, -1, cv2.LINE_AA)
                else:
                    cv2.circle(c, (px, ay + 176), 6, self.BORDER, 2, cv2.LINE_AA)
        self.text(c, right_cap or "Waiting for first ball", rx + rw // 2, top + h - 20, 0.62,
                  self.INK, 1, anchor="c")

    def _feedback(self, c):
        cv2 = self.cv2
        title, sub, mood, _, _ = self._round_texts()
        tw = max(self.width(title, 0.85, 2), self.width(sub, 0.55, 1))
        total = 64 + tw
        x0 = (self.W - total) // 2
        cy = 708
        fill = {"good": self.LIME, "bad": self.SALMON, "neutral": self.LIME}[mood]
        cv2.circle(c, (x0 + 24, cy), 24, fill, -1, cv2.LINE_AA)
        glyph = {"good": "+", "bad": "!", "neutral": "*"}[mood]
        self.text(c, glyph, x0 + 24, cy + 9, 0.95, self.DARK, 2, anchor="c")
        self.text(c, title, x0 + 64, cy - 2, 0.85, self.INK, 2)
        self.text(c, sub, x0 + 64, cy + 26, 0.55, self.MUTED, 1)
        cv2.line(c, (40, 752), (1160, 752), self.BORDER, 2)

    def _tiles(self, c, subline, detected, locked):
        self.text(c, "Choose your move", 40, 788, 1.0, self.INK, 2)
        hint = subline or "Show a number from 1 to 10"
        self.text(c, self.fit(hint, 520, 0.58), 1160, 786, 0.58, self.MUTED, 1, anchor="r")

        gap, tw, th, y = 10, 103, 60, 802
        for i, n in enumerate(NUMBERS):
            x = 40 + i * (tw + gap)
            if detected == n:
                edge = self.OK if locked else self.LIME
                self.rrect(c, x, y, tw, th, 20, self.LIME_TINT, border=edge, bw=3)
            else:
                self.rrect(c, x, y, tw, th, 20, self.WHITE, border=self.BORDER, bw=2)
            self.text(c, str(n), x + tw // 2, y + 37, 1.2, self.INK, 2, anchor="c")
            self.text(c, "run" if n == 1 else "runs", x + tw // 2, y + 53, 0.4, self.MUTED, 1, anchor="c")

    def _modal(self, c, headline, color, subline, hint, popup=None):
        cv2, np = self.cv2, self.np
        dark = np.full_like(c, (0, 0, 0))
        c[:] = cv2.addWeighted(c, 0.30, dark, 0.70, 0)
        tone = self.tone(color)

        if popup is None:  # simple result card
            w, h = 760, 270
            x, y = (self.W - w) // 2, (self.H - h) // 2
            self.rrect(c, x, y, w, h, 36, self.WHITE, border=self.BORDER, bw=2)
            self.rrect(c, x + w // 2 - 40, y + 30, 80, 8, 4, tone)
            self.text(c, self.fit(headline, w - 60, 1.15, 3), x + w // 2, y + 120, 1.15, tone, 3, anchor="c")
            if subline:
                self.text(c, self.fit(subline, w - 60, 0.8, 1), x + w // 2, y + 170, 0.8, self.INK, 1, anchor="c")
            if hint:
                self.text(c, hint, x + w // 2, y + h - 36, 0.55, self.MUTED, 1, anchor="c")
            return

        options = popup.get("options")
        w, h = 760, (412 if options else 372)
        x, y = (self.W - w) // 2, (self.H - h) // 2
        cx = x + w // 2
        self.rrect(c, x, y, w, h, 40, self.WHITE, border=self.BORDER, bw=2)
        cv2.circle(c, (cx, y + 92), 58, self.LIME_TINT, -1, cv2.LINE_AA)
        cv2.circle(c, (cx, y + 92), 58, self.LIME_EDGE, 2, cv2.LINE_AA)
        self._icon(c, popup.get("icon"), cx, y + 92)
        self.text(c, popup.get("kicker", ""), cx, y + 182, 0.55, tone, 1, anchor="c", spacing=4)
        self.text(c, self.fit(headline, w - 60, 1.4, 3), cx, y + 232, 1.4, self.INK, 3, anchor="c")
        if subline:
            self.text(c, self.fit(subline, w - 60, 0.78, 1), cx, y + 272, 0.78, self.INK, 1, anchor="c")
        if popup.get("detail"):
            self.text(c, self.fit(popup["detail"], w - 60, 0.6, 1), cx, y + 304, 0.6, self.MUTED, 1, anchor="c")
        if options:
            bw_, bh_, gap = 250, 56, 28
            bx0 = cx - (len(options) * bw_ + (len(options) - 1) * gap) // 2
            by0 = y + 302
            for i, label in enumerate(options):
                bx = bx0 + i * (bw_ + gap)
                self.button_rects.append((bx, by0, bx + bw_, by0 + bh_))
                hot = self.hover == i
                self.rrect(c, bx, by0, bw_, bh_, 28,
                           self.LIME if hot else self.LIME_TINT,
                           border=self.LIME if hot else self.LIME_EDGE, bw=3)
                cv2.circle(c, (bx + 32, by0 + bh_ // 2), 15, self.DARK if hot else self.LIME, -1, cv2.LINE_AA)
                self.text(c, str(i + 1), bx + 32, by0 + bh_ // 2 + 6, 0.55,
                          self.LIME if hot else self.DARK, 1, anchor="c")
                self.text(c, label, bx + bw_ // 2 + 16, by0 + bh_ // 2 + 9, 0.85,
                          self.DARK if hot else self.INK, 2, anchor="c")
        if hint:
            self.text(c, hint, cx, y + h - (22 if options else 38), 0.48, self.MUTED, 1, anchor="c")
        prog = popup.get("progress")
        if prog is not None:
            bw_ = w - 160
            self.rrect(c, x + 80, y + h - 22, bw_, 8, 4, self.BORDER)
            fw = int(bw_ * max(0.0, min(1.0, prog)))
            if fw >= 8:
                self.rrect(c, x + 80, y + h - 22, fw, 8, 4, tone)

    # ---------- public ----------
    def render(self, frame, landmarks, headline, color, subline="", confidence=None,
               hint="", detected=None, locked=False, modal=False, popup=None):
        np = self.np
        self.button_rects = []
        c = np.full((self.H, self.W, 3), self.BG, np.uint8)
        self._header(c)
        self._score_card(c)
        self._cards(c, frame, landmarks, headline, color, confidence)
        self._feedback(c)
        self._tiles(c, "" if modal else subline, detected, locked)
        if modal:
            self._modal(c, headline, color, subline, hint, popup)
        if self.SCALE != 1.0:
            c = self.cv2.resize(c, None, fx=self.SCALE, fy=self.SCALE, interpolation=self.cv2.INTER_AREA)
        return c


# ======================================================================
# Gesture input (webcam + MediaPipe + your trained classifier)
# ======================================================================
def resolve_path(relative):
    for base in (ROOT, Path.cwd()):
        candidate = base / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing file: {relative}\nLooked in {ROOT} and {Path.cwd()}. "
        "Check your models folder."
    )


class GestureInput:
    WINDOW = "Hand Cricket"

    def __init__(self):
        try:
            import cv2
            import joblib
            import mediapipe as mp
            import numpy as np
        except ImportError as exc:
            raise SystemExit(
                f"Missing package ({exc}). Install with:\n"
                "  pip install opencv-python mediapipe numpy joblib scikit-learn"
            )
        self.cv2, self.mp, self.np = cv2, mp, np

        hand_model_path = resolve_path(HAND_MODEL_PATH)
        classifier_path = resolve_path(CLASSIFIER_PATH)

        artifact = joblib.load(classifier_path)
        if not isinstance(artifact, dict) or "model" not in artifact:
            raise ValueError(
                "The saved model is not in the expected format "
                "(dict with a 'model' entry). Run step5_train_model.py again."
            )
        self.model = artifact["model"]
        if artifact.get("feature_count") != 63 or getattr(
            self.model, "n_features_in_", None
        ) != 63:
            raise ValueError(
                f"Feature mismatch: artifact feature_count={artifact.get('feature_count')}, "
                f"model expects {getattr(self.model, 'n_features_in_', 'unknown')} features. "
                "Retrain using step5_train_model.py."
            )

        classes = [int(c) for c in self.model.classes_]
        print("Gesture classifier loaded. Recognized labels:", classes)
        unusable = [c for c in classes if c not in NUMBERS]
        if unusable:
            print(f"Note: labels {unusable} are outside 1-10 and will be ignored.")
        missing = [n for n in NUMBERS if n not in classes]
        if missing:
            print(f"Note: your model has no gesture for {missing}, "
                  "so those numbers can never be played by hand.")

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = HandLandmarker.create_from_options(options)

        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self.camera = cv2.VideoCapture(CAMERA_INDEX, backend)
        if not self.camera.isOpened():
            self.detector.close()
            raise RuntimeError(
                "Cannot open webcam. On macOS, enable camera access for your "
                "Terminal or IDE in System Settings > Privacy & Security > Camera."
            )

        self._last_ts = 0
        self.ui = CricketUI(cv2, np)

    def setup_form(self):
        """In-window login/setup form: player ID and AI difficulty."""
        cv2, np, ui = self.cv2, self.np, self.ui
        player_id = ""
        difficulties = ["easy", "medium", "hard", "expert"]
        selected = 2
        mouse = {"x": -1, "y": -1, "click": False}
        scale = ui.SCALE

        def on_mouse(event, x, y, flags, param):
            mouse["x"], mouse["y"] = int(x / scale), int(y / scale)
            if event == cv2.EVENT_LBUTTONUP:
                mouse["click"] = True

        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WINDOW, on_mouse)
        while True:
            canvas = np.full((ui.H, ui.W, 3), ui.BG, np.uint8)
            cv2.rectangle(canvas, (0, 0), (ui.W, 76), ui.HEADER, -1)
            ui.text(canvas, "HAND CRICKET", 70, 50, 1.0, ui.INK, 2)
            ui.text(canvas, "PLAYER LOGIN", ui.W // 2, 155, 1.1, ui.LIME, 3, anchor="c")
            ui.text(canvas, "Use your Player ID. Returning IDs resume their own learning profile.",
                    ui.W // 2, 196, 0.57, ui.MUTED, 1, anchor="c")

            field = (190, 235, 1010, 300)
            ui.rrect(canvas, *field[:2], field[2] - field[0], field[3] - field[1], 16, ui.WHITE,
                     border=ui.LIME_EDGE, bw=3)
            ui.text(canvas, "PLAYER ID", 220, 260, 0.43, ui.MUTED, 1, spacing=2)
            ui.text(canvas, player_id + ("|" if int(time.time() * 2) % 2 == 0 else ""),
                    220, 289, 0.75, ui.INK, 2)

            ui.text(canvas, "CHOOSE AI DIFFICULTY", ui.W // 2, 365, 0.62, ui.INK, 2, anchor="c")
            desc = ["Mostly random", "Frequency-based", "Adaptive patterns", "Full adaptive strategy"]
            boxes = []
            bw, gap, by = 205, 18, 395
            total = 4 * bw + 3 * gap
            bx0 = (ui.W - total) // 2
            for i, name in enumerate(difficulties):
                x = bx0 + i * (bw + gap)
                box = (x, by, x + bw, by + 100)
                boxes.append(box)
                active = i == selected
                ui.rrect(canvas, x, by, bw, 100, 18, ui.LIME_TINT if active else ui.WHITE,
                         border=ui.LIME if active else ui.BORDER, bw=3 if active else 2)
                ui.text(canvas, name.upper(), x + bw // 2, by + 38, 0.62,
                        ui.LIME if active else ui.INK, 2, anchor="c")
                ui.text(canvas, desc[i], x + bw // 2, by + 69, 0.42,
                        ui.INK if active else ui.MUTED, 1, anchor="c")

            start_box = (430, 555, 770, 620)
            ui.rrect(canvas, start_box[0], start_box[1], start_box[2] - start_box[0],
                     start_box[3] - start_box[1], 22, ui.LIME, border=ui.LIME_EDGE, bw=2)
            ui.text(canvas, "START GAME", ui.W // 2, 596, 0.75, ui.DARK, 2, anchor="c")
            ui.text(canvas, "Click the ID field, type your ID, then click Start Game.",
                    ui.W // 2, 670, 0.5, ui.MUTED, 1, anchor="c")
            ui.text(canvas, "Press Q to quit", ui.W // 2, 710, 0.45, ui.MUTED, 1, anchor="c")
            display = cv2.resize(canvas, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA) if scale != 1 else canvas
            cv2.imshow(self.WINDOW, display)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                raise QuitRequested
            if key in (8, 127):
                player_id = player_id[:-1]
            elif key in (10, 13):
                if player_id.strip():
                    return player_id.strip(), difficulties[selected]
            elif 32 <= key <= 126 and len(player_id) < 32:
                ch = chr(key)
                if ch.isalnum() or ch in "_.-":
                    player_id += ch
            if mouse["click"]:
                x, y = mouse["x"], mouse["y"]
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        selected = i
                x1, y1, x2, y2 = start_box
                if x1 <= x <= x2 and y1 <= y <= y2 and player_id.strip():
                    return player_id.strip(), difficulties[selected]
                mouse["click"] = False

    # ---------- internals ----------
    def _features(self, hand_landmarks):
        np = self.np
        points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
        points -= points[0]  # wrist becomes origin
        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 1e-6:
            points /= scale
        return points.flatten().reshape(1, -1)

    def _read_frame(self):
        success, frame = self.camera.read()
        if not success:
            raise RuntimeError("Unable to read webcam frame.")
        return self.cv2.flip(frame, 1)

    def _analyse(self, frame):
        """Return (label, confidence, landmarks); label is None if no hand."""
        cv2, mp, np = self.cv2, self.mp, self.np
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_ts:  # timestamps must strictly increase
            ts = self._last_ts + 1
        self._last_ts = ts
        result = self.detector.detect_for_video(mp_image, ts)
        if not result.hand_landmarks:
            return None, 0.0, None
        landmarks = result.hand_landmarks[0]
        probabilities = self.model.predict_proba(self._features(landmarks))[0]
        best = int(np.argmax(probabilities))
        return int(self.model.classes_[best]), float(probabilities[best]), landmarks

    # ---------- data shown on screen ----------
    def set_state(self, **kwargs):
        """Scoreboard values: match, move, limit, human, ai, target, role, balls, streak."""
        self.ui.state.update(kwargs)

    def set_round(self, role=None, human=None, ai=None, runs=0, wicket=False):
        """Remember the last ball so both cards and the feedback strip can show it."""
        if role is None:
            self.ui.round = None
            return
        self.ui.round = {"role": role, "human": human, "ai": ai,
                         "runs": runs, "wicket": wicket}
        if role == "bat":
            self.ui.history.append({"human": human, "ai": ai, "good": not wicket,
                                    "text": "OUT" if wicket else f"+{runs}"})
        elif role == "bowl":
            self.ui.history.append({"human": human, "ai": ai, "good": wicket,
                                    "text": "WKT" if wicket else f"-{runs}"})

    def _show(self, frame, landmarks, headline, color, subline="", confidence=None,
              hint="Press Q to save & quit", quit_on_q=True,
              detected=None, locked=False, modal=False, popup=None):
        canvas = self.ui.render(frame, landmarks, headline, color, subline, confidence,
                                hint, detected, locked, modal, popup)
        self.cv2.imshow(self.WINDOW, canvas)
        key = self.cv2.waitKey(1) & 0xFF
        if quit_on_q and key == ord("q"):
            raise QuitRequested
        return key

    def flash(self, headline, color, subline="", seconds=1.2, modal=False):
        """Show a banner for a moment. modal=True dims the screen and shows a result card."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self._show(self._read_frame(), None, headline, color, subline,
                       hint="Next match starting..." if modal else "", modal=modal)

    def popup(self, kicker, headline, subline="", detail="", color=(0, 255, 255),
              icon=None, seconds=3.0):
        """In-window popup over the live view (toss result, innings change...). SPACE skips it."""
        start = time.monotonic()
        end = start + seconds
        while True:
            now = time.monotonic()
            if now >= end:
                return
            key = self._show(self._read_frame(), None, headline, color, subline,
                             hint="Press SPACE to continue", modal=True,
                             popup={"kicker": kicker, "detail": detail, "icon": icon,
                                    "progress": (end - now) / seconds})
            if key in (32, 13):
                return

    def choose(self, kicker, headline, subline, options, color=(0, 255, 255), icon=None):
        """
        Popup with clickable buttons. `options` is a list of (value, label).
        Click a button or press 1 / 2. Returns the chosen value. Q quits.
        """
        cv2 = self.cv2
        ui = self.ui
        labels = [label for _, label in options]
        clicked = [None]

        def on_mouse(event, x, y, flags, param):
            idx = ui.hit(x / ui.SCALE, y / ui.SCALE)
            if event == cv2.EVENT_MOUSEMOVE:
                ui.hover = idx
            elif event == cv2.EVENT_LBUTTONUP and idx is not None:
                clicked[0] = idx

        ui.hover = None
        registered = False
        chosen = None
        while chosen is None:
            key = self._show(self._read_frame(), None, headline, color, subline,
                             hint="Click an option or press 1 / 2", modal=True,
                             popup={"kicker": kicker, "icon": icon, "options": labels,
                                    "progress": None})
            if not registered:  # the window exists after the first imshow
                cv2.setMouseCallback(self.WINDOW, on_mouse)
                registered = True
            if clicked[0] is not None:
                chosen = clicked[0]
            elif ord("1") <= key < ord("1") + len(options):
                chosen = key - ord("1")
        cv2.setMouseCallback(self.WINDOW, lambda *args: None)
        ui.hover = None
        return options[chosen][0]

    def hold_final(self, headline, color, subline=""):
        """Keep the final result on screen until any key is pressed in the window."""
        while True:
            key = self._show(self._read_frame(), None, headline, color, subline,
                             hint="Press any key to close", quit_on_q=False, modal=True)
            if key != 255:
                return

    def _wait_for_hand_down(self, subline):
        clear = 0
        while clear < HAND_DOWN_FRAMES:
            frame = self._read_frame()
            _, _, landmarks = self._analyse(frame)
            clear = clear + 1 if landmarks is None else 0
            if landmarks is not None:
                self._show(frame, landmarks, "Lower your hand...", (0, 200, 255), subline)
            else:
                self._show(frame, None, "Get ready...", (0, 200, 255), subline)

    # ---------- public API ----------
    def wait_for_number(self, subline):
        """Block until a stable number gesture (1-10) is seen; return it."""
        self._wait_for_hand_down(subline)
        for _ in range(3):  # drop stale buffered frames
            self.camera.read()

        history = deque(maxlen=SMOOTHING_WINDOW)
        while True:
            frame = self._read_frame()
            label, confidence, landmarks = self._analyse(frame)

            usable = label in NUMBERS and confidence >= CONFIDENCE_THRESHOLD
            history.append(label if usable else None)

            votes_list = [p for p in history if p is not None]
            if votes_list:
                candidate, votes = Counter(votes_list).most_common(1)[0]
                if votes >= MIN_VOTES and label == candidate:
                    self._show(frame, landmarks, f"Locked: {candidate}",
                               (0, 255, 0), subline, confidence,
                               detected=candidate, locked=True)
                    self.cv2.waitKey(400)
                    return candidate

            if landmarks is None:
                self._show(frame, None, "Show your number", (0, 0, 255), subline)
            else:
                self._show(frame, landmarks, "Recognizing...", (0, 200, 255),
                           subline, confidence,
                           detected=label if usable else None)

    def close(self):
        try:
            self.camera.release()
            self.detector.close()
            self.cv2.destroyAllWindows()
            self.cv2.waitKey(1)
        except Exception:
            pass


# ======================================================================
# Integrated ML adapter: webcam/UI from this file + predictor/strategy engine
# ======================================================================
class AdaptivePlayerModel:
    """Connect the existing gesture UI to the modular ML and strategy engine."""

    def __init__(self, difficulty="expert", player_id="local_player"):
        self.difficulty = difficulty
        self.player_id = player_id
        self.predictor = MovePredictor(
            store_dir=str(ROOT / "player_profiles"),
            config=Config(legal_numbers=tuple(NUMBERS)))
        self.strategy = StrategyEngine(temperature=0.8, exploration_floor=0.10)
        self._contexts = {}
        self.last = {"batting": None, "bowling": None, "toss": None}
        self.last_ai = {"batting": None, "bowling": None, "toss": None}
        self.observations = Counter()
        self.feedback = {r: {n: Counter() for n in NUMBERS}
                         for r in ("batting", "bowling")}

    def context_for(self, role, score=0, target=None, wickets_left=1):
        ctx = self._contexts.get(role, {})
        ctx.update({
            "session_id": ctx.get("session_id", "gesture_session"),
            "game_id": ctx.get("game_id", 1),
            "innings": ctx.get("innings", 1),
            "role": role,
            "score": score,
            "target": target,
            "wickets_left": wickets_left,
            "max_wickets": 1,
        })
        self._contexts[role] = ctx
        return dict(ctx)

    def probabilities(self, role, previous=None, previous_ai=None):
        ctx = dict(self._contexts.get(role) or self.context_for(role))
        output = self.predictor.predict(self.player_id, ctx, difficulty=self.difficulty)
        raw = output.get("predicted_probabilities", {})
        probs = {n: float(raw.get(str(n), 0.1)) for n in NUMBERS}
        total = sum(probs.values()) or 1.0
        return {n: probs[n] / total for n in NUMBERS}

    def observe(self, role, number, ai_number=None):
        if role not in ("batting", "bowling") or number not in NUMBERS:
            return
        ctx = self.context_for(role)
        self.predictor.update(self.player_id, observed_move=number,
                              context=ctx, my_move=ai_number)
        self.last[role] = number
        self.last_ai[role] = ai_number
        self.observations[role] += 1

    def save(self, path=None):
        # A player may quit from the setup screen before making a prediction.
        # Ensure the profile is loaded/created before asking the store to save it.
        self.predictor.profile(self.player_id)
        self.predictor.save(self.player_id)


# ======================================================================
# Adaptive model (unchanged)
# ======================================================================
class LegacyAdaptivePlayerModel:
    """
    Online, role-specific model of the human player's number choices.

    Predictors:
      1. Bayesian-smoothed long-run frequency
      2. Recent frequency (adapts to changing habits)
      3. First-order Markov transitions: previous human number -> next number
      4. Second-order Markov transitions: previous two human numbers -> next
      5. Response model: previous AI number -> human's next number

    The model updates after every valid gameplay move and persists to JSON.
    Older version-1 profiles are loaded where possible.
    """

    PROFILE_VERSION = 2
    RECENT_WINDOW = 60
    ALPHA = 1.0

    def __init__(self):
        self.counts = {role: Counter() for role in ROLES}
        self.transitions = {role: defaultdict(Counter) for role in ROLES}
        self.second_transitions = {role: defaultdict(Counter) for role in ROLES}
        self.ai_responses = {role: defaultdict(Counter) for role in ROLES}
        self.recent = {role: deque(maxlen=self.RECENT_WINDOW) for role in ROLES}
        self.last = {role: None for role in ROLES}
        self.last_ai = {role: None for role in ROLES}
        self.observations = Counter()
        self.feedback = {
            role: {n: Counter() for n in NUMBERS}
            for role in ("batting", "bowling")
        }

    @staticmethod
    def _distribution(counts, alpha=1.0):
        """Return a smoothed distribution over all legal numbers."""
        total = sum(counts.get(n, 0) for n in NUMBERS)
        denom = total + len(NUMBERS) * alpha
        return {n: (counts.get(n, 0) + alpha) / denom for n in NUMBERS}

    def observe(self, role, number, ai_number=None):
        """Record a human move and the AI number that was played on that turn."""
        if role not in ROLES or number not in NUMBERS:
            return

        history = list(self.recent[role])
        previous = history[-1] if history else self.last[role]
        previous_two = tuple(history[-2:]) if len(history) >= 2 else None
        previous_ai = self.last_ai[role]

        self.counts[role][number] += 1
        if previous is not None:
            self.transitions[role][previous][number] += 1
        if previous_two is not None and len(previous_two) == 2:
            self.second_transitions[role][previous_two][number] += 1
        # Learn how the human responds to the AI's *previous* move, not the
        # current AI move (which would leak information from the same turn).
        if previous_ai is not None:
            self.ai_responses[role][previous_ai][number] += 1

        self.recent[role].append(number)
        self.last[role] = number
        if ai_number in NUMBERS:
            self.last_ai[role] = ai_number
        self.observations[role] += 1

    def probabilities(self, role, previous=None, previous_ai=None):
        """
        Estimate P(human chooses n next | role and observed history).
        Evidence-poor predictors are omitted rather than trusted too early.
        """
        if role not in ROLES:
            return {n: 1.0 / len(NUMBERS) for n in NUMBERS}

        alpha = self.ALPHA
        history = list(self.recent[role])
        if previous is None:
            previous = history[-1] if history else self.last[role]
        if previous_ai is None:
            previous_ai = self.last_ai[role]

        components = []
        # Baseline prior: learns long-term preferences but never assigns zero.
        components.append((0.22, self._distribution(self.counts[role], alpha)))

        # Short-term behavior: useful when the player changes strategy.
        recent_counts = Counter(history)
        components.append((0.28, self._distribution(recent_counts, alpha)))

        if previous is not None:
            row = self.transitions[role].get(previous, Counter())
            row_total = sum(row.values())
            if row_total >= 3:
                components.append((0.23, self._distribution(row, alpha)))

        if len(history) >= 2:
            key = (history[-2], history[-1])
            row2 = self.second_transitions[role].get(key, Counter())
            row2_total = sum(row2.values())
            if row2_total >= 4:
                components.append((0.15, self._distribution(row2, alpha)))

        if previous_ai is not None:
            response_row = self.ai_responses[role].get(previous_ai, Counter())
            response_total = sum(response_row.values())
            if response_total >= 3:
                components.append((0.12, self._distribution(response_row, alpha)))

        weight_total = sum(weight for weight, _ in components)
        probs = {
            n: sum(weight * dist[n] for weight, dist in components) / weight_total
            for n in NUMBERS
        }
        # Normalize against floating-point drift.
        total = sum(probs.values())
        return {n: probs[n] / total for n in NUMBERS}

    def best_prediction(self, role, previous=None, previous_ai=None):
        probs = self.probabilities(role, previous, previous_ai)
        best_p = max(probs.values())
        tied = [n for n, p in probs.items() if abs(p - best_p) < 1e-12]
        return random.choice(tied), probs

    def save(self, path=PROFILE_PATH):
        payload = {
            "version": self.PROFILE_VERSION,
            "saved_at": now_iso(),
            "counts": {
                r: {str(n): self.counts[r][n] for n in NUMBERS}
                for r in ROLES
            },
            "transitions": {
                r: {
                    str(prev): {str(n): c for n, c in row.items()}
                    for prev, row in self.transitions[r].items()
                }
                for r in ROLES
            },
            "second_transitions": {
                r: {
                    f"{pair[0]},{pair[1]}": {str(n): c for n, c in row.items()}
                    for pair, row in self.second_transitions[r].items()
                }
                for r in ROLES
            },
            "ai_responses": {
                r: {
                    str(prev_ai): {str(n): c for n, c in row.items()}
                    for prev_ai, row in self.ai_responses[r].items()
                }
                for r in ROLES
            },
            "recent": {r: list(self.recent[r]) for r in ROLES},
            "last": self.last,
            "last_ai": self.last_ai,
            "observations": dict(self.observations),
            "feedback": {
                r: {str(n): dict(self.feedback[r][n]) for n in NUMBERS}
                for r in ("batting", "bowling")
            },
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
                    if int(n) in NUMBERS:
                        model.counts[role][int(n)] = int(c)

                for prev, row in data.get("transitions", {}).get(role, {}).items():
                    if int(prev) in NUMBERS:
                        model.transitions[role][int(prev)] = Counter({
                            int(n): int(c) for n, c in row.items() if int(n) in NUMBERS
                        })

                for pair_text, row in data.get("second_transitions", {}).get(role, {}).items():
                    try:
                        a, b = (int(x) for x in pair_text.split(","))
                        if a in NUMBERS and b in NUMBERS:
                            model.second_transitions[role][(a, b)] = Counter({
                                int(n): int(c) for n, c in row.items() if int(n) in NUMBERS
                            })
                    except (ValueError, TypeError):
                        continue

                for prev_ai, row in data.get("ai_responses", {}).get(role, {}).items():
                    if int(prev_ai) in NUMBERS:
                        model.ai_responses[role][int(prev_ai)] = Counter({
                            int(n): int(c) for n, c in row.items() if int(n) in NUMBERS
                        })

                model.recent[role].extend(
                    int(n) for n in data.get("recent", {}).get(role, [])[-cls.RECENT_WINDOW:]
                    if int(n) in NUMBERS
                )
                last = data.get("last", {}).get(role)
                model.last[role] = int(last) if last is not None and int(last) in NUMBERS else (
                    model.recent[role][-1] if model.recent[role] else None
                )
                last_ai = data.get("last_ai", {}).get(role)
                model.last_ai[role] = int(last_ai) if last_ai is not None and int(last_ai) in NUMBERS else None

                # Rebuild second-order history from the recent moves for profiles
                # that predate second-order persistence.
                if not model.second_transitions[role] and len(model.recent[role]) >= 3:
                    hist = list(model.recent[role])
                    for i in range(2, len(hist)):
                        model.second_transitions[role][(hist[i-2], hist[i-1])][hist[i]] += 1

            model.observations.update(data.get("observations", {}))
            for role in ("batting", "bowling"):
                for n, stats in data.get("feedback", {}).get(role, {}).items():
                    if int(n) in NUMBERS:
                        model.feedback[role][int(n)].update(stats)
        except (ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
            print(f"Warning: profile could not be loaded ({exc}); starting with a fresh model.")
        return model

def choose_batting_number(model, target, current_score, wickets_left=1):
    context = model.context_for("bowling", score=current_score, target=target,
                                wickets_left=wickets_left)
    probs = model.probabilities("bowling", model.last["bowling"], model.last_ai["bowling"])
    ai_number = model.strategy.select_ai_move("bat", {str(k): v for k, v in probs.items()},
                                              context, difficulty=model.difficulty)
    return ai_number, probs, {n: 0.0 for n in NUMBERS}


def choose_bowling_number(model):
    context = model.context_for("batting")
    probs = model.probabilities("batting", model.last["batting"], model.last_ai["batting"])
    ai_number = model.strategy.select_ai_move("bowl", {str(k): v for k, v in probs.items()},
                                              context, difficulty=model.difficulty)
    return ai_number, probs


def append_csv(path, fields, row):
    """Append rows while safely upgrading older CSV headers when fields are added."""
    if path.exists() and path.stat().st_size:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames or []
            old_rows = list(reader)
        if old_fields != fields:
            merged_fields = list(old_fields)
            for field in fields:
                if field not in merged_fields:
                    merged_fields.append(field)
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=merged_fields)
                writer.writeheader()
                writer.writerows(old_rows)
            fields = merged_fields
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


MOVE_FIELDS = [
    "timestamp", "player_id", "session_move", "match_id", "innings", "batter", "human_role",
    "human_number", "ai_number", "runs", "wicket", "human_score", "ai_score",
    "target", "human_predicted_probability", "ai_action_probability",
]
MATCH_FIELDS = ["timestamp", "player_id", "match_id", "winner", "human_score", "ai_score", "human_won_toss", "toss_choice"]


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


# ======================================================================
# Game loop
# ======================================================================
def run_session(model, gesture):
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_moves = 0
    match_id = 0
    human_match_wins = 0
    ai_match_wins = 0
    draws = 0
    total_matches = 0
    total_human_runs = 0
    total_ai_runs = 0
    streak = 0  # consecutive balls you batted without getting out

    print("\nAdaptive Hand Cricket AI (gesture input)")
    print(f"Player ID: {model.player_id} | Difficulty: {model.difficulty.upper()}")
    print(f"Existing learned observations: batting={model.observations['batting']}, "
          f"bowling={model.observations['bowling']}")
    print("Toss and bat/bowl choices are made in popups in the game window; show numbers to the camera.")
    print("Press Q in the game window to save and quit.\n")

    while True:
        match_id += 1
        gesture.set_round(None)
        gesture.set_state(match=match_id, move=session_moves, limit=None,
                          human=0, ai=0, target=None, role="TOSS",
                          balls=session_moves, streak=streak)
        toss_choice = gesture.choose("TOSS", "Call the toss", "Odd or even?",
                                     [("odd", "Odd"), ("even", "Even")], icon="coin")
        print(f"You called {toss_choice}.")
        print("Show your toss number to the camera...")
        human_toss_number = gesture.wait_for_number("TOSS: show your number")
        print(f"Your toss number: {human_toss_number}")
        ai_toss_number = random.randint(1, 10)
        toss_sum = human_toss_number + ai_toss_number
        human_won_toss = ((toss_sum % 2 == 1 and toss_choice == "odd") or
                          (toss_sum % 2 == 0 and toss_choice == "even"))
        print(f"AI toss number: {ai_toss_number}; sum={toss_sum}.")
        gesture.set_round("toss", human_toss_number, ai_toss_number)
        if human_won_toss:
            toss_winner = "human"
            choice = gesture.choose("TOSS WON", "You won the toss!", "Bat or bowl first?",
                                    [("bat", "Bat"), ("bowl", "Bowl")],
                                    color=(0, 255, 0), icon="coin")
            print(f"You chose to {choice}.")
        else:
            toss_winner = "ai"
            choice = random.choice(["bat", "bowl"])
            print(f"AI won the toss and chooses to {choice}.")
        human_bats_first = (choice == "bat") if toss_winner == "human" else (choice == "bowl")
        gesture.popup("TOSS RESULT",
                      "You won the toss" if toss_winner == "human" else "AI won the toss",
                      f"{'You' if toss_winner == 'human' else 'The AI'} chose to {choice} first",
                      icon="coin", seconds=2.4)
        gesture.set_state(role="BAT" if human_bats_first else "BOWL")
        if human_bats_first:
            gesture.popup("INNINGS 1", "You are batting", "Show a number from 1 to 10",
                          "Match the AI's number and you're out", color=(0, 255, 0),
                          icon="bat", seconds=2.8)
        else:
            gesture.popup("INNINGS 1", "You are bowling", "Show a number from 1 to 10",
                          "Match the AI's number to take a wicket", color=(0, 200, 255),
                          icon="ball", seconds=2.8)

        human_score = 0
        ai_score = 0
        ai_actions = {"batting": [], "bowling": []}
        target = None
        innings = 1
        match_finished = False
        human_role = "batting" if human_bats_first else "bowling"
        print(f"\n--- Match {match_id}: you {'bat' if human_bats_first else 'bowl'} first ---")

        while not match_finished:
            switched = False  # becomes True when this ball ends the first innings
            # First innings: whichever side bats first; second innings chases target.
            human_batting = (human_role == "batting")
            if human_batting:
                ai_number, ai_probs = choose_bowling_number(model)
                ai_actions["bowling"].append(ai_number)
                gesture.set_state(match=match_id, move=session_moves + 1, limit=None,
                                  human=human_score, ai=ai_score, target=target, role="BAT",
                                  balls=session_moves, streak=streak)
                human_number = gesture.wait_for_number(
                    f"Ball {session_moves+1} | YOU BAT")
                print(f"Ball {session_moves+1} | YOU BAT | you showed {human_number}")
                pred_prob = ai_probs[human_number]
                model.observe("batting", human_number, ai_number=ai_number)
                session_moves += 1
                wicket = human_number == ai_number
                runs = 0 if wicket else human_number
                human_score += runs
                streak = 0 if wicket else streak + 1
                message = f"AI bowls {ai_number}. " + ("OUT!" if wicket else f"+{runs} runs.")
                print(message)
                gesture.set_state(human=human_score, balls=session_moves, streak=streak)
                gesture.set_round("bat", human_number, ai_number, runs, wicket)
                flash_info = (f"AI bowled {ai_number}: " + ("OUT!" if wicket else f"+{runs} runs"),
                              (0, 0, 255) if wicket else (0, 255, 0),
                              f"You showed {human_number}")
                append_csv(MOVES_PATH, MOVE_FIELDS, {
                    "timestamp": now_iso(), "player_id": model.player_id, "session_move": session_moves, "match_id": f"{session_id}_{match_id}",
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
                        switched = True
                        print(f"Innings over. Your total: {human_score}; AI target: {target}.")
                    else:
                        match_finished = True
                if target is not None and innings == 2 and ai_score >= target:
                    match_finished = True
            else:
                ai_number, probs, utilities = choose_batting_number(model, target, ai_score)
                ai_actions["batting"].append(ai_number)
                gesture.set_state(match=match_id, move=session_moves + 1, limit=None,
                                  human=human_score, ai=ai_score, target=target, role="BOWL",
                                  balls=session_moves, streak=streak)
                human_number = gesture.wait_for_number(
                    f"Ball {session_moves+1} | YOU BOWL")
                print(f"Ball {session_moves+1} | YOU BOWL | you showed {human_number}")
                pred_prob = probs[human_number]
                model.observe("bowling", human_number, ai_number=ai_number)
                session_moves += 1
                wicket = human_number == ai_number
                runs = 0 if wicket else ai_number
                ai_score += runs
                message = f"AI bats {ai_number}. " + ("AI OUT!" if wicket else f"AI scores {runs}.")
                print(message)
                gesture.set_state(ai=ai_score, balls=session_moves)
                gesture.set_round("bowl", human_number, ai_number, runs, wicket)
                flash_info = (f"AI batted {ai_number}: " + ("AI OUT!" if wicket else f"+{runs} runs"),
                              (0, 255, 0) if wicket else (0, 0, 255),
                              f"You showed {human_number}")
                append_csv(MOVES_PATH, MOVE_FIELDS, {
                    "timestamp": now_iso(), "player_id": model.player_id, "session_move": session_moves, "match_id": f"{session_id}_{match_id}",
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
                        switched = True
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

            gesture.set_state(target=target, human=human_score, ai=ai_score)
            gesture.flash(*flash_info)

            if switched and not match_finished:
                if human_role == "bowling":
                    gesture.set_state(role="BOWL")
                    gesture.popup("INNINGS BREAK", "Now you bowl",
                                  f"You scored {human_score}. The AI needs {target} to win.",
                                  "Match the AI's number to get it out", color=(0, 200, 255),
                                  icon="ball", seconds=3.2)
                else:
                    gesture.set_state(role="BAT")
                    gesture.popup("INNINGS BREAK", "Now you bat",
                                  f"The AI scored {ai_score}. You need {target} to win.",
                                  "Reach the target without matching the AI's number",
                                  color=(0, 255, 0), icon="bat", seconds=3.2)

        # A match is scored only after both innings conclude.
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
                "timestamp": now_iso(), "player_id": model.player_id, "match_id": f"{session_id}_{match_id}", "winner": winner,
                "human_score": human_score, "ai_score": ai_score,
                "human_won_toss": int(human_won_toss), "toss_choice": toss_choice,
            })
            print(f"\nMatch result: {winner.upper()} | You {human_score} - AI {ai_score}")
            print(f"Session results: {human_match_wins} human wins, {ai_match_wins} AI wins, {draws} draws.")
            banner, banner_color = {
                "human": ("MATCH RESULT: YOU WIN!", (0, 255, 0)),
                "ai": ("MATCH RESULT: AI WINS", (0, 0, 255)),
                "draw": ("MATCH RESULT: DRAW", (0, 255, 255)),
            }[winner]
            gesture.flash(banner, banner_color, f"You {human_score} - AI {ai_score}",
                          seconds=4.0, modal=True)
        total_human_runs += human_score
        total_ai_runs += ai_score
        model.save()
        print(f"\nStarting next match automatically. Total balls played: {session_moves}.")

    print("\n=== SESSION SUMMARY ===")
    print(f"Total human gameplay selections: {session_moves}")
    print(f"Completed matches: {total_matches} | Human wins: {human_match_wins} | AI wins: {ai_match_wins} | Draws: {draws}")
    if total_matches:
        print(f"AI win rate among completed matches: {100*ai_match_wins/total_matches:.1f}%")
    show_distribution(model, "batting")
    show_distribution(model, "bowling")
    print(f"\nSaved profile directory: {ROOT / 'player_profiles'} (player-specific profiles)")
    print(f"Saved move log: {MOVES_PATH}")
    print(f"Saved completed-match log: {MATCHES_PATH}")

    # Final result on the camera screen.
    gesture.set_state(role="DONE", move=session_moves, balls=session_moves)
    if total_matches == 0:
        headline, color = "SESSION COMPLETE", (0, 255, 255)
    elif human_match_wins > ai_match_wins:
        headline, color = "SESSION COMPLETE - YOU WIN!", (0, 255, 0)
    elif ai_match_wins > human_match_wins:
        headline, color = "SESSION COMPLETE - AI WINS", (0, 0, 255)
    else:
        headline, color = "SESSION COMPLETE - TIED", (0, 255, 255)
    print("\nPress any key in the camera window to close.")
    gesture.hold_final(headline, color,
                       f"Matches {total_matches} | You {human_match_wins} - AI {ai_match_wins}"
                       f" | Draws {draws}")


def play_session():
    gesture = GestureInput()
    model = None
    try:
        player_id, difficulty = gesture.setup_form()
        model = AdaptivePlayerModel(difficulty=difficulty, player_id=player_id)
        print(f"Logged in as player: {player_id} | Difficulty: {difficulty.upper()}")
        run_session(model, gesture)
    except (QuitRequested, KeyboardInterrupt):
        print("\nExited. Progress saved.")
    finally:
        if model is not None:
            model.save()
        gesture.close()


def main():
    parser = argparse.ArgumentParser(description="Adaptive hand-cricket opponent (gesture input)")
    parser.add_argument("--generate-synthetic", action="store_true",
                        help="generate a separate synthetic pattern benchmark CSV and exit")
    parser.add_argument("--reset-profile", action="store_true",
                        help="delete learned profiles before playing")
    args = parser.parse_args()

    if args.generate_synthetic:
        generate_synthetic_data()
        return
    if args.reset_profile:
        import shutil
        profile_dir = ROOT / "player_profiles"
        if profile_dir.exists():
            shutil.rmtree(profile_dir)
        if PROFILE_PATH.exists():
            PROFILE_PATH.unlink()
        print("Deleted learned profiles. Existing CSV logs were left untouched.")

    play_session()


if __name__ == "__main__":
    main()