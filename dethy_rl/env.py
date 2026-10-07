"""Dethy Mafia environment: a pure-python state machine (no model/GPU dependencies)."""
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

ROLES = ["Sane", "Insane", "Naive", "Paranoid", "Mafia"]

RULES = """You are playing Dethy Mafia, a social deduction game. There are 5 players: Player_0 to Player_4.

ROLES
- One player is the Mafia. The other four are Cops. The Mafia wins by surviving; the Cops (the Town) win by eliminating the Mafia.
- Each Cop has a hidden sanity that decides what their investigations tell them:
  * Sane: sees the truth ("Mafia" if the target is the Mafia, otherwise "Not Mafia").
  * Insane: sees the opposite of the truth (the Mafia looks like "Not Mafia", everyone else looks like "Mafia").
  * Naive: always sees "Not Mafia", whoever the target is, so a Naive result tells you nothing.
  * Paranoid: always sees "Mafia", whoever the target is, so a Paranoid result tells you nothing.
- You always know your own role (and sanity, if you are a Cop). Nobody else's role or sanity is revealed unless you choose to say it, and you may lie.

HOW THE GAME RUNS
1. Night: every Cop investigates one player and privately learns "Mafia" or "Not Mafia" (according to their sanity). At the same time the Mafia kills one other player, who is out of the game.
2. Day discussion: the living players talk in 1 to 3 rounds. In each round everyone speaks once, in a random order, and can read everything said so far.
3. Day vote: every living player votes for one living player to eliminate. The player with the most votes is eliminated (ties are broken at random) and everyone is told they were or were not the Mafia.
Then the next night begins.

HOW THE GAME ENDS
- The Cops win as soon as the Mafia is eliminated.
- The Mafia wins when only 2 players are left alive.

WHAT YOU SEE
- Everything above the line starting "You are Player_..." is public and visible to every player.
- That line shows your Private Role and your Private Notes (your own investigation results). Only you can see them.

HOW TO PLAY
- Cops: use your investigation results, but remember your own sanity when you interpret them (for example, an Insane Cop who sees "Not Mafia" should suspect the target). Share useful information, question inconsistent claims, and vote for who you believe is the Mafia. You cannot vote for, investigate or kill yourself.
- Mafia: stay alive. Blend in, cast suspicion on others, and kill players who might expose you.

EXAMPLE (placeholder names, only an illustration of the style: do not repeat its wording, and base what you say on your own game)
Cop X is Sane and privately saw "Mafia" when investigating Z. During discussion X says that Z showed up as Mafia and asks for a vote against Z. Z denies it and accuses X of lying. Another Cop, Y, weighs who sounds more concrete and sides with one of them. At the vote, a player's whole answer is just the digit of the player they pick.
Another Cop, W, is Insane and saw "Not Mafia" for V. Because Insane results are reversed, W privately concludes V is probably the Mafia, and may say V has seemed evasive without revealing the result. A Naive or Paranoid Cop knows their results are meaningless, so they should rely on what others say and may bluff.

HOW TO ANSWER
- Night and Vote phases: answer with a single digit, the ID of a living player other than yourself (for example: 3). Nothing else.
- Discussion phase: write one or two short sentences (about 40 words at most) as yourself. Do not write your own name or "Player_N:" at the start.
- Discussion is only talking. Nobody investigates or kills during the day: investigations and kills happen at night, and the day ends with a vote. Use the discussion to say who you suspect and why, to defend yourself, or to share (truthfully or not) what your investigations showed. Do not ask others to investigate.
"""


@dataclass
class RewardConfig:
    """Team terms are +/- for Town/Mafia. Terminal terms dominate; the rest densify the signal."""
    lynch_mafia: float = 1.0            # game over, Town win
    parity: float = 1.0                 # game over, Mafia win
    lynch_town: float = 0.2             # Town -, Mafia +
    night_kill: float = 0.2             # Town -, Mafia +  (credited to the night step)
    heat: float = 0.3                   # * fraction of day votes on the Mafia; Town +, Mafia -
    vote_mafia_bonus: float = 0.2       # individual: a Cop who voted for the Mafia
    vote_town_penalty: float = 0.05     # individual: a Cop who voted for a non-Mafia
    investigate_mafia_bonus: float = 0.1  # individual: a Cop who investigated the real Mafia


_LABEL = re.compile(r"^\s*(?:Player[_ ]?\d+\s*:\s*)+", re.IGNORECASE)


def clean_message(text: str) -> str:
    """Tidy a model's dialogue before it enters the transcript: drop self-added speaker labels
    ("Player_2: ...") and wrapping quotation marks. Training still uses the raw tokens."""
    msg = " ".join(text.split())
    for _ in range(3):
        msg = _LABEL.sub("", msg).strip()
        if len(msg) >= 2 and msg[0] in "\"“'" and msg[-1] in "\"”'":
            msg = msg[1:-1].strip()
        elif msg[:1] in "\"“" and msg.count("\"") + msg.count("“") + msg.count("”") == 1:
            msg = msg[1:].strip()  # opening quote with no close (message was cut off)
    return msg


class DethyEnv:
    def __init__(self, seed: Optional[int] = None, rewards: Optional[dict] = None,
                 min_rounds: int = 1, max_rounds: int = 3):
        self.rng = random.Random(seed)
        self.rw = RewardConfig(**(rewards or {}))
        self.min_rounds, self.max_rounds = min_rounds, max_rounds
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        roles = ROLES[:]
        self.rng.shuffle(roles)
        self.roles: Dict[int, str] = {i: r for i, r in enumerate(roles)}
        self.alive: List[int] = list(range(len(roles)))
        self.transcript: List[str] = []  # public events only; RULES is sent separately
        self.private_notes: Dict[int, List[str]] = {i: [] for i in self.roles}
        self.day = 1
        self.phase = "night"
        self.done = False
        self.winner: Optional[str] = None
        self.num_rounds = 0
        self.round = 0
        self.speakers: List[int] = []
        self.transcript.append(f"\n=== Night {self.day} ===\n")

    @property
    def mafia_id(self) -> int:
        return next(i for i, r in self.roles.items() if r == "Mafia")

    def acting_players(self) -> List[int]:
        """Players who must act now: everyone at night/vote, the current speaker in dialogue."""
        if self.done:
            return []
        if self.phase == "dialogue":
            return self.speakers[:1]
        return list(self.alive)

    def allowed_targets(self, player_id: int) -> List[int]:
        return [p for p in self.alive if p != player_id]  # never yourself, for kills, checks or votes

    # ---------------------------------------------------------------- prompts
    def public_transcript(self) -> str:
        return "\n".join(self.transcript).strip("\n")

    def build_messages(self, player_id: int, think: bool = False) -> List[Dict[str, str]]:
        """Chat form: the rules are the system message, the rest is one user message."""
        return [{"role": "system", "content": RULES.strip()},
                {"role": "user", "content": self.user_text(player_id, think)}]

    def build_prompt(self, player_id: int, think: bool = False) -> str:
        """Raw-text form (no chat template): rules followed by the user text."""
        return f"{RULES}\n{self.user_text(player_id, think)}"

    def user_text(self, player_id: int, think: bool = False) -> str:
        """[Shared Public Transcript] + \\nPrivate Role: [Role]. Phase: [Phase]. Action:

        The transcript is byte-identical across agents so vLLM can reuse the prefix KV cache.
        """
        # public and identical for every player, so the prefix cache still covers it
        head = (f"{self.public_transcript()}\n\nAlive players: "
                + ", ".join(f"Player_{p}" for p in self.alive) + ".")
        role = f"\nYou are Player_{player_id}.\n\nPrivate Role: {self.roles[player_id]}."
        if self.private_notes[player_id]:
            role += "\n\nPrivate Notes:\n\n" + "\n".join(f"- {n}" for n in self.private_notes[player_id])
        if self.phase == "night":
            verb = "kill" if self.roles[player_id] == "Mafia" else "investigate"
            phase = f"Night {self.day} (answer with the ID of the player to {verb})"
        elif self.phase == "dialogue":
            phase = f"Day {self.day} Dialogue round {self.round}/{self.num_rounds} (say one short public message)"
        else:
            phase = f"Day {self.day} Vote (answer with the ID of the player to eliminate)"
        if think:  # private reasoning stage; the decision stage re-uses this prompt as its prefix
            return (f"{head}\n{role}\n\nPhase: {phase}.\n\n"
                    "First think privately about who is the Mafia, then stop.\n\nThoughts:")
        return f"{head}\n{role}\n\nPhase: {phase}.\n\nAction:"

    # --------------------------------------------------------------- helpers
    def _sanity_result(self, sanity: str, target: int) -> bool:
        """True means the cop sees 'Mafia'."""
        is_mafia = self.roles[target] == "Mafia"
        if sanity == "Sane":
            return is_mafia
        if sanity == "Insane":
            return not is_mafia
        if sanity == "Naive":
            return False
        return True  # Paranoid

    def _team(self, r: Dict[int, float], town: float, mafia: float) -> None:
        for p, role in self.roles.items():
            r[p] += mafia if role == "Mafia" else town

    def _begin_day(self) -> None:
        self.num_rounds = self.rng.randint(self.min_rounds, self.max_rounds)
        self.transcript.append(f"\n=== Day {self.day}: Discussion ({self.num_rounds} round"
                               f"{'s' if self.num_rounds > 1 else ''}) ===\n")
        self.round = 0
        self.phase = "dialogue"
        self._next_round()

    def _next_round(self) -> None:
        self.round += 1
        self.transcript.append(("\n" if self.round > 1 else "") + f"-- Round {self.round} of {self.num_rounds} --")
        self.speakers = self.alive[:]
        self.rng.shuffle(self.speakers)  # fresh random speaking order every round

    # ------------------------------------------------------------------ steps
    def step_night(self, targets: Dict[int, int]) -> Tuple[Dict[int, float], bool]:
        """Cops investigate and the Mafia kills, simultaneously. Returns (rewards, game_over)."""
        assert self.phase == "night"
        r: Dict[int, float] = defaultdict(float)
        mafia = self.mafia_id
        for pid in self.alive:
            if pid == mafia:
                continue
            t = targets.get(pid)
            if t not in self.alive or t == pid:
                t = self.rng.choice(self.allowed_targets(pid))
            seen = "Mafia" if self._sanity_result(self.roles[pid], t) else "Not Mafia"
            self.private_notes[pid].append(f"Night {self.day}: you investigated Player_{t}: {seen}.")
            if t == mafia:
                r[pid] += self.rw.investigate_mafia_bonus

        victim = targets.get(mafia)
        if victim not in self.alive or victim == mafia:
            victim = self.rng.choice([p for p in self.alive if p != mafia])
        self.alive.remove(victim)
        self.transcript.append(f"Player_{victim} was killed in the night by the Mafia and is out of the game.")
        if len(self.alive) <= 2:
            self.transcript.append("\nOnly 2 players are left. The Mafia wins!")
            self.done, self.winner = True, "Mafia"
            self._team(r, -self.rw.parity, self.rw.parity)
            return dict(r), True
        self._team(r, -self.rw.night_kill, self.rw.night_kill)
        self._begin_day()
        return dict(r), False

    def step_dialogue(self, player_id: int, message: str) -> None:
        assert self.phase == "dialogue" and self.speakers and self.speakers[0] == player_id
        msg = clean_message(message) or "..."
        self.transcript.append(f"\nPlayer_{player_id}: {msg}")  # blank line between messages so viewers show separate paragraphs
        self.speakers.pop(0)
        if not self.speakers:
            if self.round < self.num_rounds:
                self._next_round()
            else:
                self.phase = "vote"
                self.transcript.append(f"\n=== Day {self.day}: Vote ===\n")

    def step_vote(self, votes: Dict[int, int]) -> Tuple[Dict[int, float], bool]:
        """Returns (per-player rewards, game_over)."""
        assert self.phase == "vote"
        valid = {p: v for p, v in votes.items() if p in self.alive and v in self.alive and v != p}
        for p in self.alive:  # invalid votes get a random valid vote
            valid.setdefault(p, self.rng.choice(self.allowed_targets(p)))
        self.transcript.append(
            f"Votes:\n" + "\n".join(f"\nPlayer_{p} voted for Player_{v}" for p, v in sorted(valid.items()))
        )
        counts = Counter(valid.values())
        mafia = self.mafia_id
        r: Dict[int, float] = defaultdict(float)

        # dense shaping: how much heat the Mafia took, and which Cops voted well
        heat = self.rw.heat * counts.get(mafia, 0) / len(valid)
        self._team(r, heat, -heat)
        for p, v in valid.items():
            if p != mafia:
                r[p] += self.rw.vote_mafia_bonus if v == mafia else -self.rw.vote_town_penalty

        top = max(counts.values())
        victim = self.rng.choice(sorted(v for v, c in counts.items() if c == top))
        self.alive.remove(victim)

        if self.roles[victim] == "Mafia":
            self.transcript.append(f"\nPlayer_{victim} was eliminated and was the Mafia. The Cops win!")
            self.done, self.winner = True, "Town"
            self._team(r, self.rw.lynch_mafia, -self.rw.lynch_mafia)
            return dict(r), True

        self.transcript.append(f"\nPlayer_{victim} was eliminated and was not the Mafia.")
        if len(self.alive) <= 2:
            self.transcript.append("\nOnly 2 players are left. The Mafia wins!")
            self.done, self.winner = True, "Mafia"
            self._team(r, -self.rw.parity, self.rw.parity)
            return dict(r), True

        self._team(r, -self.rw.lynch_town, self.rw.lynch_town)
        self.day += 1
        self.phase = "night"
        self.transcript.append(f"\n=== Night {self.day} ===\n")
        return dict(r), False
