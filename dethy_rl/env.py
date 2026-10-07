"""Dethy Mafia environment: a pure-python state machine (no model/GPU dependencies)."""
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

ROLES = ["Sane", "Insane", "Naive", "Paranoid", "Mafia"]

RULES = (
    "Dethy Mafia. 5 players: 1 Mafia and 4 Cops with hidden sanities. "
    "Each night every Cop investigates one player and privately learns 'Mafia' or 'Not Mafia', "
    "and the Mafia kills one other player. "
    "Sane Cops see the truth, Insane Cops see the opposite, Naive Cops always see 'Not Mafia', "
    "Paranoid Cops always see 'Mafia'. Each day all players discuss in turns, then vote to eliminate one player. "
    "Town wins if the Mafia is eliminated; Mafia wins when 2 or fewer players remain.\n"
)


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
        self.transcript: List[str] = [RULES]
        self.private_notes: Dict[int, List[str]] = {i: [] for i in self.roles}
        self.day = 1
        self.phase = "night"
        self.done = False
        self.winner: Optional[str] = None
        self.num_rounds = 0
        self.round = 0
        self.speakers: List[int] = []
        self.transcript.append(f"Night {self.day} begins.")

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
        if self.phase == "night" and self.roles[player_id] == "Mafia":
            return [p for p in self.alive if p != player_id]
        return list(self.alive)

    # ---------------------------------------------------------------- prompts
    def public_transcript(self) -> str:
        return "\n".join(self.transcript)

    def build_prompt(self, player_id: int) -> str:
        """[Shared Public Transcript] + \\nPrivate Role: [Role]. Phase: [Phase]. Action:

        The transcript is byte-identical across agents so vLLM can reuse the prefix KV cache.
        """
        notes = " ".join(self.private_notes[player_id])
        role = f"You are Player_{player_id}. Private Role: {self.roles[player_id]}."
        if notes:
            role += f" Private Notes: {notes}"
        if self.phase == "night":
            verb = "kill" if self.roles[player_id] == "Mafia" else "investigate"
            phase = f"Night {self.day} (answer with the ID of the player to {verb})"
        elif self.phase == "dialogue":
            phase = f"Day {self.day} Dialogue round {self.round}/{self.num_rounds} (say one short public message)"
        else:
            phase = f"Day {self.day} Vote (answer with the ID of the player to eliminate)"
        return f"{self.public_transcript()}\n{role} Phase: {phase}. Action:"

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
        self.transcript.append(f"Day {self.day} begins.")
        self.num_rounds = self.rng.randint(self.min_rounds, self.max_rounds)
        self.round = 0
        self.phase = "dialogue"
        self._next_round()

    def _next_round(self) -> None:
        self.round += 1
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
            if t not in self.alive:
                t = self.rng.choice(self.alive)
            seen = "Mafia" if self._sanity_result(self.roles[pid], t) else "Not Mafia"
            self.private_notes[pid].append(f"Night {self.day}: you investigated Player_{t}: {seen}.")
            if t == mafia:
                r[pid] += self.rw.investigate_mafia_bonus

        victim = targets.get(mafia)
        if victim not in self.alive or victim == mafia:
            victim = self.rng.choice([p for p in self.alive if p != mafia])
        self.alive.remove(victim)
        self.transcript.append(f"Player_{victim} was killed in the night by the Mafia.")
        if len(self.alive) <= 2:
            self.transcript.append("The Mafia has reached parity. Mafia wins!")
            self.done, self.winner = True, "Mafia"
            self._team(r, -self.rw.parity, self.rw.parity)
            return dict(r), True
        self._team(r, -self.rw.night_kill, self.rw.night_kill)
        self._begin_day()
        return dict(r), False

    def step_dialogue(self, player_id: int, message: str) -> None:
        assert self.phase == "dialogue" and self.speakers and self.speakers[0] == player_id
        msg = " ".join(message.split()) or "..."
        self.transcript.append(f"Player_{player_id}: {msg}")
        self.speakers.pop(0)
        if not self.speakers:
            if self.round < self.num_rounds:
                self._next_round()
            else:
                self.phase = "vote"

    def step_vote(self, votes: Dict[int, int]) -> Tuple[Dict[int, float], bool]:
        """Returns (per-player rewards, game_over)."""
        assert self.phase == "vote"
        valid = {p: v for p, v in votes.items() if p in self.alive and v in self.alive}
        for p in self.alive:  # invalid votes get a random valid vote
            valid.setdefault(p, self.rng.choice(self.alive))
        self.transcript.append(
            f"Day {self.day} votes: " + ", ".join(f"Player_{p}->Player_{v}" for p, v in sorted(valid.items()))
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
            self.transcript.append(f"Player_{victim} was eliminated and was the Mafia. Town wins!")
            self.done, self.winner = True, "Town"
            self._team(r, self.rw.lynch_mafia, -self.rw.lynch_mafia)
            return dict(r), True

        self.transcript.append(f"Player_{victim} was eliminated and was not the Mafia.")
        if len(self.alive) <= 2:
            self.transcript.append("The Mafia has reached parity. Mafia wins!")
            self.done, self.winner = True, "Mafia"
            self._team(r, -self.rw.parity, self.rw.parity)
            return dict(r), True

        self._team(r, -self.rw.lynch_town, self.rw.lynch_town)
        self.day += 1
        self.phase = "night"
        self.transcript.append(f"Night {self.day} begins.")
        return dict(r), False
