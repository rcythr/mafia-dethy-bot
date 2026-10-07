"""Dethy Mafia environment: a pure-python state machine (no model/GPU dependencies)."""
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

ROLES = ["Sane", "Insane", "Naive", "Paranoid", "Mafia"]

RULES = (
    "Dethy Mafia. 5 players: 1 Mafia and 4 Cops with hidden sanities. "
    "Each night every Cop investigates one player and privately learns 'Mafia' or 'Not Mafia'. "
    "Sane Cops see the truth, Insane Cops see the opposite, Naive Cops always see 'Not Mafia', "
    "Paranoid Cops always see 'Mafia'. Each day all players discuss, then vote to eliminate one player. "
    "Town wins if the Mafia is eliminated; Mafia wins when 2 or fewer players remain.\n"
)

# Intermediate rewards (Town, Mafia)
R_LYNCH_MAFIA = (1.0, -1.0)
R_LYNCH_TOWN = (-0.2, 0.2)
R_PARITY = (-1.0, 1.0)


class DethyEnv:
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
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
        self.transcript.append(f"Night {self.day} begins.")

    @property
    def mafia_id(self) -> int:
        return next(i for i, r in self.roles.items() if r == "Mafia")

    def acting_players(self) -> List[int]:
        """Players who must act in the current phase."""
        if self.done:
            return []
        if self.phase == "night":
            return [p for p in self.alive if self.roles[p] != "Mafia"]
        return list(self.alive)

    def allowed_targets(self, player_id: int) -> List[int]:
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
        phase = {
            "night": f"Night {self.day} (answer with the ID of the player to investigate)",
            "dialogue": f"Day {self.day} Dialogue (say one short public message)",
            "vote": f"Day {self.day} Vote (answer with the ID of the player to eliminate)",
        }[self.phase]
        return f"{self.public_transcript()}\n{role} Phase: {phase}. Action:"

    # ------------------------------------------------------------------ steps
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

    def step_night(self, targets: Dict[int, int]) -> None:
        assert self.phase == "night"
        for pid in self.acting_players():
            t = targets.get(pid)
            if t not in self.alive:
                t = self.rng.choice(self.alive)
            seen = "Mafia" if self._sanity_result(self.roles[pid], t) else "Not Mafia"
            self.private_notes[pid].append(f"Night {self.day}: you investigated Player_{t}: {seen}.")
        self.transcript.append(f"Day {self.day} begins.")
        self.phase = "dialogue"

    def step_dialogue(self, messages: Dict[int, str]) -> None:
        assert self.phase == "dialogue"
        for pid in self.alive:
            msg = " ".join(messages.get(pid, "").split()) or "..."
            self.transcript.append(f"Player_{pid}: {msg}")
        self.phase = "vote"

    def step_vote(self, votes: Dict[int, int]) -> Tuple[Dict[int, float], bool]:
        """Returns (per-player rewards, game_over)."""
        assert self.phase == "vote"
        valid = {p: v for p, v in votes.items() if p in self.alive and v in self.alive}
        for p in self.alive:  # abstainers/invalid votes get a random valid vote
            valid.setdefault(p, self.rng.choice(self.alive))
        self.transcript.append(
            f"Day {self.day} votes: " + ", ".join(f"Player_{p}->Player_{v}" for p, v in sorted(valid.items()))
        )
        counts = Counter(valid.values())
        top = max(counts.values())
        victim = self.rng.choice(sorted(v for v, c in counts.items() if c == top))
        self.alive.remove(victim)

        if self.roles[victim] == "Mafia":
            self.transcript.append(f"Player_{victim} was eliminated and was the Mafia. Town wins!")
            self.done, self.winner = True, "Town"
            return self._team_rewards(R_LYNCH_MAFIA), True

        self.transcript.append(f"Player_{victim} was eliminated and was not the Mafia.")
        if len(self.alive) <= 2:
            self.transcript.append("The Mafia has reached parity. Mafia wins!")
            self.done, self.winner = True, "Mafia"
            return self._team_rewards(R_PARITY), True

        self.day += 1
        self.phase = "night"
        self.transcript.append(f"Night {self.day} begins.")
        return self._team_rewards(R_LYNCH_TOWN), False

    def _team_rewards(self, pair: Tuple[float, float]) -> Dict[int, float]:
        town, mafia = pair
        return {p: (mafia if r == "Mafia" else town) for p, r in self.roles.items()}
