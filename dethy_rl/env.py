"""Dethy Mafia environment: a pure-python state machine (no model/GPU dependencies)."""
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

ROLES = ["Sane", "Insane", "Naive", "Paranoid", "Mafia"]
NO_LYNCH = 9  # a vote "for" this digit means "eliminate no one"; never a real player id (0-4)

_RULES_TEMPLATE = """You are playing Dethy Mafia, a social deduction game. There are 5 players: Player_0 to Player_4.

ROLES
- One player is the Mafia. The other four are Cops. The Mafia wins by surviving; the Cops (the Town) win by eliminating the Mafia.
- Each Cop has a hidden sanity type, and the four Cops have four different types. The type decides what their investigations tell them:
  * Sane: sees the truth ("Mafia" if the target is the Mafia, otherwise "Not Mafia").
  * Insane: sees the opposite of the truth (the Mafia looks like "Not Mafia", everyone else looks like "Mafia").
  * Naive: always sees "Not Mafia", whoever the target is, so a Naive result tells you nothing.
  * Paranoid: always sees "Mafia", whoever the target is, so a Paranoid result tells you nothing.
- Only the Mafia knows exactly who they are. The Mafia knows they are the Mafia. A Cop knows they are a Cop but does NOT know which sanity type they are, so a Cop can never be sure whether their own results are true, reversed or meaningless. Nobody's role or type is revealed unless a player chooses to say it, and players may lie.

HOW THE GAME RUNS
1. Night: every Cop investigates one player and privately learns "Mafia" or "Not Mafia" (according to their sanity). At the same time the Mafia kills one other player, who is out of the game.
2. Day discussion: the living players talk in 1 to 3 rounds. In each round everyone speaks once, in a random order, and can read everything said so far.
3. Day vote: every living player votes for one living player to eliminate{NOLYNCH_RULE}. {ELIM_RULE} Whoever is eliminated is announced as having been or not been the Mafia.
Then the next night begins.

HOW THE GAME ENDS
- The Cops win as soon as the Mafia is eliminated.
- The Mafia wins when only 2 players are left alive.

WHAT YOU SEE
- Everything above the line starting "You are Player_..." is public and visible to every player.
- That line shows your Private Role ("Mafia" if you are the Mafia, otherwise "Cop") and your Private Notes (your own investigation results). Only you can see them.

HOW TO PLAY
- Cops: your results are clues, not facts, because you do not know your own type. Compare notes: the four Cops all have different types, so what different Cops report about the same player can contradict each other in revealing ways. If you get the same answer for every player you investigate, you may be a Naive Cop (always "Not Mafia") or a Paranoid Cop (always "Mafia"). Share useful information, question inconsistent claims, and vote for who you believe is the Mafia. You cannot vote for, investigate or kill yourself.
- Mafia: stay alive. Blend in, cast suspicion on others, and kill players who might expose you.

EXAMPLE (placeholder names, only an illustration of the style: do not repeat its wording, and base what you say on your own game)
Cop X privately saw "Mafia" when investigating Z. During discussion X says that Z showed up as Mafia, adds that they cannot be sure their results are reliable, and asks for a vote against Z. Cop Y says they investigated Z too and saw "Not Mafia", so one of the two results must be unreliable. Z denies being the Mafia. Cop W notes that X also reported "Mafia" for another player last night, which makes X's results look less trustworthy, and suggests waiting for more information. At the vote, a player's whole answer is just the digit of the player they pick.

HOW TO ANSWER
- Night and Vote phases: answer with a single digit, the ID of a living player other than yourself (for example: 3). Nothing else.{NOLYNCH_ANSWER}
- Discussion phase: write one or two short sentences (about 40 words at most) as yourself. Do not write your own name or "Player_N:" at the start.
- Discussion is only talking. Nobody investigates or kills during the day: investigations and kills happen at night, and the day ends with a vote. Use the discussion to say who you suspect and why, to defend yourself, or to share (truthfully or not) what your investigations showed. Do not ask others to investigate.
"""


_COMPACT_TEMPLATE = """Dethy Mafia, a social deduction game with 5 players (Player_0 to Player_4): 1 Mafia and 4 Cops. The Cops win by eliminating the Mafia; the Mafia wins when only 2 players are alive.
The four Cops have four different hidden types, and no Cop knows their own: Sane (investigations are true), Insane (reversed), Naive (always "Not Mafia"), Paranoid (always "Mafia"). Only the Mafia knows exactly who they are. Your Private Role says "Mafia" or "Cop". Anyone may lie about their role or results. Everything before the line starting "You are Player_" is public.
Each night every Cop investigates one player and privately learns "Mafia" or "Not Mafia" (according to their type), and the Mafia kills one other player. Each day there are 1 to 3 discussion rounds (everyone speaks once per round, random order), then everyone votes for another living player{NOLYNCH_RULE}. {ELIM_RULE}
Cops: your results are clues, not facts. Compare notes: the Cops' types differ, so their reports can contradict each other. Vote for who you think is the Mafia. Mafia: blend in, deflect suspicion, kill whoever might expose you. Nobody can investigate, kill or vote for themselves.
Answers: Night and Vote: only a single digit, the ID of a living player other than yourself.{NOLYNCH_ANSWER} Discussion: one or two short sentences (about 40 words), without a "Player_N:" prefix; nobody investigates or kills during the day.
"""


def build_rules(allow_no_lynch: bool = True, lynch_rule: str = "plurality", style: str = "full") -> str:
    """The rules text, with or without the no-lynch option. style: "full" or "compact" (about half the tokens)."""
    assert style in ("full", "compact"), style
    rule = (f" (or for {NO_LYNCH}, meaning no one: if that gets the most votes, nobody is eliminated "
            "and the next night begins)") if allow_no_lynch else ""
    answer = (f" In the Vote phase you may instead answer {NO_LYNCH} to vote for no one. Voting for no one is safe "
              "if you are unsure, but the Mafia keeps killing every night, so skipping too long lets it win."
              ) if allow_no_lynch else ""
    if style == "compact" and allow_no_lynch:
        answer = f" In the Vote phase you may answer {NO_LYNCH} to vote for no one (safe, but the Mafia keeps killing)."
    if lynch_rule == "majority":
        elim = ("A player is eliminated only if MORE THAN HALF of the living players vote for them. "
                "If nobody gets that many votes (the votes are split, or most players vote for no one), "
                "nobody is eliminated and the next night begins.")
    else:
        elim = ("The player with the most votes is eliminated. If two or more players tie for the most votes"
                + (", or \"no one\" gets the most," if allow_no_lynch else ",")
                + " nobody is eliminated and the next night begins.")
    template = _COMPACT_TEMPLATE if style == "compact" else _RULES_TEMPLATE
    return (template.replace("{NOLYNCH_RULE}", rule).replace("{NOLYNCH_ANSWER}", answer)
            .replace("{ELIM_RULE}", elim))


RULES = build_rules(True)


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
                 min_rounds: int = 1, max_rounds: int = 3, allow_no_lynch: bool = True,
                 lynch_rule: str = "plurality", rules_style: str = "full"):
        self.rng = random.Random(seed)
        self.rw = RewardConfig(**(rewards or {}))
        self.min_rounds, self.max_rounds = min_rounds, max_rounds
        self.allow_no_lynch = allow_no_lynch
        assert lynch_rule in ("majority", "plurality"), lynch_rule
        self.lynch_rule = lynch_rule
        self.rules_style = rules_style
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
        # omniscient spectator log (never shown to agents): everything incl. true roles, results, votes
        self._md: List[str] = []
        self.fate: Dict[int, str] = {}
        self.end_note = ""

    def _role_name(self, pid: int) -> str:
        role = self.roles[pid]
        return "Mafia" if role == "Mafia" else f"{role} Cop"

    def _who(self, pid: int) -> str:
        return f"Player_{pid} ({self._role_name(pid)})"

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

    def private_role(self, player_id: int) -> str:
        """What the player is told about themselves: the Mafia knows it is the Mafia; a Cop only
        knows it is a Cop (its sanity type stays hidden, even from itself)."""
        return "Mafia" if self.roles[player_id] == "Mafia" else "Cop"

    def allowed_targets(self, player_id: int) -> List[int]:
        targets = [p for p in self.alive if p != player_id]  # never yourself, for kills, checks or votes
        if self.phase == "vote" and self.allow_no_lynch:
            targets.append(NO_LYNCH)
        return targets

    # ---------------------------------------------------------------- prompts
    def public_transcript(self) -> str:
        return "\n".join(self.transcript).strip("\n")

    def build_messages(self, player_id: int, think: bool = False) -> List[Dict[str, str]]:
        """Chat form: the rules are the system message, the rest is one user message."""
        return [{"role": "system", "content": build_rules(self.allow_no_lynch, self.lynch_rule, self.rules_style).strip()},
                {"role": "user", "content": self.user_text(player_id, think)}]

    def build_prompt(self, player_id: int, think: bool = False) -> str:
        """Raw-text form (no chat template): rules followed by the user text."""
        return f"{build_rules(self.allow_no_lynch, self.lynch_rule, self.rules_style)}\n{self.user_text(player_id, think)}"

    def user_text(self, player_id: int, think: bool = False) -> str:
        """[Shared Public Transcript] + \\nPrivate Role: [Role]. Phase: [Phase]. Action:

        The transcript is byte-identical across agents so vLLM can reuse the prefix KV cache.
        """
        # public and identical for every player, so the prefix cache still covers it
        head = (f"{self.public_transcript()}\n\nAlive players: "
                + ", ".join(f"Player_{p}" for p in self.alive) + ".")
        role = f"\nYou are Player_{player_id}.\n\nPrivate Role: {self.private_role(player_id)}."
        if self.private_notes[player_id]:
            role += "\n\nPrivate Notes:\n\n" + "\n".join(f"- {n}" for n in self.private_notes[player_id])
        if self.phase == "night":
            verb = "kill" if self.roles[player_id] == "Mafia" else "investigate"
            phase = f"Night {self.day} (answer with the ID of the player to {verb})"
        elif self.phase == "dialogue":
            phase = f"Day {self.day} Dialogue round {self.round}/{self.num_rounds} (say one short public message)"
        else:
            skip = f", or {NO_LYNCH} to vote for no one" if self.allow_no_lynch else ""
            phase = f"Day {self.day} Vote (answer with the ID of the player to eliminate{skip})"
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
        self._md.append(f"\n## Day {self.day}: discussion ({self.num_rounds} round"
                        f"{'s' if self.num_rounds > 1 else ''})")
        self.round = 0
        self.phase = "dialogue"
        self._next_round()

    def _next_round(self) -> None:
        self.round += 1
        self.transcript.append(("\n" if self.round > 1 else "") + f"-- Round {self.round} of {self.num_rounds} --")
        self._md.append(f"\n**Round {self.round} of {self.num_rounds}**")
        self.speakers = self.alive[:]
        self.rng.shuffle(self.speakers)  # fresh random speaking order every round

    # ------------------------------------------------------------------ steps
    def step_night(self, targets: Dict[int, int]) -> Tuple[Dict[int, float], bool]:
        """Cops investigate and the Mafia kills, simultaneously. Returns (rewards, game_over)."""
        assert self.phase == "night"
        r: Dict[int, float] = defaultdict(float)
        mafia = self.mafia_id
        self._md.append(f"\n## Night {self.day}\n")
        for pid in self.alive:
            if pid == mafia:
                continue
            t = targets.get(pid)
            if t not in self.alive or t == pid:
                t = self.rng.choice(self.allowed_targets(pid))
            seen = "Mafia" if self._sanity_result(self.roles[pid], t) else "Not Mafia"
            self.private_notes[pid].append(f"Night {self.day}: you investigated Player_{t}: {seen}.")
            truthful = (seen == "Mafia") == (t == mafia)
            self._md.append(f"- {self._who(pid)} investigated {self._who(t)} and was told **{seen}**"
                            + ("" if truthful else " (misleading: that player " + ("is not" if seen == "Mafia" else "is")
                               + " the Mafia)"))
            if t == mafia:
                r[pid] += self.rw.investigate_mafia_bonus

        victim = targets.get(mafia)
        if victim not in self.alive or victim == mafia:
            victim = self.rng.choice([p for p in self.alive if p != mafia])
        self._md.append(f"- {self._who(mafia)} killed {self._who(victim)}")
        self.fate[victim] = f"killed on night {self.day}"
        self.alive.remove(victim)
        self.transcript.append(f"Player_{victim} was killed in the night by the Mafia and is out of the game.")
        if len(self.alive) <= 2:
            self.transcript.append("\nOnly 2 players are left. The Mafia wins!")
            self.end_note = (f"On night {self.day} the Mafia's kill left only 2 players alive, so the Mafia "
                             "reached parity.")
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
        self._md.append(f"\n- **{self._who(player_id)}**: {msg}")
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
        legal = set(self.alive) | ({NO_LYNCH} if self.allow_no_lynch else set())
        valid = {p: v for p, v in votes.items() if p in self.alive and v in legal and v != p}
        for p in self.alive:  # invalid votes get a random valid vote
            valid.setdefault(p, self.rng.choice(self.allowed_targets(p)))
        self.transcript.append(
            "Votes:\n" + "\n".join(
                f"\nPlayer_{p} voted for " + ("no one" if v == NO_LYNCH else f"Player_{v}")
                for p, v in sorted(valid.items()))
        )
        counts = Counter(valid.values())
        mafia = self.mafia_id
        r: Dict[int, float] = defaultdict(float)
        self._md.append(f"\n### Day {self.day}: vote\n")
        for p, v in sorted(valid.items()):
            target = "no one" if v == NO_LYNCH else self._who(v)
            hit = " ✔ (voted for the Mafia)" if (p != mafia and v == mafia) else ""
            self._md.append(f"- {self._who(p)} → {target}{hit}")
        tally = ", ".join(("no one" if v == NO_LYNCH else f"Player_{v}") + f": {c}"
                          for v, c in sorted(counts.items(), key=lambda kv: -kv[1]))
        self._md.append(f"\nTally: {tally}")

        # dense shaping: how much heat the Mafia took, and which Cops voted well
        heat = self.rw.heat * counts.get(mafia, 0) / len(valid)
        self._team(r, heat, -heat)
        for p, v in valid.items():
            if p != mafia:
                if v == NO_LYNCH:
                    continue  # abstaining: no bonus, no penalty
                r[p] += self.rw.vote_mafia_bonus if v == mafia else -self.rw.vote_town_penalty

        top = max(counts.values())
        leaders = [v for v, c in counts.items() if c == top]
        victim = leaders[0] if len(leaders) == 1 else NO_LYNCH   # a tie for first elects no one
        if self.lynch_rule == "majority" and victim != NO_LYNCH and top < len(self.alive) // 2 + 1:
            victim = NO_LYNCH                                    # strict mode: need > half of the living
        if victim == NO_LYNCH:
            if len(leaders) > 1:
                why = "two or more tied for the most votes"
            elif leaders[0] == NO_LYNCH:
                why = "\"no one\" got the most votes"
            else:
                why = "no player had more than half of the votes"
            self._md.append(f"\n**Result:** nobody was eliminated ({why}).")
        else:
            self._md.append(f"\n**Result:** {self._who(victim)} was eliminated.")
        if victim == NO_LYNCH:  # nobody is eliminated; the game goes straight to the next night
            self.transcript.append("\nNo one was eliminated (no player got enough votes).")
            self.day += 1
            self.phase = "night"
            self.transcript.append(f"\n=== Night {self.day} ===\n")
            return dict(r), False
        self.fate[victim] = f"eliminated on day {self.day}"
        self.alive.remove(victim)

        if self.roles[victim] == "Mafia":
            self.transcript.append(f"\nPlayer_{victim} was eliminated and was the Mafia. The Cops win!")
            self.end_note = (f"On day {self.day} the Cops eliminated the Mafia ({self._who(victim)}) with "
                             f"{counts[victim]} of {len(valid)} votes.")
            self.done, self.winner = True, "Town"
            self._team(r, self.rw.lynch_mafia, -self.rw.lynch_mafia)
            return dict(r), True

        self.transcript.append(f"\nPlayer_{victim} was eliminated and was not the Mafia.")
        if len(self.alive) <= 2:
            self.transcript.append("\nOnly 2 players are left. The Mafia wins!")
            self.end_note = (f"On day {self.day} the table eliminated an innocent Cop ({self._who(victim)}), "
                             "leaving only 2 players alive, so the Mafia reached parity.")
            self.done, self.winner = True, "Mafia"
            self._team(r, -self.rw.parity, self.rw.parity)
            return dict(r), True

        self._team(r, -self.rw.lynch_town, self.rw.lynch_town)
        self.day += 1
        self.phase = "night"
        self.transcript.append(f"\n=== Night {self.day} ===\n")
        return dict(r), False

    # ------------------------------------------------------------ spectator log
    def render_game_log(self, title: str, returns: Optional[Dict[int, float]] = None) -> str:
        """Markdown for human inspection: true roles, private results, dialogue, votes, and an outro."""
        mafia = self.mafia_id
        out = [f"# {title}", "", "## Cast", "", "| Player | True role | Told to the agent | Fate |", "|---|---|---|---|"]
        for pid in sorted(self.roles):
            role = self.roles[pid]
            fate = self.fate.get(pid, "survived" if pid in self.alive else "-")
            out.append(f"| Player_{pid} | {self._role_name(pid)} | {self.private_role(pid)} | {fate} |")
        out += self._md
        out += ["", "---", "", "## Outro", ""]
        if not self.done:
            out.append("The game did not finish.")
        else:
            out.append(f"**{'The Cops (Town)' if self.winner == 'Town' else 'The Mafia'} won.** {self.end_note}")
            out.append("")
            out.append("- Players alive at the end: " + (", ".join(f"Player_{p}" for p in self.alive) or "none"))
            out.append(f"- The Mafia was {self._who(mafia)}.")
            ended = "during the night" if self.end_note.startswith("On night") else "during the day"
            out.append(f"- Game length: {self.day} night{'s' if self.day > 1 else ''}, ended {ended}.")
        if returns:
            out += ["", "| Player | Role | Episode reward |", "|---|---|---|"]
            for pid in sorted(returns):
                out.append(f"| Player_{pid} | {self._role_name(pid)} | {returns[pid]:+.2f} |")
        return "\n".join(out) + "\n"
