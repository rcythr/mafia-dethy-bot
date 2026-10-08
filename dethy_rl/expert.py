"""Exact inference and scripted players for Dethy Mafia (pure python, no model needed).

The hidden state is which of the five roles each player has: 5! = 120 assignments. A Cop's result is a
deterministic function of its (hidden) type and whether the target is the Mafia, so the posterior over who is
the Mafia can be computed exactly by enumeration. Scripted Cops report honestly and vote for the posterior's
favourite; the scripted Mafia bluffs and frames. They are used to evaluate the LLM seat by seat.
"""
import itertools
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

TYPES = ["Sane", "Insane", "Naive", "Paranoid"]
ASSIGNMENTS = tuple(itertools.permutations(TYPES + ["Mafia"]))   # a[i] = role of player i


def told_mafia(sanity: str, target_is_mafia: bool) -> bool:
    """What a Cop of this type is told about a target (True = "Mafia")."""
    return {"Sane": target_is_mafia, "Insane": not target_is_mafia, "Naive": False, "Paranoid": True}[sanity]


@dataclass(frozen=True)
class Claim:
    speaker: int
    target: int
    told_mafia: bool


LIE_WEIGHT = 0.5   # relative likelihood that the Mafia emits one particular fabricated claim vs a Cop's true one


def _consistent_counts(claims: Sequence[Claim], dead: Sequence[int], me: int, me_is_mafia: bool,
                       lie_weight: float = LIE_WEIGHT) -> Counter:
    """Weighted count of role assignments compatible with the claims, keyed by who the Mafia is.

    A Cop's claim must be true. The Mafia's claim is unconstrained but each one is less probable than a Cop's
    true report (a fabricated claim has more ways to be wrong), hence lie_weight per claim. Without that
    penalty a lone claimant would look like the Mafia just because its claim excludes nothing."""
    out: Counter = Counter()
    for a in ASSIGNMENTS:
        if (a[me] == "Mafia") != me_is_mafia:
            continue
        if any(a[d] == "Mafia" for d in dead):          # the dead were announced as not the Mafia
            continue
        weight, ok = 1.0, True
        for c in claims:
            if a[c.speaker] == "Mafia":
                weight *= lie_weight
            elif told_mafia(a[c.speaker], a[c.target] == "Mafia") != c.told_mafia:
                ok = False
                break
        if ok:
            out[a.index("Mafia")] += weight
    return out


def mafia_posterior(claims: Sequence[Claim], dead: Sequence[int], me: int, me_is_mafia: bool = False) -> Counter:
    """Unnormalised posterior over who the Mafia is, assuming only the Mafia lies.

    If the claims contradict that assumption (another player lied or was misparsed), fall back to ignoring
    one other speaker at a time, and finally to the player's own claims only."""
    counts = _consistent_counts(claims, dead, me, me_is_mafia)
    if sum(counts.values()):
        return counts
    total: Counter = Counter()
    for s in sorted({c.speaker for c in claims} - {me}):
        total.update(_consistent_counts([c for c in claims if c.speaker != s], dead, me, me_is_mafia))
    if sum(total.values()):
        return total
    return _consistent_counts([c for c in claims if c.speaker == me], dead, me, me_is_mafia)


# --------------------------------------------------------------------------- reading claims from text
_CUE = re.compile(r"investigat|checked|\bcheck\b|result|told|came back|came up|learned|\bsaw\b|\bgot\b|found out|"
                  r"showed|revealed|read as|reading", re.I)
_NOT_MAFIA = re.compile(r"not (?:the |a )?mafia|isn'?t (?:the |a )?mafia|innocent|\bclean\b|not guilty|"
                        r"not a threat|came back (?:as )?safe", re.I)
_MAFIA = re.compile(r"\bmafia\b|\bguilty\b", re.I)
_PLAYER = re.compile(r"player[_ ]?(\d)", re.I)


def parse_claims(speaker: int, text: str, num_players: int = 5) -> List[Claim]:
    """Best-effort extraction of investigation claims ("I investigated Player_3 and was told Mafia").

    A sentence counts when it has an investigation cue, exactly one other player, and a Mafia / Not-Mafia
    verdict. Accusations without a cue ("I think Player_3 is the Mafia") are not claims."""
    claims = []
    for sent in re.split(r"[.!?;\n]+", text):
        if not sent.strip() or not _CUE.search(sent):
            continue
        others = {int(m) for m in _PLAYER.findall(sent) if int(m) < num_players and int(m) != speaker}
        if len(others) != 1:
            continue
        if _NOT_MAFIA.search(sent):
            verdict = False
        elif _MAFIA.search(sent):
            verdict = True
        else:
            continue
        claims.append(Claim(speaker, others.pop(), verdict))
    return claims


def public_claims(env, exclude: Optional[int] = None) -> List[Claim]:
    """Every investigation claim made in the public discussion so far."""
    out = []
    for _day, speaker, text in env.messages:
        if speaker != exclude:
            out.extend(parse_claims(speaker, text))
    return out


def own_claims(env, pid: int) -> List[Claim]:
    return [Claim(pid, t, r) for (_n, i, t, r) in env.investigations if i == pid]


def posterior_leaders(env, pid: int, use_claims: bool) -> Tuple[List[int], Counter]:
    """All living players (not pid) tied for most likely Mafia under the exact posterior, and the posterior."""
    claims = own_claims(env, pid) + (public_claims(env, exclude=pid) if use_claims else [])
    dead = [p for p in env.roles if p not in env.alive]
    post = mafia_posterior(claims, dead, pid)
    cands = [p for p in env.alive if p != pid]
    best = max(post.get(p, 0) for p in cands)
    return [p for p in cands if post.get(p, 0) == best], post


def posterior_pick(env, pid: int, use_claims: bool, rng: random.Random) -> Tuple[int, Counter]:
    """The living player (not pid) an exact reasoner would vote for (ties broken at random), and the posterior."""
    leaders, post = posterior_leaders(env, pid, use_claims)
    return rng.choice(leaders), post


# --------------------------------------------------------------------------- scripted players
class ScriptedTeam:
    """Scripted behaviour for any seat. Cops report honestly and vote with the exact posterior; the Mafia
    bluffs (fake investigation reports, framing a random Cop) and joins the frame at the vote.

    share_prob: chance that a given scripted Cop shares its notes at all (the rest stay silent all game);
    use_claims: whether Cops use what others said (False = reason from their own notes only)."""

    def __init__(self, seed: int = 0, share_prob: float = 1.0, use_claims: bool = True):
        self.rng = random.Random(seed)
        self.share_prob, self.use_claims = share_prob, use_claims
        self._shares: Dict[int, bool] = {}
        self._fake: Dict[int, Tuple[int, bool]] = {}      # night -> (target, claimed_mafia) for the Mafia's bluff
        self.pick_log: List[Tuple[int, int]] = []         # (pid, voted_for) for debugging

    # --- actions
    def night(self, env, pid: int) -> int:
        return self.rng.choice(env.allowed_targets(pid))

    def speak(self, env, pid: int) -> str:
        if env.roles[pid] == "Mafia":
            return self._mafia_speech(env, pid)
        if pid not in self._shares:
            self._shares[pid] = self.rng.random() < self.share_prob
        if not self._shares[pid]:
            return "I have nothing to report yet."
        notes = [f"On night {n} I investigated Player_{t} and was told {'Mafia' if r else 'Not Mafia'}."
                 for (n, i, t, r) in env.investigations if i == pid]
        text = " ".join(notes)
        if self.use_claims:
            pick, _ = posterior_pick(env, pid, True, random.Random(0))
            text += f" Going by all the reports so far, I think Player_{pick} is the Mafia."
        return text

    def vote(self, env, pid: int) -> int:
        if env.roles[pid] == "Mafia":
            framed = [t for (t, said_mafia) in self._fake.values() if said_mafia and t in env.alive and t != pid]
            return framed[-1] if framed else self.rng.choice(env.allowed_targets(pid))
        pick, _ = posterior_pick(env, pid, self.use_claims, self.rng)
        self.pick_log.append((pid, pick))
        return pick

    # --- the Mafia's bluff
    def _mafia_speech(self, env, pid: int) -> str:
        night = env.day            # the night before this day
        if night not in self._fake:
            cops = [p for p in env.alive if p != pid]
            target = self.rng.choice(cops)
            self._fake[night] = (target, self.rng.random() < 0.5)    # half the time: frame a Cop as "Mafia"
        lines = [f"On night {n} I investigated Player_{t} and was told {'Mafia' if m else 'Not Mafia'}."
                 for n, (t, m) in sorted(self._fake.items())]
        return " ".join(lines)


def simulate(seed: int, **team_kwargs):
    """Play one whole game with every seat scripted (no model). Returns the finished env."""
    from dethy_rl.env import DethyEnv

    env, team = DethyEnv(seed=seed), ScriptedTeam(seed=seed, **team_kwargs)
    while not env.done:
        acting = env.acting_players()
        if env.phase == "night":
            env.step_night({p: team.night(env, p) for p in acting})
        elif env.phase == "dialogue":
            env.step_dialogue(acting[0], team.speak(env, acting[0]))
        else:
            env.step_vote({p: team.vote(env, p) for p in acting})
    return env
