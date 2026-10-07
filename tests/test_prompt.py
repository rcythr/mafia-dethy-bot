"""Prompt test: checks the prompt invariants and emits example prompts for reading.

    python tests/test_prompt.py                       # print examples (generic chat rendering)
    python tests/test_prompt.py --out prompts.txt     # also write them to a file
    python tests/test_prompt.py --model meta-llama/Llama-3.2-3B-Instruct   # render with the real
                                                      # tokenizer's chat template + token counts
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dethy_rl.env import DethyEnv  # noqa: E402

CHAT_DATE = "26 Jul 2024"  # keep in sync with VllmWorker.CHAT_DATE


def render(messages, tokenizer=None):
    if tokenizer is not None:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                             date_string=CHAT_DATE)
    out = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages)
    return out + "<|assistant|>\n"


def play_to_day_two_vote(seed=7):
    """Advance one game to Night 2 so the prompts contain a rich transcript and private notes."""
    rng = random.Random(seed)
    env = DethyEnv(seed=seed)
    lines = ["I am a Cop and I have nothing to hide.", "Player_3 sounds suspicious to me.",
             "My investigation made me doubt Player_1.", "Let's vote carefully."]
    for _ in range(1):  # Night 1 -> Day 1 -> Night 2
        env.step_night({p: rng.choice(env.allowed_targets(p)) for p in env.alive})
        while env.phase == "dialogue":
            env.step_dialogue(env.acting_players()[0], rng.choice(lines))
        env.step_vote({p: rng.choice(env.alive) for p in env.alive})
    return env


def check_invariants(env):
    marker = "You are Player_"
    prompts = {p: env.build_prompt(p) for p in env.alive}
    prefixes = {pr.rsplit(marker, 1)[0] for pr in prompts.values()}
    assert len(prefixes) == 1, "everything before the private line must be identical for all players"
    for p, pr in prompts.items():
        head, tail = pr.rsplit(marker, 1)
        assert tail.startswith(f"{p}.\n\nPrivate Role: {env.roles[p]}.")      # transcript -> role -> phase
        assert tail.rstrip().endswith("Action:")
        assert "\nPhase:" in tail and tail.index("Private Role") < tail.index("\nPhase:")
        for q in env.alive:
            if q != p:
                assert f"You are Player_{q}." not in pr  # no leakage of other players' private lines
        for note in env.private_notes[p]:
            assert note in tail and note not in head
    msgs = env.build_messages(env.alive[0])
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "Dethy Mafia" in msgs[0]["content"] and "Dethy Mafia" not in msgs[1]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--model")
    args = ap.parse_args()
    tok = None
    if args.model:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)

    env = play_to_day_two_vote()
    check_invariants(env)
    cop = next(p for p in env.alive if env.roles[p] != "Mafia")
    raw_examples = [
        (f"NIGHT 2 - Cop (Player_{cop}, {env.roles[cop]})", env, cop, False),
        (f"NIGHT 2 - Mafia (Player_{env.mafia_id})", env, env.mafia_id, False),
        (f"NIGHT 2 - Cop, thinking stage", env, cop, True),
    ]
    # a dialogue and a vote prompt from a fresh day
    env2 = DethyEnv(seed=11)
    env2.step_night({p: random.choice(env2.allowed_targets(p)) for p in env2.alive})
    speaker = env2.acting_players()[0]
    raw_examples.append((f"DIALOGUE - Player_{speaker}", env2, speaker, False))
    while env2.phase == "dialogue":
        env2.step_dialogue(env2.acting_players()[0], "Let's hear from everyone.")
    check_invariants(env2)
    voter = env2.alive[0]
    raw_examples.append((f"VOTE - Player_{voter}", env2, voter, False))

    chunks = []
    for title, e, pid, think in raw_examples:
        text = render(e.build_messages(pid, think), tok)
        n = f"  [{len(tok.encode(text, add_special_tokens=False))} tokens]" if tok else ""
        chunks.append(f"{'=' * 20} {title}{n} {'=' * 20}\n{text}")
    output = "\n\n".join(chunks)
    print(output)
    if args.out:
        Path(args.out).write_text(output)
    print("\nprompt invariants ok")


if __name__ == "__main__":
    main()
