"""Prompt test: checks the prompt invariants and emits example prompts for reading.

    python tests/test_prompt.py                       # print examples (generic chat rendering)
    python tests/test_prompt.py --out prompts.txt     # also write them to a file
    python tests/test_prompt.py --model meta-llama/Llama-3.2-3B-Instruct   # render with the real
                                                      # tokenizer's chat template + token counts
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dethy_rl.env import DethyEnv, clean_message  # noqa: E402

CHAT_DATE = "26 Jul 2024"  # keep in sync with VllmWorker.CHAT_DATE


def render(messages, tokenizer=None, chat_kwargs=None):
    if tokenizer is not None:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                             date_string=CHAT_DATE, **(chat_kwargs or {}))
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
        assert tail.startswith(f"{p}.\n\nPrivate Role: {env.private_role(p)}.")      # transcript -> role -> phase
        assert tail.rstrip().endswith("Action:")
        assert "\nPhase:" in tail and tail.index("Private Role") < tail.index("\nPhase:")
        for q in env.alive:
            if q != p:
                assert f"You are Player_{q}." not in pr  # no leakage of other players' private lines
        if env.roles[p] != "Mafia":  # a Cop is never told its own sanity type
            assert env.roles[p] not in tail, (p, env.roles[p])
            assert "Private Role: Cop." in tail
        else:
            assert "Private Role: Mafia." in tail
        for note in env.private_notes[p]:
            assert note in tail and note not in head
    msgs = env.build_messages(env.alive[0])
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "Dethy Mafia" in msgs[0]["content"] and "Dethy Mafia" not in msgs[1]["content"]


def check_template_kwargs():
    """model.chat_template_kwargs reaches apply_chat_template for both prompts and the think suffix."""
    from types import SimpleNamespace as NS
    from dethy_rl.vllm_worker import VllmWorker

    seen = []

    class Tok:
        def apply_chat_template(self, messages, **kw):
            seen.append(kw)
            return "<s>" + "".join(m["content"] for m in messages) + "@@THOUGHT@@END" if len(messages) > 2 else "<s>x"

        def encode(self, text, add_special_tokens=True):
            return [1, 2]

    class Cfg(dict):
        def get(self, k, d=None):
            return dict.get(self, k, d)

    w = VllmWorker.__new__(VllmWorker)
    w.tokenizer = Tok()
    w.cfg = NS(model=NS(use_chat_template=True, get=Cfg(chat_template_kwargs={"enable_thinking": False}).get))
    w.prompt_ids(DethyEnv(seed=1), 0)
    w.think_suffix_ids()
    assert len(seen) == 2
    assert all(kw["enable_thinking"] is False and "date_string" in kw and kw["add_generation_prompt"] for kw in seen)


def check_rules_styles():
    from dethy_rl.env import build_rules

    full, compact = build_rules(True, "plurality", "full"), build_rules(True, "plurality", "compact")
    assert len(compact) < 0.5 * len(full)
    for text in (full, compact):
        for must in ("Sane", "Insane", "Naive", "Paranoid", "Paranoid", "9", "tie", "single digit"):
            assert must in text, must
    e = DethyEnv(seed=3, rules_style="compact", lynch_rule="plurality")
    assert e.build_messages(0)[0]["content"] == compact.strip()
    check_invariants(e)


def check_clean_message():
    assert clean_message("Player_2: Player_2: This was a coordinated attack.") == "This was a coordinated attack."
    assert clean_message('"I think Player 1 is lying."') == "I think Player 1 is lying."
    assert clean_message('Player_4: "Hello there"') == "Hello there"
    assert clean_message('"cut off without a close') == "cut off without a close"
    assert clean_message("I trust Player_2: they were clear.") == "I trust Player_2: they were clear."
    assert clean_message("  spaced \n out  ") == "spaced out"
    assert clean_message("Player_1:") == ""


def main():
    check_clean_message()
    check_template_kwargs()
    check_rules_styles()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--model")
    ap.add_argument("--chat-kwargs", default="{}", help='JSON, e.g. \'{"enable_thinking": false}\'')
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
        text = render(e.build_messages(pid, think), tok, json.loads(args.chat_kwargs))
        n = f"  [{len(tok.encode(text, add_special_tokens=False))} tokens]" if tok else ""
        chunks.append(f"{'=' * 20} {title}{n} {'=' * 20}\n{text}")
    output = "\n\n".join(chunks)
    print(output)
    if args.out:
        Path(args.out).write_text(output)
    print("\nprompt invariants ok")


if __name__ == "__main__":
    main()
