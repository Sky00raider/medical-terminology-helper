"""Command-line assistant: `python -m app.cli --help`.

  complete "The femur is"      finish a phrase (several options with -n, length with --words)
  define femur                 dictionary-style definition
  chat                         interactive: type a partial phrase, pick an option, keep going
  next "inflammation of the"   the most likely next tokens, with probabilities
  quiz                         flashcards from held-out dictionary entries, scored against the real definition
  roots gastroenteritis        break a term into Greek/Latin parts
  explain gastroenteritis      roots plus a short tutor explanation from the chat model set in .env (optional)
  neighbours cardi             tokens the model embeds close to a token
  saved                        completions you kept
"""
import argparse
import sys

from app import core
from app.storage import Store

HELP_CHAT = ("Type a partial phrase to complete it. Commands: /define TERM, /roots TERM, /n K, /words N, /temp T, "
             "/explain TERM, /save, /saved, /help, /quit. After options appear, type 1, 2, ... to keep one and continue from it.")


def _model(args):
    try:
        return core.load_model(args.model)
    except FileNotFoundError:
        sys.exit(f"No trained model in {args.model!r}. Run: python scripts/train.py   (see README)")


def _sampling(p: argparse.ArgumentParser) -> None:
    p.add_argument("-n", type=int, default=3, help="how many completion options (default 3)")
    p.add_argument("--words", type=int, default=25, help="desired completion length in words (default 25)")
    p.add_argument("--temperature", type=float, default=0.7, help="0 = greedy, higher = more varied")
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=None, help="make the sampling repeatable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medical-terminology-helper",
        description="Medical students get a command-line assistant completing medical phrases and definitions, "
                    "trained on public domain medical texts to reinforce terminology learning.")
    parser.add_argument("--model", default=core.DEFAULT_MODEL, help="folder with model.pt and tokenizer.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    c = sub.add_parser("complete", help="complete a partial phrase")
    c.add_argument("text")
    _sampling(c)
    c.add_argument("--pick", action="store_true", help="choose one option and save it")
    d = sub.add_parser("define", help="dictionary-style definition of a term")
    d.add_argument("term")
    _sampling(d)
    ch = sub.add_parser("chat", help="interactive session")
    _sampling(ch)
    nx = sub.add_parser("next", help="most likely next tokens")
    nx.add_argument("text")
    nx.add_argument("-k", type=int, default=8)
    q = sub.add_parser("quiz", help="flashcards from held-out dictionary entries")
    q.add_argument("-n", type=int, default=5)
    q.add_argument("--seed", type=int, default=None)
    q.add_argument("--glossary", default=core.GLOSSARY_PATH)
    q.add_argument("--no-model", action="store_true", help="skip the model's answer, just score yours")
    r = sub.add_parser("roots", help="split a term into Greek/Latin parts")
    r.add_argument("term", nargs="+")
    ex = sub.add_parser("explain", help="roots plus a tutor explanation from the chat model in .env")
    ex.add_argument("term")
    nb = sub.add_parser("neighbours", help="nearest tokens in embedding space")
    nb.add_argument("token", help="a single token such as 'cardi' or ' heart'")
    nb.add_argument("-k", type=int, default=8)
    sub.add_parser("saved", help="show saved completions")
    return parser


def show_options(prompt: str, options, out=print) -> None:
    for i, o in enumerate(options, 1):
        out(f"  {i}. {prompt}{o.text}   [{o.score:.2f}]")


def run_complete(model, tok, prompt, args, out=print):
    opts = core.complete(model, tok, prompt, max_words=args.words, n=args.n, temperature=args.temperature,
                         top_k=args.top_k, top_p=args.top_p, seed=args.seed)
    show_options(prompt, opts, out)
    return opts


def explain(term: str, out=print) -> None:
    parts = core.explain_term(term)
    out(f"{term}: " + " + ".join(p for p, _ in parts))
    for p, meaning in parts:
        out(f"  {p:<12} {meaning}")


def tutor(term: str, out=print, ask=None) -> None:
    """Word roots, then the optional chat-model explanation (needs LLM_MODEL in .env; see .env.example)."""
    explain(term, out)
    try:
        out(core.tutor_explain(term, ask))
    except Exception as e:  # no model configured, server down, bad key: say so, keep the roots
        out(f"(tutor unavailable: {e})")


def chat(model, tok, args, store, inp=input, out=print) -> None:
    opts, prompt = [], ""
    settings = {"n": args.n, "words": args.words, "temperature": args.temperature}
    history = core.History(store)
    out("Medical Terminology Helper. " + HELP_CHAT)

    def refresh(p: str) -> None:
        nonlocal opts, prompt
        prompt = p
        opts = core.complete(model, tok, p, settings["words"], settings["n"], settings["temperature"])
        show_options(p, opts, out)

    while True:
        try:
            line = inp("> ").strip()
        except EOFError:
            return
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        if cmd in ("/quit", "/exit", "/q"):
            return
        if cmd == "/help":
            out(HELP_CHAT)
        elif cmd in ("/n", "/words", "/temp"):
            key = {"/n": "n", "/words": "words", "/temp": "temperature"}[cmd]
            try:
                settings[key] = type(settings[key])(rest)
                out(f"{key} = {settings[key]}")
            except ValueError:
                out(f"usage: {cmd} <number>")
        elif cmd == "/roots" and rest:
            explain(rest, out)
        elif cmd == "/explain" and rest:
            tutor(rest, out)
        elif cmd == "/saved":
            for i, it in enumerate(history.items(), 1):
                out(f"{i}. {it['prompt']}{it['completion']}")
        elif cmd == "/save":
            if opts:
                out(f"saved ({history.add(prompt, opts[0].text)} total)")
            else:
                out("nothing to save yet")
        elif cmd == "/define" and rest:
            refresh(core.define_prompt(rest))
        elif line.isdigit() and opts and 1 <= int(line) <= len(opts):
            kept = prompt + opts[int(line) - 1].text  # keep that option and continue from it
            out(f"kept: {kept}")
            history.add(prompt, opts[int(line) - 1].text)
            refresh(kept)
        elif line.startswith("/"):
            out(f"unknown command {cmd!r}; try /help")
        else:
            refresh(line)


def quiz(args, model, tok, inp=input, out=print) -> float:
    entries = core.pick_quiz_entries(core.load_glossary(args.glossary), args.n, args.seed)
    total = 0.0
    for i, (term, ref) in enumerate(entries, 1):
        out(f"\n[{i}/{len(entries)}] Define: {term}")
        try:
            ans = inp("your definition> ")
        except EOFError:
            ans = ""
        score = core.overlap_score(ans, ref)
        total += score
        out(f"  reference: {ref}")
        if model:
            guess = core.complete(model, tok, core.define_prompt(term), max_words=25, n=1, seed=i)
            out(f"  model:     {guess[0].text if guess else '(no completion)'}")
        out(f"  you used {score:.0%} of the reference's key words")
    out(f"\nAverage key-word overlap: {total / max(1, len(entries)):.0%}")
    return total / max(1, len(entries))


def main(argv: list[str] | None = None, inp=input, out=print) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "hello":
        return out("Medical Terminology Helper is ready. Open README.md for your milestones.")
    if args.command == "roots":
        for t in args.term:
            explain(t, out)
        return
    if args.command == "explain":
        return tutor(args.term, out)
    if args.command == "saved":
        for i, it in enumerate(core.History(Store()).items(), 1):
            out(f"{i}. {it['prompt']}{it['completion']}")
        return
    if args.command == "quiz":
        model, tok = (None, None) if args.no_model else _model(args)
        quiz(args, model, tok, inp, out)
        return
    model, tok = _model(args)
    if args.command == "complete":
        text = args.text
        opts = run_complete(model, tok, text, args, out)
        if args.pick and opts:
            choice = inp(f"keep which option (1-{len(opts)}, Enter to skip)? ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(opts):
                core.History(Store()).add(text, opts[int(choice) - 1].text)
                out("saved")
    elif args.command == "define":
        run_complete(model, tok, core.define_prompt(args.term), args, out)
    elif args.command == "next":
        for t, p in core.next_token_candidates(model, tok, args.text, args.k):
            out(f"  {p:6.1%}  {t!r}")
    elif args.command == "neighbours":
        try:
            for t, s in core.nearest_tokens(tok, model.emb.tok.weight, args.token, args.k):
                out(f"  {s:5.2f}  {t!r}")
        except ValueError as e:
            sys.exit(str(e))
    elif args.command == "chat":
        chat(model, tok, args, Store(), inp, out)


if __name__ == "__main__":
    main()
