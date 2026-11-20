"""Clean data/raw/*.txt into the committed corpus: data/train.txt, data/val.txt, data/glossary.json, results/corpus_stats.md.
Run: python scripts/build_corpus.py   (after scripts/download_data.py)"""
import json
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import core  # noqa: E402

DICTIONARY = "goulds_students_dictionary_1900"
HELD_OUT_PERCENT = 6   # dictionary entries held out for validation and the quiz, picked by a hash of the term


def main() -> None:
    raw = ROOT / "data" / "raw"
    train, val, stats, glossary = [], [], [], []
    for path in sorted(raw.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.stem == DICTIONARY:
            entries = core.parse_dictionary(text)
            held = [e for e in entries if zlib.crc32(e[0].encode()) % 100 < HELD_OUT_PERCENT]
            train += [core.format_entry(*e) for e in entries if e not in held]
            val += [core.format_entry(*e) for e in held]
            glossary = [list(e) for e in held if 40 <= len(e[1]) <= 300]
            stats.append((path.stem, f"{len(entries)} dictionary entries ({len(held)} held out)", sum(len(e[1]) for e in entries)))
        else:
            paras = [c for p in core.clean_text(text).split("\n\n") for c in core.chunk_paragraph(p)]
            tr, va = core.split_corpus(paras, 0.06, seed=0)
            train += tr
            val += va
            stats.append((path.stem, f"{len(paras)} paragraphs", sum(map(len, paras))))
    train, val = core.split_corpus(train, 0.0, seed=1)[0], val   # shuffle train so sources mix
    import random
    random.Random(2).shuffle(train)
    random.Random(3).shuffle(val)
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "train.txt").write_text("\n\n".join(train), encoding="utf-8")
    (ROOT / "data" / "val.txt").write_text("\n\n".join(val), encoding="utf-8")
    (ROOT / "data" / "glossary.json").write_text(json.dumps(glossary[:400], indent=0), encoding="utf-8")
    tr_stats, va_stats = core.corpus_stats("\n\n".join(train)), core.corpus_stats("\n\n".join(val))
    lines = ["# Corpus statistics", "", "| source | kept | characters |", "|---|---|---|"]
    lines += [f"| {n} | {k} | {c:,} |" for n, k, c in stats]
    lines += ["", "| split | characters | words | unique words | paragraphs |", "|---|---|---|---|---|"]
    lines += [f"| {n} | {s['characters']:,} | {s['words']:,} | {s['unique_words']:,} | {s['paragraphs']:,} |"
              for n, s in (("train", tr_stats), ("val", va_stats))]
    lines += ["", f"Quiz glossary (held-out dictionary entries): {min(len(glossary), 400)}"]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "corpus_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
