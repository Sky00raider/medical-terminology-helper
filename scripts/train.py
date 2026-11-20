"""Train the BPE tokenizer and the GPT on data/train.txt; writes models/model.pt, models/tokenizer.json and results/loss.csv.

  python scripts/train.py                         # small preset, about 12 minutes on a laptop CPU
  python scripts/train.py --preset large --device cuda   # bigger model for a GPU (e.g. a free Colab T4)

CPU runs are polite by default: 8 threads and below-normal process priority, so other work keeps running."""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import core  # noqa: E402

PRESETS = {  # vocab, block, n_embd, heads, layers, steps, batch, lr (large: about 17 passes over the corpus, so more dropout)
    "small": dict(vocab=1024, block=128, n_embd=128, n_heads=4, n_layers=4, steps=1800, batch=32, lr=3e-3),
    "medium": dict(vocab=2048, block=128, n_embd=192, n_heads=6, n_layers=6, steps=2500, batch=48, lr=2e-3),  # about 2 s/step on 8 CPU threads: use a GPU
    "large": dict(vocab=2048, block=256, n_embd=384, n_heads=6, n_layers=6, steps=2500, batch=64, lr=1.5e-3, dropout=0.2),
}


def lower_priority() -> None:
    """Below-normal priority on Windows, nice(10) elsewhere."""
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
        else:
            import os
            os.nice(10)
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="small")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--eval-every", type=int, default=100)
    a = ap.parse_args()
    p = dict(PRESETS[a.preset])
    if a.steps:
        p["steps"] = a.steps
    if a.device == "cpu":
        torch.set_num_threads(a.threads)
        lower_priority()

    train_text = (ROOT / "data" / "train.txt").read_text(encoding="utf-8")
    val_text = (ROOT / "data" / "val.txt").read_text(encoding="utf-8")
    t0 = time.time()
    tok = core.BPETokenizer.train(train_text, p["vocab"])   # learned on the training split only
    train_ids = torch.tensor(tok.encode(train_text))
    val_ids = torch.tensor(tok.encode(val_text, skip_unknown=True))
    print(f"BPE: vocab {tok.vocab_size}, {len(train_ids):,} train tokens, {len(train_ids) / len(train_text):.3f} tokens/char "
          f"({time.time() - t0:.0f}s)")
    cfg = core.Config(vocab_size=tok.vocab_size, block_size=p["block"], n_embd=p["n_embd"], n_heads=p["n_heads"],
                      n_layers=p["n_layers"], dropout=p.get("dropout", 0.1))
    model = core.GPT(cfg)
    print(f"GPT: {model.n_params():,} parameters, {p['steps']} steps x {p['batch']} x {p['block']} tokens on {a.device}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    best = [float("inf")]

    def on_eval(row):   # keep the best-validation checkpoint on disk as we go, so an interrupted run still leaves a model
        if row["val_loss"] < best[0]:
            best[0] = row["val_loss"]
            core.save_model(model, tok, out)

    rows = core.train(model, train_ids, val_ids, p["steps"], p["batch"], p["lr"], eval_every=a.eval_every,
                      device=a.device, on_eval=on_eval)
    with open(ROOT / "results" / "loss.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"best val loss {best[0]:.3f}; saved {out}/model.pt ({(out / 'model.pt').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
