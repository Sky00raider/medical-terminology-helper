# Medical Terminology Helper

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/4quatvyjas/badge.svg)](https://ask.thili.ai/b/4quatvyjas)
[![tests](https://github.com/Sky00raider/medical-terminology-helper/actions/workflows/ci.yml/badge.svg)](https://github.com/Sky00raider/medical-terminology-helper/actions/workflows/ci.yml)

A command-line assistant that completes medical phrases and dictionary definitions for students learning terminology. The GPT behind it is written from scratch in PyTorch (own byte-pair tokenizer, own causal self-attention, no pretrained weights) and trained on four public-domain medical books: Gray's *Anatomy* (1918), Gould's *Student's Medical Dictionary* (1900), Howell's *Physiology* (1921) and Lewis's *Anatomy and Physiology for Nurses* (1913).

![The CLI completing a phrase, defining a term, listing next-token candidates and splitting a word into Greek/Latin roots](docs/demo.svg)

<sub>Real output of the commands shown, rendered by `python scripts/render_demo.py`.</sub>

> **Headline.** The 11.5M-parameter GPT, trained for 20 minutes on a free Colab T4, beats a 4-gram baseline on held-out text: **1.14 against 1.35 bits per character**, with next-token top-1 accuracy of 43.7% against 41.6% (top-5: 65.8% against 62.1%). A first, smaller model (0.94M parameters, 13 minutes on a laptop CPU) did *not* beat the 4-gram (1.47 bits per character), so scale mattered. What the model has **not** learned is what the terms mean: it writes fluent definitions in the right form (`GASTRITIS. Inflammation of the cornea.`) but with the wrong content, and its key-word overlap with the real definition is only 4.0%, no better than the baseline. Details, caveats and limits are below.

## Quick start

You need Python 3.12. Dependencies are pinned in `requirements.txt` (httpx 0.28.1, pytest 9.2.0, torch 2.15.0).

```bash
python3.12 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m app.cli complete "Inflammation of the"
```

The trained model (`models/model.pt`, 49 MB) and the cleaned corpus are committed, so nothing is downloaded and nothing needs training. Run the tests with `pytest` (51 tests, about 10 seconds, no network).

## Commands

| command | what it does |
|---|---|
| `complete "The femur is" -n 3 --words 20` | finish a partial phrase: 3 options, ranked by the model's confidence, about 20 words each |
| `define pericarditis` | dictionary-style definition (`PERICARDITIS. ...`) |
| `chat` | interactive: type a phrase, type `1`/`2`/`3` to keep an option and continue from it; `/define X`, `/roots X`, `/n`, `/words`, `/temp`, `/save`, `/saved` |
| `next "The femur is the" -k 5` | the most likely next tokens with probabilities (finishing the word you are typing) |
| `quiz -n 5` | flashcards: you define a held-out dictionary term, then see the real definition, the model's attempt and your key-word overlap |
| `roots gastroenteritis` | split a term into Greek/Latin parts (gastr + o + enter + itis) with meanings |
| `explain gastroenteritis` | the roots, then a short explanation from an optional chat model (see Settings below); the from-scratch GPT cannot explain terms reliably, so this is clearly a separate tutor |
| `neighbours " heart"` | tokens the model embeds closest to a token |
| `saved` | completions you kept (`complete --pick` saves one) |

Sampling flags on `complete`, `define` and `chat`: `--words` (length), `-n` (options), `--temperature` (0 = greedy), `--top-k`, `--top-p`, `--seed`. `--model DIR` loads another checkpoint.

## What goes beyond the brief

The brief asks for BPE, a causal-attention GPT, training, generation and a CLI. Added on top:

- **Several ranked completions and a length control** (the stretch ideas), plus saving and loading the model (`save_model`/`load_model`, committed checkpoint).
- **Definition mode, an interactive chat, next-token candidates** and a **study quiz** that scores a student's own definition against held-out dictionary entries the model never saw.
- **A Greek/Latin root explainer** (`roots`), an **optional tutor** (`explain`) that calls an Ollama, Groq or Gemini chat model through the starter's `app/llm.py`, a **saved-completions history** (through the starter's `Store`) and **embedding neighbours**.
- **A baseline and honest measurement**: interpolated n-gram models (orders 1 to 4) over the same BPE tokens, scored in bits per character (tokenizer-independent), next-token accuracy and definition overlap (`scripts/evaluate.py`).
- **A tokenizer-aware prompt rule**: words carry their leading space inside tokens, so a prompt that ends in a space is encoded without it. Without this, `"Gastrocnemius muscle. "` produced garbage in a test that the model had memorised.
- **Two presets:** `small` (0.94M parameters, a light CPU run) and `large` (11.5M, a Colab T4), plus a one-click Colab notebook. The committed model is the large one.

## Milestones

| # | Milestone | Where |
|---|---|---|
| 1 | Curate a public-domain corpus, clean it | `scripts/download_data.py`, `scripts/build_corpus.py`, `clean_text`, `parse_dictionary` in `app/core.py`, `data/`, `results/corpus_stats.md` |
| 2 | Byte-pair tokenizer and embeddings | `BPETokenizer`, `Embeddings`, `nearest_tokens` |
| 3 | Causal self-attention and the transformer block | `CausalSelfAttention`, `Block`, `GPT` |
| 4 | Training loop on the medical text | `train`, `lr_at`, `scripts/train.py`, `notebooks/colab_train.ipynb`, `models/`, `results/loss.csv` |
| 5 | Generation: sampling strategies and the loop | `filter_logits`, `generate_ids`, `complete`, `scripts/evaluate.py` |
| 6 | The CLI | `app/cli.py` |

All the model code is in [app/core.py](app/core.py), one section per milestone; tests are in [tests/test_core.py](tests/test_core.py), grouped the same way.

## Data and cleaning

| source (Internet Archive item) | what it contributes |
|---|---|
| Gould, *The Student's Medical Dictionary* (1900), [`studentsmedical00goul`](https://archive.org/details/studentsmedical00goul) | 8,833 parsed `TERM. definition` entries (494 held out) |
| Gray, *Anatomy of the Human Body* (1918), [`anatomyofhumanbo1918gray`](https://archive.org/details/anatomyofhumanbo1918gray) | 9,054 paragraphs |
| Howell, *A Text-book of Physiology* (1921), [`textbookofphysio00howe`](https://archive.org/details/textbookofphysio00howe) | 4,575 paragraphs |
| Lewis, *Anatomy and Physiology for Nurses* (1913), [`anatomyphysiolog00lewirich`](https://archive.org/details/anatomyphysiolog00lewirich) | 1,475 paragraphs |

All four were published before 1929 and are in the US public domain. `scripts/download_data.py` fetches each one's plain-text OCR from `https://archive.org/download/<item>/<item>_djvu.txt` into `data/raw/` (git-ignored); the cleaned result is committed, so nothing needs downloading to run the project. `clean_text` folds typography to ASCII, strips accents, rejoins words hyphenated across lines, drops figure references, headings, page furniture, number tables, all-capital running heads, OCR-garbled paragraphs and duplicates; long paragraphs are cut at sentence ends. `parse_dictionary` turns `Acetate (as-et-at) [acetum, vinegar]. Any salt of acetic acid.` into `ACETATE. Any salt of acetic acid.` and keeps only the leading sentences, so definitions are short and clean. Result: **6.78M training characters (1.10M words, 31,471 distinct)** and **0.43M validation characters**. The validation split is separate paragraphs plus 6% of dictionary entries chosen by a hash of the term; those held-out entries feed `quiz` and the definition test. The tokenizer is trained on the training split only.

## Architecture and training

```
text -> BPE tokens (vocab 2048) -> token embedding + position embedding           (B, T, 384)
     -> 6 x [ LayerNorm -> causal multi-head self-attention (6 heads) -> + residual
              LayerNorm -> feed-forward (4x, GELU)                    -> + residual ]
     -> LayerNorm -> output layer tied to the token embedding -> next-token distribution
```

**The committed model (`--preset large`):** 11,525,376 parameters, context 256 tokens, dropout 0.2. AdamW (lr 1.5e-3, weight decay 0.1 on matrices only), linear warm-up then cosine decay, gradient clipping at 1.0, batch 64, 2,500 steps (41M tokens, about 21 passes over the corpus), 1,220 seconds on a Colab T4. Validation loss fell from 7.71 (uniform guessing is ln 2048 = 7.62) to 2.743 while training loss reached 2.410, so it is starting to overfit but was still improving at the last step (2.748 to 2.743 over the final 100 steps); the best-validation checkpoint is the one saved.

**The first model (`--preset small`):** 0.94M parameters, vocab 1024, context 128, 1,800 steps, 795 seconds on 8 CPU threads at below-normal priority. Its validation loss was 2.96 with training loss 2.95 (underfitting). I keep it as a point of comparison; its numbers below come from that earlier run and were not regenerated.

## Results

Full tables, side-by-side definitions and the training curve are in [results/results.md](results/results.md), produced by `python scripts/evaluate.py` on the committed model.

**Predicting held-out text** (123,648 predictions on 483 windows of text the model never saw; the n-gram models use the same BPE tokens):

| model | bits per character | next-token top-1 | top-5 |
|---|---|---|---|
| **GPT large** (this project, 11.5M) | **1.141** | **43.7%** | **65.8%** |
| 4-gram | 1.351 | 41.6% | 62.1% |
| 3-gram | 1.362 | 38.0% | 60.9% |
| 2-gram | 1.645 | 22.9% | 47.9% |
| unigram | 2.605 | 6.9% | 20.6% |
| *GPT small (0.94M, earlier run, vocab 1024; its 4-gram scored 1.363)* | *1.472* | *36.9%* | *60.0%* |

Bits per character divides the total bits by the characters covered, so scores are comparable across the two tokenizers. The n-gram accuracies use a subsample of 1,600 positions; the GPT uses all of them. The large GPT sees up to 256 tokens of context, the n-grams at most 3 (and the small GPT 128).

**Defining 80 held-out dictionary terms** (prompt `TERM. `, scored against Gould's real definition):

| method | key-word overlap | real-word rate | mean words |
|---|---|---|---|
| GPT large, greedy | 4.0% | 100.0% | 7.6 |
| GPT large, sampled (T=0.7) | 4.1% | 97.7% | 9.0 |
| 4-gram, greedy | 4.6% | 100.0% | 9.8 |

Real outputs (temperature 0.7, seed 0; the prompt is on the left):

```
Inflammation of the           -> urethra.
GASTRITIS.                    -> Inflammation of the cornea.
PERICARDITIS.                 -> Inflammation of the cornea.
HYPERTENSION.                 -> The act of bacteria.
The femur is                  -> the largest, and consists of three layers - an external layer. ...
The pancreas secretes         -> that secretin is split off from the pancreatic duct. ...
```

```
$ python -m app.cli define pericarditis -n 2 --seed 3
  1. PERICARDITIS. Inflammation of the temporal bone.   [-0.65]
  2. PERICARDITIS. Inflammation of the spinal cord and a tract of the spinal cord.   [-1.20]
```

## Findings

1. **Scale decided it.** The 0.94M-parameter CPU model lost to a 4-gram (1.47 against 1.36 bits per character). The 11.5M-parameter T4 model beats the 4-gram by 0.21 bits per character (1.14 against 1.35) and by 2.1 points of top-1 accuracy. Caveats: one run each, different context lengths and a different tokenizer (compared through bits per character), and the n-gram accuracies are on a subsample.
2. **It learned the form of a definition and the meaning of word endings, not the facts.** Both `GASTRITIS` and `PERICARDITIS` start with "Inflammation of the", so the `-itis` suffix is understood, but the organ is wrong ("cornea", "temporal bone"). Other definitions are fluent and wrong ("THYME: An instrument for measuring the size of the eye"). Key-word overlap is 4.0%, no better than the 4-gram's 4.6%, which writes longer, wordier definitions.
3. **Its next-token guesses are sensible.** After "The femur is the" the top choices are ` most`, ` larg(est)` and ` long`.
4. **The embeddings picked up anatomy.** Nearest neighbours of ` kidney` are ` liver`, ` lung`, ` bladder`, ` uterus`; of ` artery`, ` vein`, ` arteries`, ` vessel`; of ` heart`, ` lung`, ` beat`, ` aorta`, ` ventricle`; of `itis`, `osis` and `ammation`.
5. **BPE still fragments rare terms** (3.46 characters per token): `hemoglobin` is `hem · oglobin` and `pericarditis` is `per · ic · ard · itis`, but `gastrocnemius` is 6 tokens and `thrombocytopenia` 9.
6. **More data would help more than more steps:** the gap between training and validation loss (2.41 against 2.74) after 21 passes over a 2M-token corpus points at the data size.

## Limitations

- **Do not use it for medical facts.** It generates plausible text, not verified knowledge, and its sources are 100 years old: terminology, doses ("Dose gr. ij-x") and treatments are outdated. Most definitions it writes for held-out terms are wrong.
- One training run per model, one seed; no confidence intervals. The large and small models differ in context length and tokenizer, so the comparison between them is through bits per character only. The small model's row was not regenerated.
- The key-word overlap metric is crude (it ignores synonyms and word order), and the n-gram accuracies use a subsample.
- The corpus is OCR text; some noise survives cleaning (garbled words in dictionary entries, "I." for "1.").
- Rare terms are fragmented by the BPE vocabulary (finding 5), and the 2M-token corpus is small for 11.5M parameters (finding 6).
- `explain` needs a chat model of your own (a local Ollama or a free Groq/Gemini key) and is not part of the evaluation; without one it prints the roots and says the tutor is unavailable. Everything else runs locally with no key.

## Reproduce everything

```bash
python scripts/download_data.py     # 4 books from archive.org into data/raw/ (about 12 MB)
python scripts/build_corpus.py      # clean -> data/train.txt, data/val.txt, data/glossary.json, results/corpus_stats.md
python scripts/train.py --preset large --device cuda   # the committed model: a Colab T4, about 20 minutes
python scripts/train.py             # small preset on CPU (0.94M parameters), about 13 minutes on 8 threads
python scripts/evaluate.py          # -> results/results.md (about 8 minutes on a laptop CPU for the large model)
python scripts/render_demo.py       # -> docs/demo.svg from the real CLI
pytest
```

**Colab T4** (this is how the committed model was trained): open [notebooks/colab_train.ipynb](https://colab.research.google.com/github/Sky00raider/medical-terminology-helper/blob/main/notebooks/colab_train.ipynb) in Colab, pick a T4 runtime and Run all. It trains `--preset large --device cuda`; the evaluation cell is slow on Colab's CPU, so I ran `scripts/evaluate.py` locally instead. The best-validation checkpoint is kept, so overfitting cannot make the saved model worse.

## Settings and secrets

Only the optional `explain` command uses `.env`: copy `.env.example` to `.env` and fill in the three `LLM_*` lines for Ollama, Groq or Gemini. `.env` is ignored by git; never commit it or paste a key into code. The rest of the project needs no settings (`DATA_PATH` only moves the saved-completions file).
