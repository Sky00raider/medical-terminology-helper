"""Tests for app/core.py, grouped per milestone. They use tiny inline fixtures: no network, no downloads, no checkpoint."""
import math

import pytest
import torch

from app import core

TEXT = ("Gastrocnemius muscle. The calf muscle of the leg. Inflammation of the stomach is gastritis. "
        "The femur is the long bone of the thigh. The heart pumps blood through the arteries. ") * 12


# --- Milestone 1: Curate a small corpus of public domain medical texts (e.g., anatomy, physiology).
RAW = """
[Illustration: Fig. 12.]

The femur is the longest and strongest bone in the body. It articulates with the acetabulum above and the
tibia be-
low (see Fig. 123). The café of the shaft is cylindrical.

THE SKELETON OF THE LEG

12 | 14 | 33 | 5 | 70 | 82

The femur is the longest and strongest bone in the body. It articulates with the acetabulum above and the
tibia below (see Fig. 123). The café of the shaft is cylindrical.

Tbe vvh1ch 4nd tn%e s1gn ot 0f the b0ne 1s ,.; broken ~ up by noi5e here.
"""


def test_clean_text_rejoins_hyphenated_words_and_drops_figure_refs_and_accents():
    out = core.clean_text(RAW)
    assert "tibia below" in out and "Fig" not in out and "cafe of the shaft" in out
    assert all(ord(c) < 128 for c in out)


def test_clean_text_drops_headings_numbers_ocr_noise_and_duplicates():
    out = core.clean_text(RAW)
    assert out.count("The femur is the longest") == 1     # the duplicate paragraph is gone
    assert "SKELETON" not in out and "| 14 |" not in out and "noi5e" not in out
    assert "\n\n" not in out                               # one paragraph left


def test_is_clean_paragraph_thresholds():
    assert core.is_clean_paragraph("The heart is a hollow muscular organ lying in the chest.")
    assert not core.is_clean_paragraph("Too short here.")
    assert not core.is_clean_paragraph("THE HEART IS A HOLLOW MUSCULAR ORGAN LYING IN THE CHEST CAVITY")
    assert core.is_clean_paragraph("A state of partial unconsciousness.", min_words=3, min_chars=20)


def test_chunk_paragraph_splits_long_text_at_sentence_ends():
    para = " ".join(f"Sentence number {i} describes a muscle." for i in range(40))
    chunks = core.chunk_paragraph(para, max_chars=200)
    assert len(chunks) > 5 and all(len(c) <= 200 for c in chunks) and all(c.endswith(".") for c in chunks)
    assert core.chunk_paragraph("short.", 200) == ["short."]


DICT_RAW = """
Acetate (as-et-at) [acetum, vinegar]. Any salt of acetic
acid.

Acetin (asf-et-in) [acetum, vinegar], C3H5-(C2H302)3. A chemic compound formed by the union of glycerol and
acetic acid.

Aceto-acetic Acid.     See Acid, Diacetic.

Hydramnios {hi-dramf-ne-os) [vtfup, water]. An abnormal amount of amniotic fluid.

The heart pumps. Blood moves through the vessels of the whole body in a circle.

Acetate (as-et-at) [acetum, vinegar]. A duplicate entry that must not appear again.
"""


def test_parse_dictionary_extracts_terms_and_drops_pronunciation_and_etymology():
    entries = dict(core.parse_dictionary(DICT_RAW))
    assert entries["ACETATE"] == "Any salt of acetic acid."
    assert entries["ACETIN"].startswith("A chemic compound formed")
    assert entries["HYDRAMNIOS"] == "An abnormal amount of amniotic fluid."


def test_parse_dictionary_skips_cross_references_sentences_and_repeats():
    terms = [t for t, _ in core.parse_dictionary(DICT_RAW)]
    assert terms == ["ACETATE", "ACETIN", "HYDRAMNIOS"]


def test_shorten_definition_keeps_leading_sentences_and_drops_sub_entries():
    d = "Inflammation of the liver. It has many causes. A. Acute, a sudden form. B. Chronic, a slow form."
    assert core.shorten_definition(d) == "Inflammation of the liver. It has many causes."
    assert core.shorten_definition("x" * 400) == ""
    assert core.shorten_definition("I. A carbuncle. 2. An infection.") == "1. A carbuncle. 2. An infection."


def test_format_entry_is_term_dot_definition():
    assert core.format_entry("FEMUR", "The thigh bone.") == "FEMUR. The thigh bone."


def test_split_corpus_is_seeded_disjoint_and_complete():
    paras = [f"paragraph {i}" for i in range(100)]
    tr, va = core.split_corpus(paras, 0.1, seed=0)
    assert len(va) == 10 and not set(tr) & set(va) and sorted(tr + va) == sorted(paras)
    assert core.split_corpus(paras, 0.1, seed=0) == (tr, va)


def test_corpus_stats_counts_words_and_paragraphs():
    s = core.corpus_stats("The heart beats.\n\nThe heart rests.")
    assert s["words"] == 6 and s["unique_words"] == 4 and s["paragraphs"] == 2


def test_committed_corpus_and_glossary_are_well_formed():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "data"
    train = (root / "train.txt").read_text(encoding="utf-8")
    val = (root / "val.txt").read_text(encoding="utf-8")
    assert len(train) > 1_000_000 and len(val) > 50_000 and all(ord(c) < 128 for c in train[:200_000])
    glossary = core.load_glossary(root / "glossary.json")
    assert len(glossary) > 100 and all(t.isupper() and len(d) >= 40 for t, d in glossary)
    held_out_terms = {t for t, _ in glossary}
    assert not any(f"\n\n{t}. " in "\n\n" + train for t in list(held_out_terms)[:50])   # held-out entries are not trained on


# --- Milestone 2: Implement the byte-pair tokenizer and generate embeddings from the medical corpus.
@pytest.fixture(scope="module")
def tok():
    return core.BPETokenizer.train(TEXT, vocab_size=120)


def test_bpe_round_trips_text(tok):
    s = "The femur is the long bone of the thigh."
    assert tok.decode(tok.encode(s)) == s


def test_bpe_merges_frequent_words_into_few_tokens(tok):
    assert len(tok.tokens("Inflammation of the stomach")) < len("Inflammation of the stomach") / 2
    assert tok.vocab_size <= 120


def test_bpe_never_merges_across_words(tok):
    for t in tok.tokens("the femur is"):
        assert t.count(" ") <= 1 and (" " not in t[1:])


def test_bpe_unknown_characters_fail_loudly_or_are_skipped(tok):
    with pytest.raises(ValueError):
        tok.encode("café")
    assert tok.decode(tok.encode("café", skip_unknown=True)) == "caf"


def test_bpe_save_and_load(tok, tmp_path):
    tok.save(tmp_path / "t.json")
    again = core.BPETokenizer.load(tmp_path / "t.json")
    assert again.encode("The calf muscle") == tok.encode("The calf muscle")


def test_embeddings_add_token_and_position(tok):
    emb = core.Embeddings(tok.vocab_size, 16, 32)
    out = core.embed_text(tok, emb, "The femur")
    assert out.shape == (1, len(tok.encode("The femur")), 16)
    ids = torch.tensor([tok.encode("The femur")])
    assert torch.allclose(emb(ids), emb.tok(ids) + emb.pos(torch.arange(ids.shape[1])))


def test_nearest_tokens_returns_other_tokens_by_cosine(tok):
    emb = core.Embeddings(tok.vocab_size, 16, 32)
    single = next(t for t in tok.itos.values() if len(t) == 1 and t.isalpha())
    near = core.nearest_tokens(tok, emb.tok.weight, single, k=3)
    assert len(near) == 3 and all(t != single and -1 <= s <= 1 for t, s in near)
    with pytest.raises(ValueError):
        core.nearest_tokens(tok, emb.tok.weight, "The femur is")


# --- Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
CFG = core.Config(vocab_size=50, block_size=16, n_embd=32, n_heads=4, n_layers=2, dropout=0.0)


def test_attention_rows_sum_to_one_and_future_is_masked():
    attn = core.CausalSelfAttention(CFG).eval()
    _, w = attn(torch.randn(2, 8, 32), return_weights=True)
    assert w.shape == (2, 4, 8, 8)
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, 8), atol=1e-5)
    assert torch.all(torch.triu(w, diagonal=1) == 0)


def test_changing_a_future_token_does_not_change_earlier_outputs():
    model = core.GPT(CFG).eval()
    a = torch.randint(0, 50, (1, 10))
    b = a.clone()
    b[0, 7] = (b[0, 7] + 1) % 50
    la, lb = model(a)[0], model(b)[0]
    assert torch.allclose(la[:, :7], lb[:, :7], atol=1e-5)
    assert not torch.allclose(la[:, 7:], lb[:, 7:])


def test_block_keeps_shape_and_gpt_outputs_vocab_logits():
    x = torch.randn(2, 5, 32)
    assert core.Block(CFG)(x).shape == x.shape
    model = core.GPT(CFG)
    logits, loss = model(torch.randint(0, 50, (3, 12)), torch.randint(0, 50, (3, 12)))
    assert logits.shape == (3, 12, 50) and loss.ndim == 0


def test_untrained_loss_is_near_log_vocab_and_weights_are_tied():
    model = core.GPT(CFG)
    _, loss = model(torch.randint(0, 50, (16, 16)), torch.randint(0, 50, (16, 16)))
    assert abs(loss.item() - math.log(50)) < 0.5
    assert model.lm_head.weight is model.emb.tok.weight


def test_attention_matches_a_manual_single_head_computation():
    cfg = core.Config(vocab_size=10, block_size=8, n_embd=8, n_heads=1, n_layers=1, dropout=0.0)
    attn = core.CausalSelfAttention(cfg).eval()
    x = torch.randn(1, 4, 8)
    q, k, v = attn.qkv(x).split(8, dim=-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(8)
    scores = scores.masked_fill(torch.triu(torch.ones(4, 4), 1).bool(), float("-inf"))
    expected = attn.proj(torch.softmax(scores, -1) @ v)
    assert torch.allclose(attn(x), expected, atol=1e-5)


# --- Milestone 4: Set up the training loop and train the GPT model on the medical text data.
@pytest.fixture(scope="module")
def ids(tok):
    return torch.tensor(tok.encode(TEXT))


def test_get_batch_targets_are_inputs_shifted_by_one(ids):
    x, y = core.get_batch(ids, 16, 4, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_lr_schedule_warms_up_then_decays():
    lrs = [core.lr_at(s, 100, 1.0, warmup=10) for s in range(100)]
    assert lrs[0] < lrs[9] <= 1.0 and lrs[-1] < lrs[20] and lrs[-1] >= 0.1 - 1e-9


def test_training_lowers_the_loss(tok, ids):
    cfg = core.Config(vocab_size=tok.vocab_size, block_size=16, n_embd=32, n_heads=2, n_layers=2, dropout=0.0)
    model = core.GPT(cfg)
    rows = core.train(model, ids, ids, steps=60, batch_size=8, lr=3e-3, eval_every=30, log=lambda *_: None)
    assert rows[-1]["train_loss"] < rows[0]["train_loss"] - 0.5
    assert [r["step"] for r in rows] == [0, 30, 60]


def test_save_and_load_give_identical_logits(tok, tmp_path):
    cfg = core.Config(vocab_size=tok.vocab_size, block_size=16, n_embd=32, n_heads=2, n_layers=1)
    model = core.GPT(cfg).eval()
    core.save_model(model, tok, tmp_path)
    again, tok2 = core.load_model(tmp_path)
    x = torch.randint(0, tok.vocab_size, (1, 8))
    assert torch.allclose(model(x)[0], again(x)[0])
    assert tok2.encode("the femur") == tok.encode("the femur")


def test_ngram_baseline_probabilities_sum_to_one_and_learn_patterns():
    ng = core.NgramBaseline(3, vocab_size=6).fit([0, 1, 2, 3] * 50)
    assert math.isclose(sum(ng.distribution([0, 1])), 1.0, rel_tol=1e-6)
    assert ng.prob([0, 1], 2) > 0.8
    assert ng.nll([0, 1, 2, 3] * 5) < math.log(6)


def test_bits_per_char_is_tokenizer_independent_accounting():
    assert math.isclose(core.bits_per_char(math.log(2), 100, 100), 1.0)
    assert math.isclose(core.bits_per_char(math.log(2), 50, 100), 0.5)
