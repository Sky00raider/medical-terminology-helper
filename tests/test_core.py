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


# --- Milestone 5: Implement a text generation function to complete medical phrases or definitions.
@pytest.fixture(scope="module")
def trained(tok, ids):
    cfg = core.Config(vocab_size=tok.vocab_size, block_size=32, n_embd=48, n_heads=2, n_layers=2, dropout=0.0)
    model = core.GPT(cfg)
    core.train(model, ids, ids, steps=250, batch_size=16, lr=4e-3, eval_every=250, log=lambda *_: None)
    return model


def test_filter_logits_top_k_and_top_p():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]])
    assert torch.isfinite(core.filter_logits(logits, top_k=2)).sum() == 2
    probs = torch.softmax(logits, -1)
    kept = torch.isfinite(core.filter_logits(logits, top_p=0.5)).sum().item()
    assert kept == 1 and probs[0, 0] > 0.5
    assert torch.isfinite(core.filter_logits(logits, top_p=0.99)).sum() >= 3


def test_greedy_is_deterministic_and_seeded_sampling_is_repeatable(trained, tok):
    p = tok.encode("Inflammation of the ")
    a, _ = core.generate_ids(trained, p, 10, temperature=0)
    b, _ = core.generate_ids(trained, p, 10, temperature=0)
    assert a == b
    s1, _ = core.generate_ids(trained, p, 10, seed=3)
    s2, _ = core.generate_ids(trained, p, 10, seed=3)
    assert s1 == s2


def test_trained_model_completes_a_memorised_definition(trained, tok):
    out = core.complete(trained, tok, "Gastrocnemius muscle. ", max_words=6, temperature=0, repetition_penalty=1.0)
    assert out[0].text.startswith("The calf muscle")


def test_complete_returns_distinct_options_best_first_and_respects_length(trained, tok):
    opts = core.complete(trained, tok, "The ", max_words=5, n=3, temperature=1.0, top_p=None, seed=1)
    assert 1 <= len(opts) <= 3 and len({o.text for o in opts}) == len(opts)
    assert [o.score for o in opts] == sorted((o.score for o in opts), reverse=True)
    assert all(len(o.text.split()) <= 5 for o in opts)


def test_next_token_candidates_are_sorted_probabilities(trained, tok):
    top = core.next_token_candidates(trained, tok, "Inflammation of the", k=4)
    assert len(top) == 4 and all(0 < p <= 1 for _, p in top)
    assert [p for _, p in top] == sorted((p for _, p in top), reverse=True)


def test_trim_cuts_at_paragraph_word_limit_and_sentence_end():
    assert core._trim("one two three four five six", 3) == "one two three"
    assert core._trim("The calf muscle of the leg. The cal", 20) == "The calf muscle of the leg."
    assert core._trim("first\n\nsecond", None) == "first"
    assert core._trim("\nstarts after a newline\nnot this", None) == "starts after a newline"


def test_define_prompt_uses_dictionary_format():
    assert core.define_prompt(" femur ") == "FEMUR. "


# --- Milestone 6: Create a CLI for students to enter partial phrases and get completions.
from app import cli  # noqa: E402
from app.storage import Store  # noqa: E402


@pytest.fixture()
def model_dir(tmp_path, trained, tok):
    core.save_model(trained, tok, tmp_path / "m")
    return str(tmp_path / "m")


def test_explain_term_splits_greek_and_latin_parts():
    assert [p for p, _ in core.explain_term("gastroenteritis")] == ["gastr", "o", "enter", "itis"]
    assert dict(core.explain_term("nephrectomy"))["ectomy"] == "surgical removal"
    assert dict(core.explain_term("gastroenteritis"))["o"] == "(connecting vowel)"
    assert core.explain_term("zzzq") == [("zzzq", "?")]


def test_overlap_score_counts_the_reference_keywords_used():
    ref = "The long bone of the thigh"
    assert core.overlap_score("it is the thigh bone", ref) == pytest.approx(2 / 3)
    assert core.overlap_score("", ref) == 0.0


def test_history_saves_through_the_store(tmp_path):
    h = core.History(Store(str(tmp_path / "s.json")))
    assert h.add("a ", "b") == 1 and h.items() == [{"prompt": "a ", "completion": "b"}]
    h.clear()
    assert h.items() == []


def test_cli_complete_prints_numbered_options(model_dir, capsys):
    cli.main(["--model", model_dir, "complete", "Inflammation of the", "-n", "2", "--words", "5", "--seed", "1"])
    out = capsys.readouterr().out
    assert out.startswith("  1. Inflammation of the")


def test_cli_define_next_and_neighbours(model_dir, capsys):
    cli.main(["--model", model_dir, "define", "femur", "-n", "1", "--seed", "0"])
    assert "FEMUR. " in capsys.readouterr().out
    cli.main(["--model", model_dir, "next", "Inflammation of the", "-k", "3"])
    assert capsys.readouterr().out.count("%") == 3
    cli.main(["--model", model_dir, "neighbours", "e", "-k", "2"])
    assert len(capsys.readouterr().out.strip().splitlines()) == 2


def test_cli_roots_prints_each_part(capsys):
    cli.main(["roots", "gastroenteritis"])
    out = capsys.readouterr().out
    assert "stomach" in out and "inflammation" in out


def test_cli_chat_session_picks_an_option_and_saves(model_dir, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "chat.json"))
    lines = iter(["The heart", "1", "/roots nephritis", "/n 2", "/save", "/saved", "/bogus", "/quit"])
    cli.main(["--model", model_dir, "chat", "--seed", "1"], inp=lambda _: next(lines))
    out = capsys.readouterr().out
    assert "kept: The heart" in out and "n = 2" in out and "saved (" in out and "unknown command" in out


def test_cli_quiz_scores_the_students_answer(tmp_path, monkeypatch, capsys):
    gloss = tmp_path / "g.json"
    gloss.write_text('[["FEMUR", "The long bone of the thigh."]]')
    answers = iter(["the long thigh bone"])
    cli.main(["quiz", "-n", "1", "--no-model", "--glossary", str(gloss)], inp=lambda _: next(answers))
    out = capsys.readouterr().out
    assert "Define: FEMUR" in out and "100%" in out


def test_cli_without_a_trained_model_gives_a_clear_message(tmp_path):
    with pytest.raises(SystemExit) as e:
        cli.main(["--model", str(tmp_path / "none"), "complete", "x"])
    assert "scripts/train.py" in str(e.value)
