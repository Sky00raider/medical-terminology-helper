# Results

Model: 939,264 parameters, vocab 1024, context 128 tokens. Evaluation text: 1148 held-out windows of 129 BPE tokens (146,944 predictions) from `data/val.txt`, text the model never saw in training. Every model scores the same windows with the same tokenizer.

## 1. Predicting held-out text

| model | loss (nats/token) | bits per character | next-token top-1 | top-5 |
|---|---|---|---|---|
| GPT (this project) | 2.999 | 1.472 | 36.9% | 60.0% |
| 1-gram (unigram) | 5.958 | 2.925 | 7.5% | 17.8% |
| 2-gram | 3.752 | 1.842 | 23.0% | 47.6% |
| 3-gram | 2.876 | 1.412 | 39.5% | 65.2% |
| 4-gram | 2.777 | 1.363 | 44.4% | 67.2% |

Uniform guessing would score 6.93 nats per token. Bits per character divides the total bits by the characters covered, so scores stay comparable if the tokenizer changes. The n-gram accuracies use a subsample of 800 positions (the same positions for every order).

## 2. Defining held-out dictionary terms

80 entries from Gould's dictionary that were held out of training. Prompt: `TERM. `; the completion is scored against the real definition. *Key-word overlap* is the share of the reference's content words the completion also uses. *Real-word rate* is the share of generated words that occur in the training corpus.

| method | key-word overlap | real-word rate | mean words |
|---|---|---|---|
| GPT greedy | 1.7% | 99.6% | 5.4 |
| GPT sampled (T=0.7) | 1.2% | 93.2% | 7.2 |
| 4-gram greedy | 3.6% | 99.8% | 13.5 |

Side by side (reference, GPT greedy, 4-gram greedy):

- **CHROMIDROSIS**
  - reference: A rare condition of the sweat in which it is variously colored, being bluish, blackish, reddish, greenish, or yellowish. Black szveat {Seborrhcea nigricans) occurs usually in hysteric women, the face being most often affected.
  - GPT: A poisonous disease of the pancreas.
  - 4-gram: The formation of the pelvis.
- **THYME**
  - reference: The genus Thymus, of the order Labiatae. Thymus vulgaris yields a volatile oil (Oleum thymi, U.
  - GPT: A pain of the pancreas.
  - 4-gram: A salt of the pelvis.
- **ARTERIOGRAPHY**
  - reference: 1. A description of the arteries. 2. The graphic representation of the pulse waves.
  - GPT: The pancreas of the pancreas.
  - 4-gram: The science of the cranial nerves, and the pulmonary artery, and is inserted into the lateral side of the forearm, and is inserted into
- **HEDGE-HOG CRYSTALS**
  - reference: Crystals of ammonium urate found as a urinary deposit in the form of globular crystals, with spiny projections.
  - GPT: A poisonous disease of the pulse.
  - 4-gram: Inflammation of the pelvis.
- **CENTAURY**
  - reference: A popular name for various plants of the genera Centaurea, Eiythrcea, Sabbatia, Chlora, etc. It is used as a simple, bitter tonic. Dose ss-j in decoction several times a day. Unof.
  - GPT: The pancreas of the pancreas.
  - 4-gram: The science of the cranial nerves, and the pulmonary artery, and is inserted into the lateral side of the forearm, and is inserted into
- **REINOCULATION**
  - reference: Inoculation a second time with the same kind of virus.
  - GPT: The power of the pancreas.
  - 4-gram: The act of rendered immune serum of the forearm, and is inserted into the lateral side of the forearm, and is inserted into the

## 3. Sample completions (temperature 0.7, seed 0)

- `Inflammation of the` ->  splanchnic cavity.
- `The femur is` ->  formed by the intercostal vessels.
- `GASTRITIS. ` -> Situated in which the bottom of teeth.
- `HYPERTENSION. ` -> The prominent for excess of the nature.
- `The function of the kidney` ->  is in the pelvic cavity.
- `The three layers of the heart wall` -> s of the pons and beneath the tympanic cavity.
- `The cerebrum` ->  is continuous with the posterior border of the tympanic fissure.
- `A fracture of the` ->  splanchnic acids.
- `Blood is carried from the heart by` ->  the brain.
- `PERICARDITIS. ` -> Pertaining to or characterized by the mesentery.
- `The pancreas secretes` ->  of the disease inhibitory, but little against, etc.
- `ANEURYSM. ` -> Situated in which the bottom of a little mye.

## 4. What the BPE tokenizer learned

Vocabulary 1024; the training text is 0.344 tokens per character (2.90 characters per token).

| term | tokens |
|---|---|
| gastrocnemius |  g · ast · ro · c · n · em · ius |
| pericarditis |  per · ic · ard · it · is |
| thrombocytopenia |  th · rom · b · oc · y · t · op · en · ia |
| nephrectomy |  ne · ph · re · ct · om · y |
| hemoglobin |  hem · og · lo · bin |
| cholecystitis |  ch · ole · c · y · st · it · is |

## 5. Nearest neighbours in the learned embedding space (cosine)

- ` heart`: ` brain` 0.65, ` blood` 0.57, ` system` 0.56, ` body` 0.56, ` pressure` 0.50, ` cord` 0.49
- ` artery`: ` arteries` 0.72, ` vein` 0.69, ` veins` 0.63, ` plexus` 0.61, ` vessels` 0.56, ` canal` 0.55

## 6. Training curve

| step | train loss | val loss | seconds |
|---|---|---|---|
| 0 | 6.940 | 6.937 | 2.0 |
| 200 | 4.235 | 4.231 | 98.6 |
| 400 | 3.907 | 3.900 | 185.2 |
| 600 | 3.645 | 3.647 | 271.1 |
| 800 | 3.417 | 3.409 | 355.7 |
| 1000 | 3.242 | 3.245 | 438.2 |
| 1200 | 3.125 | 3.129 | 524.9 |
| 1400 | 3.046 | 3.045 | 613.3 |
| 1600 | 2.982 | 2.993 | 706.3 |
| 1800 | 2.951 | 2.960 | 795.4 |

_Generated by `scripts/evaluate.py` in 110s._
