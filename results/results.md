# Results

Model: 11,525,376 parameters, vocab 2048, context 256 tokens. Evaluation text: 483 held-out windows of 257 BPE tokens (123,648 predictions) from `data/val.txt`, text the model never saw in training. Every model scores the same windows with the same tokenizer.

## 1. Predicting held-out text

| model | loss (nats/token) | bits per character | next-token top-1 | top-5 |
|---|---|---|---|---|
| GPT (this project) | 2.762 | 1.141 | 43.7% | 65.8% |
| 1-gram (unigram) | 6.307 | 2.605 | 6.9% | 20.6% |
| 2-gram | 3.982 | 1.645 | 22.9% | 47.9% |
| 3-gram | 3.298 | 1.362 | 38.0% | 60.9% |
| 4-gram | 3.271 | 1.351 | 41.6% | 62.1% |

Uniform guessing would score 7.62 nats per token. Bits per character divides the total bits by the characters covered, so scores stay comparable if the tokenizer changes. The n-gram accuracies use a subsample of 1600 positions (the same positions for every order).

## 2. Defining held-out dictionary terms

80 entries from Gould's dictionary that were held out of training. Prompt: `TERM. `; the completion is scored against the real definition. *Key-word overlap* is the share of the reference's content words the completion also uses. *Real-word rate* is the share of generated words that occur in the training corpus.

| method | key-word overlap | real-word rate | mean words |
|---|---|---|---|
| GPT greedy | 4.0% | 100.0% | 7.6 |
| GPT sampled (T=0.7) | 4.1% | 97.7% | 9.0 |
| 4-gram greedy | 4.6% | 100.0% | 9.8 |

Side by side (reference, GPT greedy, 4-gram greedy):

- **CHROMIDROSIS**
  - reference: A rare condition of the sweat in which it is variously colored, being bluish, blackish, reddish, greenish, or yellowish. Black szveat {Seborrhcea nigricans) occurs usually in hysteric women, the face being most often affected.
  - GPT: A monster with a microscope.
  - 4-gram: The formation of the penis, and is inserted into the lateral side of the body.
- **THYME**
  - reference: The genus Thymus, of the order Labiatae. Thymus vulgaris yields a volatile oil (Oleum thymi, U.
  - GPT: An instrument for measuring the size of the eye.
  - 4-gram: An instrument for measuring the degree of the order Composite.
- **ARTERIOGRAPHY**
  - reference: 1. A description of the arteries. 2. The graphic representation of the pulse waves.
  - GPT: The science of the science of the malleus.
  - 4-gram: The science of the head of the radius and the capsule; (2) the lateral, and medialward, and the posterior part of the body.
- **HEDGE-HOG CRYSTALS**
  - reference: Crystals of ammonium urate found as a urinary deposit in the form of globular crystals, with spiny projections.
  - GPT: A peculiar small peculiar pain in the skin of the eyelids.
  - 4-gram: Inflammation of the connective tissue, and the posterior part of the body.
- **CENTAURY**
  - reference: A popular name for various plants of the genera Centaurea, Eiythrcea, Sabbatia, Chlora, etc. It is used as a simple, bitter tonic. Dose ss-j in decoction several times a day. Unof.
  - GPT: The science of the microscope.
  - 4-gram: Plastic operation of the penis, and is inserted into the lateral side of the body.
- **REINOCULATION**
  - reference: Inoculation a second time with the same kind of virus.
  - GPT: The act of destroying the blood.
  - 4-gram: The act of the will.

## 3. Sample completions (temperature 0.7, seed 0)

- `Inflammation of the` ->  urethra.
- `The femur is` ->  the largest, and consists of three layers - an external layer. The anterior or meniscus is thicker than that on its posterior surface;
- `GASTRITIS. ` -> Inflammation of the cornea.
- `HYPERTENSION. ` -> The act of bacteria.
- `The function of the kidney` ->  is to say, and the function of the uterus has a distinct structure.
- `The three layers of the heart wall` ->  is composed of a layer of white and flattened epithelium, which enclose the bloodvessels.
- `The cerebrum` ->  is a thin, flattened plate of bone situated within the cavity and behind.
- `A fracture of the` ->  spleen, and characterized by the presence of mucin in the blood.
- `Blood is carried from the heart by` ->  the following classes:
- `PERICARDITIS. ` -> Inflammation of the cornea.
- `The pancreas secretes` ->  that secretin is split off from the pancreatic duct. It may be considered as being of the bile pigments or as a hydrolytic cleavage.
- `ANEURYSM. ` -> The treatment of diseases.

## 4. What the BPE tokenizer learned

Vocabulary 2048; the training text is 0.289 tokens per character (3.46 characters per token).

| term | tokens |
|---|---|
| gastrocnemius |  g · ast · roc · n · em · ius |
| pericarditis |  per · ic · ard · itis |
| thrombocytopenia |  th · rom · b · oc · y · t · op · en · ia |
| nephrectomy |  ne · ph · re · ct · om · y |
| hemoglobin |  hem · oglobin |
| cholecystitis |  ch · ole · cy · st · itis |

## 5. Nearest neighbours in the learned embedding space (cosine)

- `itis`: `osis` 0.40, `orum` 0.38, `lexor` 0.38, `illa` 0.35, `ammation` 0.34, `icles` 0.33
- ` heart`: ` lung` 0.49, ` beat` 0.47, ` aorta` 0.44, ` brain` 0.44, ` kidney` 0.43, ` ventricle` 0.43
- ` kidney`: ` liver` 0.60, ` lung` 0.57, ` bladder` 0.48, ` uterus` 0.47, ` rib` 0.46, ` intestine` 0.45
- ` artery`: ` vein` 0.62, ` arteries` 0.61, ` vessel` 0.50, ` vessels` 0.50, ` veins` 0.49, ` nerve` 0.46

## 6. Training curve

| step | train loss | val loss | seconds |
|---|---|---|---|
| 0 | 7.711 | 7.709 | 1.7 |
| 300 | 4.430 | 4.415 | 133.6 |
| 600 | 3.689 | 3.712 | 280.5 |
| 900 | 3.219 | 3.297 | 428.9 |
| 1200 | 2.942 | 3.065 | 577.6 |
| 1500 | 2.728 | 2.924 | 725.9 |
| 1800 | 2.584 | 2.833 | 874.2 |
| 2100 | 2.490 | 2.781 | 1022.5 |
| 2400 | 2.427 | 2.748 | 1170.5 |
| 2500 | 2.410 | 2.743 | 1219.9 |

_Generated by `scripts/evaluate.py` in 452s._
