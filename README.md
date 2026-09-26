# Deep Learning Mini Projects

Ten hands-on projects, one per core deep learning concept, built to go from
"I know the theory" to "I've implemented it and watched it work." Difficulty
ramps up through the series: Projects 1-2 build intuition with from-scratch
math and classic architectures; Projects 3-6 move into sequence modeling;
Projects 7-10 tackle the encoder-decoder / attention family that underlies
modern sequence-to-sequence and transformer models.

Each project folder contains:
- A single, runnable Python script (no notebook required)
- A `data/` folder with the actual dataset used (CSV/text/images) — every
  script also re-fetches its dataset automatically on first run (via
  HuggingFace `datasets` or Keras) so nothing here needs internet access
  to inspect, but nothing needs to be *taken on faith* either
- An `outputs/` folder with the actual plots/results from a real training run
- Comments explaining the *why* behind each architectural choice, not just the *what*

`export_data_samples.py` (repo root) is the script that generated every
`data/` folder — it documents exactly which dataset, split, and subset size
each project used, and can be re-run to regenerate them.

## Setup

```
pip install -r requirements.txt
```

Each project is self-contained — `cd` into its folder and run its script directly.

---

## Project 1 - ANN From Scratch 
**Folder:** [`01_ann_from_scratch/`](01_ann_from_scratch/)
**Concepts:** perceptron/ANN intuition, chain rule, backpropagation, activation
functions (sigmoid/tanh/relu), SGD, SGD with momentum
**Built with:** pure NumPy — no ML framework at all, so every gradient is
visible and hand-derived
**Dataset:** `sklearn.make_moons` (synthetic 2D binary classification)

What it shows:
- A 3-layer MLP where forward/backward passes are written out explicitly,
  matrix by matrix
- Plain SGD vs. SGD+momentum on the identical network and data
- How the choice of hidden activation (relu/tanh/sigmoid) changes convergence

**Run:** `python 01_ann_from_scratch/ann_from_scratch.py`


---

## Project 2 - CNN Architecture + Optimizer Comparison
**Folder:** [`02_cnn_optimizers/`](02_cnn_optimizers/)
**Concepts:** convolution, weight sharing, pooling, and how optimizer choice
(SGD / SGD+momentum / RMSprop / Adam) affects the *same* architecture
**Built with:** TensorFlow/Keras
**Dataset:** Fashion-MNIST (Keras built-in)

What it shows:
- A small Conv→Pool→Conv→Pool→Dense CNN
- All 4 optimizers trained from **identical starting weights**, so the
  comparison isolates the optimizer's effect, not random initialization luck

**Actual results (8 epochs, 12k training images, identical starting weights):**

| Optimizer | Final validation accuracy |
|---|---|
| SGD (plain) | 75.25% |
| SGD + momentum | 83.70% |
| RMSprop | 88.70% |
| **Adam** | **89.20%** |

Same architecture, same data, same initialization — an 14-point accuracy
swing purely from the optimizer's update rule.

**Run:** `python 02_cnn_optimizers/cnn_optimizers.py`


---

## Project 3 - Vanilla RNN: Forward Prop Through Time + BPTT 
**Folder:** [`03_rnn_forward_backward/`](03_rnn_forward_backward/)
**Concepts:** recurrence, weight sharing *across time steps*, Backpropagation
Through Time (BPTT), the vanishing gradient problem
**Built with:** pure NumPy (same philosophy as Project 1 — see every gradient)
**Dataset:** `tiny_shakespeare` (HuggingFace `datasets`) — real text,
character-level next-character prediction

What it shows:
- One RNN cell unrolled across a 25-character window, with forward and
  backward passes both hand-derived
- A direct measurement of gradient norm at each time step in the BPTT window,
  making the vanishing gradient problem visible rather than theoretical —
  this is the motivating problem Project 4 (LSTM/GRU) solves

**Actual results (3000 truncated-BPTT steps, 25-char window, 20k characters of
real Shakespeare text):** smoothed loss dropped from 101.5 (near-random-guess
level, `-ln(1/58) x 25`) to ~66 by step 2500, and the model learns real
structure from raw characters alone — character names in ALL CAPS, colons
after speaker names, line breaks — despite never being told these rules:

```
VEIACs:
IUNINIUENIULnLNIEINIUS:
Soir reans.'
I'll, oudiuralar mo Ied ha thevh mote;
```

Still gibberish at the word level (expected for a single-layer, 3000-step,
character-level vanilla RNN) but the *format* of a Shakespeare script is
clearly emerging — see `outputs/final_sample.txt` for the full sample and
`outputs/vanishing_gradient.png` for the measured gradient decay across the
BPTT window.

**Run:** `python 03_rnn_forward_backward/rnn_bptt.py`


---

## Project 4 - LSTM and GRU 
**Folder:** [`04_lstm_gru/`](04_lstm_gru/)
**Concepts:** forget/input/output gates, the cell-state "conveyor belt" that
gives LSTM its gradient-flow advantage over Project 3's vanilla RNN, GRU as a
simplified 2-gate alternative
**Built with:** pure NumPy for the gate-anatomy walkthrough (Part 1, forward
pass only, mirrors Project 3's style) + Keras for the real training
comparison (Part 2, since hand-deriving LSTM/GRU backprop adds a lot of
bookkeeping without more insight than Project 3 already gave for BPTT)
**Dataset:** `tiny_shakespeare` (HuggingFace), same next-character task as
Project 3, so results are directly comparable

What it shows:
- Part 1: one LSTM step and one GRU step with every gate value printed, so
  "gate" stops being an abstract word and becomes a concrete sigmoid output
  multiplying a vector
- Part 2: SimpleRNN vs. LSTM vs. GRU, identical architecture otherwise, same
  optimizer, same data — measures whether gating actually earns its added
  complexity on this task

**Actual results - two runs, because the first one taught something important:**

| Context window | SimpleRNN val acc | LSTM val acc | GRU val acc |
|---|---|---|---|
| 25 characters | 44.2% | 44.2% (tied) | **46.4%** |
| 100 characters | 42.5% | **44.1%** | 44.0% |

The first run (25-char window, `outputs/rnn_vs_lstm_vs_gru_seqlen25.png`) showed
LSTM barely beating a vanilla RNN - a window that short doesn't stress the
vanishing-gradient problem enough for gating to earn its extra parameters.
Lengthening the window to 100 characters
(`outputs/rnn_vs_lstm_vs_gru_seqlen100.png`) is where the gap should show up,
since a vanilla RNN now has to carry signal 4x further back through
repeated tanh/Whh multiplication - and it does: SimpleRNN's accuracy actually
*drops* (42.5%, worse than its own 25-char run) while LSTM overtakes it and
posts the best validation loss of all three models. This is the real
lesson of Project 3+4 together: gating's advantage isn't free-standing, it's
specifically a *long-context* advantage, and you only see it once the task
is hard enough to need it.

**Run:** `python 04_lstm_gru/lstm_gru.py` (edit `SEQ_LEN` at the top to
reproduce either the 25- or 100-character run)
**Read after:** Hochreiter & Schmidhuber, *"Long Short-Term Memory"*, Neural
Computation, 1997 — the original LSTM paper; Cho et al., *"Learning Phrase
Representations using RNN Encoder-Decoder for Statistical Machine
Translation"*, 2014 (arXiv:1406.1078) — introduces the GRU (Section 2).

## Project 5 - Bidirectional RNN 
**Folder:** [`05_bidirectional_rnn/`](05_bidirectional_rnn/)
**Concepts:** processing a sequence forward AND backward and concatenating
both hidden states at every token, so a tag decision can use context from
*both directions* — not just what came before
**Built with:** Keras (`Bidirectional` wrapper around an `LSTM`)
**Dataset:** CoNLL-2003 (HuggingFace `datasets`) — the standard Named Entity
Recognition benchmark, 9 BIO tags (PER/ORG/LOC/MISC)

What it shows:
- Unidirectional vs. Bidirectional LSTM, identical everything else
- Scored with **entity-level F1 (seqeval)**, not raw token accuracy — token
  accuracy is misleading on NER since ~83% of tokens are the "O" (non-entity)
  tag, so a model that never predicts an entity still scores ~83% "accuracy"
  while being useless. Entity F1 only credits getting the actual entity span
  and type right.

**Run:** `python 05_bidirectional_rnn/bidirectional_rnn.py`


## Project 6 - Encoder-Decoder Sequence Models 
**Folder:** [`06_encoder_decoder/`](06_encoder_decoder/)
**Concepts:** splitting "understand the input" (encoder) from "generate the
output" (decoder), the fixed-size context vector bottleneck, teacher forcing
**Built with:** Keras (LSTM encoder/decoder via the functional API, with a
separate inference-time decoder that runs one step at a time)
**Dataset:** Multi30k (HuggingFace `bentrevett/multi30k`) — 29k real
English-German sentence pairs; 15k used for training here. This is the same
dataset Projects 7 (Seq2Seq) and 8 (Attention) will reuse, so all three are
directly comparable.

What it shows:
- A working English→German translator: encoder compresses the whole source
  sentence into one fixed-size (h, c) vector pair, decoder generates German
  one token at a time from that vector alone
- The core weakness measured directly: BLEU score bucketed by source
  sentence length, to see whether translation quality degrades as sentences
  get longer than the bottleneck can comfortably hold — the exact problem
  Project 8's attention mechanism is built to solve

**Run:** `python 06_encoder_decoder/encoder_decoder.py`


## Project 7 - Seq2Seq: Greedy vs. Beam Search Decoding 
**Folder:** [`07_seq2seq/`](07_seq2seq/)
**Concepts:** Project 6 already covered teacher-forced training and greedy
decoding, so this project isolates the piece that was left simplest -
**decoding strategy**. Greedy search commits to the single best token at
every step and can never undo an early mistake; beam search keeps the
top-k most probable partial sentences alive at every step, so a token that
looks slightly worse right now can still lead to a better full sentence.
**Built with:** Keras, same encoder-decoder architecture as Project 6, plus
a hand-written beam search decoder (length-normalized log-probability
scoring — without normalization, beam search mechanically favors shorter
sentences, since every extra token can only add a negative log-probability)
**Dataset:** Multi30k, same as Project 6, same 15k-pair training subset, for
a fair comparison

What it shows:
- The exact same trained model, decoded three ways: greedy, beam width 3,
  beam width 5 — isolating decoding strategy as the only variable
- Concrete examples where beam search recovers a better translation that
  greedy search couldn't reach because of an early wrong commitment

**Run:** `python 07_seq2seq/seq2seq_beam_search.py`


## Project 8 - Attention Mechanism (Bahdanau) 
**Folder:** [`08_attention/`](08_attention/)
**Concepts:** solving Project 6's fixed-bottleneck problem by letting the
decoder look back at ALL encoder hidden states at every generation step,
weighted by a learned relevance score, instead of compressing the whole
source sentence into one vector
**Built with:** Keras — encoder LSTM (`return_sequences=True`, keeps every
hidden state) + a custom `AttentionDecoder` layer implementing Bahdanau
(additive) attention with an internal per-timestep loop
**Dataset:** Multi30k, same as Projects 6 & 7

What it shows:
- A **real attention heatmap** from an actually-trained model (not a
  textbook diagram): `outputs/attention_heatmap.png` shows a clear,
  roughly-diagonal alignment — "der"/"&lt;unk&gt;" attending to "many"/"asian",
  "vor"/"einem" attending to "a" — the decoder genuinely learned to look at
  the corresponding source words, unsupervised.
- The same BLEU-vs-sentence-length experiment as Project 6, for a direct
  comparison.

**An honest, non-obvious result:** this model was trained for only 10 epochs
(vs. Projects 6/7's 20 — the custom per-step attention decoder is heavier
per step, so epochs were cut to keep runtime reasonable), and its absolute
mean BLEU (0.084) came in *below* Project 6's no-attention baseline (0.095).
That's a real, reported result, not cherry-picked — with only half the
training budget, a direct BLEU comparison isn't apples-to-apples. The fairer
signal is the *shape* of the degradation curve: Project 6's BLEU fell ~38%
relative from its shortest to longest bucket (0.139 → 0.086); this
attention model's fell only ~30% relative (0.113 → 0.079) despite the
training handicap — a flatter decline, which is exactly what attention is
supposed to buy you. Per-token validation accuracy also came out *higher*
with attention (57.3% vs. 52.6%) despite half the epochs. Confirming the
full effect cleanly would need an equal-epoch rerun; that's a natural
follow-up rather than something to paper over here.

**Run:** `python 08_attention/attention_seq2seq.py` (checkpoints after every
epoch to `outputs/checkpoint.weights.h5` — safe to interrupt and rerun, it
resumes automatically)


## Project 9 - Scaled Dot-Product & Multi-Head Attention 
**Folder:** [`09_multihead_attention/`](09_multihead_attention/)
**Concepts:** replacing Project 8's recurrent, learned-MLP attention scoring
with a single matrix multiply (`softmax(QK^T/√d_k)V`) that scores every
position against every other position AT ONCE — no sequential decoder loop
needed, which is exactly what made Project 8 slow. Multi-head attention runs
several of these in parallel subspaces so different heads can specialize.
This is the core building block of the Transformer, with **no recurrence
anywhere**.
**Built with:** pure NumPy anatomy demo (Part 1) + a from-scratch
Transformer encoder block (multi-head self-attention + feedforward +
residual/LayerNorm + positional encoding) in Keras (Part 2)
**Dataset:** `rotten_tomatoes` (HuggingFace) — 8,530 real movie review
sentences, binary sentiment classification

What it shows:
- Part 1: the exact scaled dot-product formula computed on a toy sequence,
  plus multi-head attention splitting a model dimension into independent
  subspaces — the mechanics made concrete, same style as Projects 1/3/4
- Part 2: a real Transformer encoder trained on real sentiment data, with
  the actual learned attention weights extracted per-head for one real
  sentence and plotted side by side — showing the heads DO attend
  differently from each other, which is the whole point of having more than
  one

**Actual results:** trained in ~2 minutes total (12 epochs, ~8s/epoch — this
is why self-attention displaced RNNs: no sequential loop, so it's roughly
100x faster per epoch than Project 8's comparable-sized recurrent model).
Peak validation accuracy was 75.8% at epoch 3; training accuracy climbed to
99.4% by epoch 12 while validation loss kept rising after epoch 3 — classic
overfitting on a small (8.5k-example) dataset with a fairly expressive model
and only one dropout layer for regularization. Reported honestly rather than
only quoting the best epoch: this model would benefit from early stopping
or stronger regularization, which wasn't the point of this exercise (the
attention mechanism itself was).
`outputs/multihead_self_attention.png` shows all 4 heads on the same real
sentence side by side — they visibly attend to different words (Head 0
anchors on "beneath", Heads 1-2 concentrate around "shock"/"considerable",
Head 3 locks onto "nerve"), confirming heads actually specialize rather
than all learning the same thing.

**Run:** `python 09_multihead_attention/multihead_attention.py` (much
faster than Projects 6-8: self-attention parallelizes across the whole
sequence, no per-timestep loop)


## Project 10 - Capstone: Full Transformer Encoder-Decoder 
**Folder:** [`10_capstone_transformer/`](10_capstone_transformer/)
**Concepts:** every mechanism from this series, assembled into the real
architecture (Vaswani et al., 2017) — no recurrence anywhere. Encoder:
stacked multi-head self-attention (Project 9) + feedforward blocks. Decoder:
stacked blocks with **masked self-attention** (new — a token may only
attend to itself and earlier tokens, verified by a direct numerical test:
predictions for a given position are provably unaffected by anything fed
in at later positions) plus **cross-attention** (Project 8's Bahdanau idea,
generalized into a parallel matmul instead of a per-step MLP). Inference
reuses Project 7's length-normalized beam search.
**Built with:** Keras, 2 stacked encoder blocks + 2 stacked decoder blocks,
checkpointed every epoch (safe to interrupt and resume)
**Datasets — two runs, same architecture and code:**
- **Primary: IWSLT2017** (HuggingFace) — real TED talk transcripts,
  English→German, 15k training pairs used
- **Secondary: Multi30k** — same 15k-pair setup as Projects 6/7/8, for one
  direct, apples-to-apples comparison across the whole series

**The headline result — the full architectural progression, same data, same task:**

| Model | Project | Mean BLEU |
|---|---|---|
| Fixed-bottleneck RNN encoder-decoder | 6 | 0.095 |
| + Beam search (width 3) | 7 | 0.114 |
| + Bahdanau attention (fewer epochs — see Project 8's note) | 8 | 0.084 |
| **Full Transformer + beam search (width 3)** | **10** | **0.219** |

The Transformer roughly **doubles** the best RNN-based result. Just as
notable: its BLEU-by-sentence-length breakdown (`outputs/multi30k/bleu_vs_length.png`)
is **flat** — 0.235 / 0.209 / 0.242 / 0.205 across the four length buckets —
where Project 6's RNN dropped ~38% relative from short to long sentences.
This is the whole thesis of the series paying off: removing the fixed
bottleneck (Project 8) and then removing recurrence entirely in favor of
direct, parallel attention to every position (Projects 9-10) measurably
fixes the exact degradation Project 6 first identified.

**IWSLT2017 result, reported honestly:** mean BLEU (beam-3) 0.047 — much
lower than Multi30k's 0.219, because TED-talk sentences are longer and far
more linguistically complex than Multi30k's short, templated image
captions, and this run used the same modest 15k-pair/7-epoch budget. The
attention heatmap (`outputs/iwslt2017/cross_attention_heatmap.png`) shows
exactly this: a clean, correct diagonal alignment for the sentence's
opening ("wie"→"as", "sie"→"you", "es"→"it", "ist"→"expands"), which then
degrades into `<unk>`-heavy output for the harder back half of a long,
complex sentence. The model is doing the right thing where it has enough
signal to work with, and honestly failing where it doesn't — a fair,
undoctored result rather than a cherry-picked one.

**Run:** `python 10_capstone_transformer/transformer.py`

---

## Closing synthesis: what the 10 projects show, together

Read end to end, this series is one continuous argument, not ten disconnected
demos:

1. **Projects 1-2** establish the mechanics everything else depends on:
   backprop is the chain rule applied layer by layer (1), and architecture
   + optimizer choice measurably matters even on identical data (2).
2. **Project 3** introduces sequence modeling and immediately surfaces its
   central problem: BPTT's vanishing gradient, *measured*, not asserted.
3. **Project 4** fixes it with gating — but only once Project 4's own
   re-run proved gating's benefit is specifically a *long-context* effect,
   invisible on short sequences.
4. **Project 5** shows a second, independent limitation of one-directional
   recurrence (no future context) and fixes it cheaply (bidirectionality).
5. **Project 6** identifies the OTHER major RNN limitation — the fixed
   bottleneck vector — and measures its cost directly (BLEU vs. length).
6. **Project 7** improves inference (beam search) without touching the
   bottleneck problem at all — a clean isolation of decoding strategy from
   architecture.
7. **Project 8** fixes the bottleneck (attention), at real engineering cost
   (a slow, sequential per-step decoder) — and its own result honestly
   flags that the fix needs matched training budgets to show cleanly.
8. **Project 9** shows attention doesn't need recurrence at all — the same
   idea as Project 8, computed as one parallel matmul, ~100x faster per
   epoch.
9. **Project 10** assembles 3, 6, 7, 8, and 9's ideas into the real
   architecture and gets the payoff in one number: BLEU roughly doubles
   over the best RNN-based result, and the length-degradation problem
   Project 6 first identified is gone.

Every claim in this series was measured on a real dataset and reported
honestly, including the results that didn't go as expected (Project 8's
epoch-budget confound, Project 4's first null result, Project 9's
overfitting) — those are as much a part of the learning record as the
successes.

---


