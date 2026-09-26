"""
Project 10 - Capstone: Full Transformer Encoder-Decoder for Translation

Brings together every mechanism built across this whole series into the
real architecture (Vaswani et al., 2017):

  Project 3 (RNN/BPTT)         -> replaced entirely: no recurrence anywhere
  Project 4 (LSTM/GRU gates)   -> replaced entirely: no gates, no cell state
  Project 6 (encoder-decoder)  -> same INPUT/OUTPUT framing, new internals
  Project 7 (beam search)      -> reused directly for inference decoding
  Project 8 (Bahdanau attn)    -> generalized: cross-attention is Project 8's
                                   idea, computed as a parallel matmul instead
                                   of a per-step MLP
  Project 9 (multi-head attn)  -> reused directly, PLUS a new piece: masked
                                   self-attention in the decoder (a token may
                                   only attend to itself and earlier tokens -
                                   without this, training would let the model
                                   "cheat" by looking at the answer)

Two training runs, same architecture and code:
  1. PRIMARY: IWSLT2017 English->German (real TED talk transcripts) - a
     larger, richer, more realistic dataset than Projects 6-8 used.
  2. SECONDARY: Multi30k, same 15k-pair subset and similar budget as
     Projects 6/7/8, so there is one direct, apples-to-apples BLEU
     comparison across the whole architectural progression (RNN bottleneck
     -> +beam search -> +RNN attention -> full Transformer).

RUN WITH:
    python transformer.py
"""

import os
import re
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

np.random.seed(42)
tf.random.set_seed(42)
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

MAX_LEN = 26
MAX_VOCAB = 8000
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 256
NUM_LAYERS = 2       # stacked encoder blocks AND stacked decoder blocks
BATCH_SIZE = 64


def tokenize(text):
    text = text.lower().strip()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-ZäöüÄÖÜß.,!?' ]+", " ", text)
    return text.split()


class Vocab:
    def __init__(self, tokenized_sentences, max_vocab):
        freq = {}
        for sent in tokenized_sentences:
            for tok in sent:
                freq[tok] = freq.get(tok, 0) + 1
        most_common = sorted(freq.items(), key=lambda x: -x[1])[:max_vocab - 4]
        self.itos = ['<pad>', '<sos>', '<eos>', '<unk>'] + [w for w, _ in most_common]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, tokens, max_len):
        ids = [self.stoi.get(t, self.stoi['<unk>']) for t in tokens]
        ids = [self.stoi['<sos>']] + ids[:max_len - 2] + [self.stoi['<eos>']]
        return ids

    def __len__(self):
        return len(self.itos)


def load_multi30k(num_examples):
    ds = load_dataset('bentrevett/multi30k')
    train = ds['train'].select(range(num_examples))
    val = ds['validation']
    src_train = [tokenize(x) for x in train['en']]
    tgt_train = [tokenize(x) for x in train['de']]
    src_val = [tokenize(x) for x in val['en']]
    tgt_val = [tokenize(x) for x in val['de']]
    return src_train, tgt_train, src_val, tgt_val


def load_iwslt(num_examples, num_val=1000):
    ds = load_dataset('iwslt2017', 'iwslt2017-en-de', trust_remote_code=True)
    train = ds['train'].select(range(num_examples))
    val = ds['validation'].select(range(min(num_val, len(ds['validation']))))

    def extract(split):
        src, tgt = [], []
        for row in split['translation']:
            s, t = tokenize(row['en']), tokenize(row['de'])
            if 1 <= len(s) <= MAX_LEN - 2 and 1 <= len(t) <= MAX_LEN - 2:
                src.append(s)
                tgt.append(t)
        return src, tgt

    src_train, tgt_train = extract(train)
    src_val, tgt_val = extract(val)
    return src_train, tgt_train, src_val, tgt_val


def build_vocabs_and_encode(src_train, tgt_train, src_val, tgt_val):
    src_vocab = Vocab(src_train, MAX_VOCAB)
    tgt_vocab = Vocab(tgt_train, MAX_VOCAB)
    print(f"  Source vocab: {len(src_vocab)}, target vocab: {len(tgt_vocab)}")

    def encode(src_list, tgt_list):
        src_ids = pad_sequences([src_vocab.encode(s, MAX_LEN) for s in src_list], maxlen=MAX_LEN, padding='post', value=0)
        tgt_ids = pad_sequences([tgt_vocab.encode(t, MAX_LEN) for t in tgt_list], maxlen=MAX_LEN, padding='post', value=0)
        return src_ids, tgt_ids

    X_train, y_train = encode(src_train, tgt_train)
    X_val, y_val = encode(src_val, tgt_val)
    print(f"  Train pairs: {len(X_train)}, val pairs: {len(X_val)}")
    return (X_train, y_train), (X_val, y_val), src_vocab, tgt_vocab


class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        positions = np.arange(max_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(d_model))
        angles = positions * angle_rates
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pos_encoding = tf.constant(angles[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = models.Sequential([layers.Dense(ff_dim, activation='relu'), layers.Dense(d_model)])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

    def call(self, x):
        attn_out = self.attention(x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class TransformerDecoderBlock(layers.Layer):
    """
    Two attention sub-layers per block, unlike the encoder's one:
      1. MASKED self-attention over the target sequence so far - the causal
         mask blocks each position from attending to any FUTURE position.
         Without this, the model could trivially "cheat" during training by
         reading the answer directly off the teacher-forcing input.
      2. CROSS-attention: query = decoder's own representation, key/value =
         the ENCODER's output. This is a direct generalization of Project
         8's Bahdanau attention - "which source words matter for this
         decoder position" - computed as one parallel matmul instead of a
         small MLP re-run at every sequential decoder step.
    """
    def __init__(self, d_model, num_heads, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.self_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.cross_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = models.Sequential([layers.Dense(ff_dim, activation='relu'), layers.Dense(d_model)])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()

    def call(self, x, encoder_output, causal_mask, return_attention_scores=False):
        self_out, self_scores = self.self_attention(x, x, attention_mask=causal_mask, return_attention_scores=True)
        x = self.norm1(x + self_out)
        cross_out, cross_scores = self.cross_attention(x, encoder_output, return_attention_scores=True)
        x = self.norm2(x + cross_out)
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        if return_attention_scores:
            return x, self_scores, cross_scores
        return x


def get_causal_mask(seq_len):
    """Lower-triangular boolean mask, shape (1, seq_len, seq_len): position i
    may attend to position j only if j <= i. Broadcasts across the batch."""
    mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return tf.cast(mask, tf.bool)[tf.newaxis, ...]


class TranslationTransformer:
    """Bundles everything needed for BOTH training (one parallel forward
    pass, teacher forcing) and inference (autoregressive, one token at a
    time, re-running the decoder stack on the growing prefix each step -
    no KV-caching, which is a standard optimization skipped here to keep
    the code readable)."""

    def __init__(self, src_vocab_size, tgt_vocab_size):
        self.src_embedding = layers.Embedding(src_vocab_size, EMBED_DIM)
        self.tgt_embedding = layers.Embedding(tgt_vocab_size, EMBED_DIM)
        self.pos_encoding = PositionalEncoding(MAX_LEN, EMBED_DIM)
        self.encoder_blocks = [TransformerEncoderBlock(EMBED_DIM, NUM_HEADS, FF_DIM) for _ in range(NUM_LAYERS)]
        self.decoder_blocks = [TransformerDecoderBlock(EMBED_DIM, NUM_HEADS, FF_DIM) for _ in range(NUM_LAYERS)]
        self.output_dense = layers.Dense(tgt_vocab_size, activation='softmax')
        self.causal_mask = get_causal_mask(MAX_LEN)

    def encode(self, encoder_input):
        x = self.pos_encoding(self.src_embedding(encoder_input))
        for block in self.encoder_blocks:
            x = block(x)
        return x

    def decode(self, decoder_input, encoder_output, return_attention_scores=False):
        y = self.pos_encoding(self.tgt_embedding(decoder_input))
        last_cross_scores = None
        for block in self.decoder_blocks:
            if return_attention_scores:
                y, _, last_cross_scores = block(y, encoder_output, self.causal_mask, return_attention_scores=True)
            else:
                y = block(y, encoder_output, self.causal_mask)
        preds = self.output_dense(y)
        if return_attention_scores:
            return preds, last_cross_scores
        return preds

    def build_train_model(self):
        encoder_input = layers.Input(shape=(MAX_LEN,), name='encoder_input')
        decoder_input = layers.Input(shape=(MAX_LEN,), name='decoder_input')
        encoder_output = self.encode(encoder_input)
        preds = self.decode(decoder_input, encoder_output)
        model = models.Model([encoder_input, decoder_input], preds)
        model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),
                       loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def build_encoder_model(self):
        encoder_input = layers.Input(shape=(MAX_LEN,))
        return models.Model(encoder_input, self.encode(encoder_input))


def greedy_decode(src_ids, encoder_model, transformer, tgt_vocab):
    enc_out = encoder_model.predict(src_ids, verbose=0)
    dec_input = np.zeros((1, MAX_LEN), dtype=np.int32)
    dec_input[0, 0] = tgt_vocab.stoi['<sos>']
    output_ids = []
    for t in range(1, MAX_LEN):
        preds = transformer.decode(tf.constant(dec_input), tf.constant(enc_out))
        next_id = int(np.argmax(preds.numpy()[0, t - 1, :]))
        if next_id == tgt_vocab.stoi['<eos>']:
            break
        output_ids.append(next_id)
        dec_input[0, t] = next_id
    return output_ids


def beam_search_decode(src_ids, encoder_model, transformer, tgt_vocab, beam_width=3):
    """Same length-normalized log-probability beam search as Project 7,
    adapted to a Transformer decoder that recomputes over the full prefix
    each step instead of carrying an RNN hidden state forward."""
    enc_out = encoder_model.predict(src_ids, verbose=0)
    sos_id, eos_id = tgt_vocab.stoi['<sos>'], tgt_vocab.stoi['<eos>']
    beams = [([sos_id], 0.0, False)]

    for t in range(1, MAX_LEN):
        all_candidates = []
        for tokens, log_prob, finished in beams:
            if finished:
                all_candidates.append((tokens, log_prob, True))
                continue
            dec_input = np.zeros((1, MAX_LEN), dtype=np.int32)
            dec_input[0, :len(tokens)] = tokens
            preds = transformer.decode(tf.constant(dec_input), tf.constant(enc_out))
            probs = preds.numpy()[0, len(tokens) - 1, :]
            top_ids = np.argsort(probs)[-beam_width:]
            for tid in top_ids:
                p = max(probs[tid], 1e-12)
                new_tokens = tokens + [int(tid)]
                new_log_prob = log_prob + np.log(p)
                all_candidates.append((new_tokens, new_log_prob, tid == eos_id))
        all_candidates.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
        beams = all_candidates[:beam_width]
        if all(b[2] for b in beams):
            break

    best_tokens = beams[0][0]
    return [t for t in best_tokens if t not in (sos_id, eos_id)]


def run_experiment(name, src_train, tgt_train, src_val, tgt_val, epochs, n_bleu_eval):
    print(f"\n{'='*70}\nEXPERIMENT: {name}\n{'='*70}")
    out_dir = os.path.join(BASE_OUTPUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    (X_train, y_train), (X_val, y_val), src_vocab, tgt_vocab = build_vocabs_and_encode(
        src_train, tgt_train, src_val, tgt_val)

    decoder_input_train = np.zeros_like(y_train)
    decoder_input_train[:, 1:] = y_train[:, :-1]
    decoder_input_train[:, 0] = tgt_vocab.stoi['<sos>']
    decoder_input_val = np.zeros_like(y_val)
    decoder_input_val[:, 1:] = y_val[:, :-1]
    decoder_input_val[:, 0] = tgt_vocab.stoi['<sos>']

    transformer = TranslationTransformer(len(src_vocab), len(tgt_vocab))
    train_model = transformer.build_train_model()
    encoder_model = transformer.build_encoder_model()
    train_model.summary()

    ckpt_path = os.path.join(out_dir, 'checkpoint.weights.h5')
    initial_epoch = 0
    if os.path.exists(ckpt_path):
        print(f"Found checkpoint - resuming.")
        train_model.load_weights(ckpt_path)
        if os.path.exists(ckpt_path + '.epoch'):
            initial_epoch = int(open(ckpt_path + '.epoch').read().strip())

    class SaveEveryEpoch(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            self.model.save_weights(ckpt_path)
            with open(ckpt_path + '.epoch', 'w') as f:
                f.write(str(epoch + 1))

    history = train_model.fit(
        [X_train, decoder_input_train], np.expand_dims(y_train, -1),
        validation_data=([X_val, decoder_input_val], np.expand_dims(y_val, -1)),
        epochs=epochs, initial_epoch=initial_epoch, batch_size=BATCH_SIZE, verbose=2,
        callbacks=[SaveEveryEpoch()],
    )

    fig1, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history.history['loss'], label='train'); axes[0].plot(history.history['val_loss'], label='val')
    axes[0].set_title(f'{name}: Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history.history['accuracy'], label='train'); axes[1].plot(history.history['val_accuracy'], label='val')
    axes[1].set_title(f'{name}: Accuracy'); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, 'training_curves.png'), dpi=150)

    print(f"\nEvaluating (greedy + beam-3) on {n_bleu_eval} validation sentences...")
    smoother = SmoothingFunction().method1
    rng = np.random.RandomState(42)
    val_pairs = list(zip(src_val, tgt_val))
    sample = [val_pairs[i] for i in rng.choice(len(val_pairs), size=min(n_bleu_eval, len(val_pairs)), replace=False)]

    bleu_greedy, bleu_beam, src_lens = [], [], []
    examples = []
    for src_tokens, tgt_tokens in sample:
        src_ids = pad_sequences([src_vocab.encode(src_tokens, MAX_LEN)], maxlen=MAX_LEN, padding='post', value=0)
        g_ids = greedy_decode(src_ids, encoder_model, transformer, tgt_vocab)
        b_ids = beam_search_decode(src_ids, encoder_model, transformer, tgt_vocab, beam_width=3)
        g_words = [tgt_vocab.itos[i] for i in g_ids]
        b_words = [tgt_vocab.itos[i] for i in b_ids]
        bleu_greedy.append(sentence_bleu([tgt_tokens], g_words, smoothing_function=smoother))
        bleu_beam.append(sentence_bleu([tgt_tokens], b_words, smoothing_function=smoother))
        src_lens.append(len(src_tokens))
        if len(examples) < 6:
            examples.append((src_tokens, tgt_tokens, g_words, b_words))

    bleu_greedy, bleu_beam, src_lens = np.array(bleu_greedy), np.array(bleu_beam), np.array(src_lens)
    print(f"Mean BLEU - greedy: {bleu_greedy.mean():.4f}, beam-3: {bleu_beam.mean():.4f}")

    bins = [0, 5, 8, 11, 14, 100]
    bin_labels = ['1-5', '6-8', '9-11', '12-14', '15+']
    bucket_means = []
    for label, lo, hi in zip(bin_labels, bins[:-1], bins[1:]):
        mask = (src_lens > lo) & (src_lens <= hi)
        bucket_means.append(float(bleu_beam[mask].mean()) if mask.sum() > 0 else None)
        print(f"  length {label}: mean BLEU (beam) = {bucket_means[-1]}  (n={mask.sum()})")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    plot_labels = [l for l, m in zip(bin_labels, bucket_means) if m is not None]
    plot_means = [m for m in bucket_means if m is not None]
    ax2.bar(plot_labels, plot_means, color='#9467bd')
    ax2.set_xlabel('Source sentence length (tokens)')
    ax2.set_ylabel('Mean BLEU (beam-3)')
    ax2.set_title(f'{name}: Transformer BLEU vs. sentence length')
    ax2.grid(alpha=0.3, axis='y')
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, 'bleu_vs_length.png'), dpi=150)

    with open(os.path.join(out_dir, 'examples.txt'), 'w', encoding='utf-8') as f:
        for src, tgt, g, b in examples:
            block = (f"EN:   {' '.join(src)}\nDE true:  {' '.join(tgt)}\n"
                      f"DE greedy: {' '.join(g)}\nDE beam-3: {' '.join(b)}\n\n")
            f.write(block)
            print(block)

    results = dict(
        name=name, mean_bleu_greedy=float(bleu_greedy.mean()), mean_bleu_beam=float(bleu_beam.mean()),
        bucket_labels=bin_labels, bucket_means=bucket_means,
        final_val_accuracy=float(history.history['val_accuracy'][-1]),
        final_val_loss=float(history.history['val_loss'][-1]),
        params=int(train_model.count_params()),
    )
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved all outputs to {out_dir}")
    return results, transformer, encoder_model, tgt_vocab, src_vocab, val_pairs


def main():
    print("Loading IWSLT2017 (English -> German, TED talk transcripts)...")
    # NOTE: this 2-encoder-block + 2-decoder-block model (self-attn + cross-attn
    # + FFN per decoder block, over a 26-length sequence, 8000-word vocab) is
    # far heavier per training step than Project 9's single classifier block.
    # 40k examples / 12 epochs at this model's measured ~1.3s/step would run
    # 3+ hours - scaled down to keep total wall-clock reasonable, matching the
    # same lesson learned (and fixed) in Project 8.
    iw_src_train, iw_tgt_train, iw_src_val, iw_tgt_val = load_iwslt(num_examples=15000)
    results_iwslt, transformer, encoder_model, tgt_vocab, src_vocab, val_pairs = run_experiment(
        'iwslt2017', iw_src_train, iw_tgt_train, iw_src_val, iw_tgt_val, epochs=7, n_bleu_eval=250)

    # --- Attention visualization: cross-attention on a real IWSLT sentence ---
    sample_src, sample_tgt = val_pairs[5]
    src_ids = pad_sequences([src_vocab.encode(sample_src, MAX_LEN)], maxlen=MAX_LEN, padding='post', value=0)
    enc_out = encoder_model.predict(src_ids, verbose=0)
    dec_input = np.zeros((1, MAX_LEN), dtype=np.int32)
    dec_input[0, 0] = tgt_vocab.stoi['<sos>']
    generated = []
    for t in range(1, MAX_LEN):
        preds, cross_scores = transformer.decode(tf.constant(dec_input), tf.constant(enc_out), return_attention_scores=True)
        next_id = int(np.argmax(preds.numpy()[0, t - 1, :]))
        if next_id == tgt_vocab.stoi['<eos>']:
            break
        generated.append(next_id)
        dec_input[0, t] = next_id
    gen_words = [tgt_vocab.itos[i] for i in generated]

    if len(gen_words) >= 2:
        cross_scores_np = cross_scores.numpy()[0].mean(axis=0)  # average over heads: (tgt_len, src_len)
        src_len_actual = len(sample_src) + 2
        gen_len_actual = len(gen_words)
        fig3, ax3 = plt.subplots(figsize=(1 + 0.6 * src_len_actual, 1 + 0.6 * gen_len_actual))
        im = ax3.imshow(cross_scores_np[:gen_len_actual, :src_len_actual], cmap='viridis', aspect='auto')
        ax3.set_xticks(range(src_len_actual))
        ax3.set_xticklabels(['<sos>'] + sample_src + ['<eos>'], rotation=90)
        ax3.set_yticks(range(gen_len_actual))
        ax3.set_yticklabels(gen_words)
        ax3.set_xlabel('Source (English)')
        ax3.set_ylabel('Generated (German)')
        ax3.set_title('Transformer cross-attention (averaged over heads)')
        fig3.colorbar(im, ax=ax3)
        fig3.tight_layout()
        fig3.savefig(os.path.join(BASE_OUTPUT_DIR, 'iwslt2017', 'cross_attention_heatmap.png'), dpi=150)
        print("Saved: cross_attention_heatmap.png")

    print("\nLoading Multi30k (secondary run - same setup as Projects 6/7/8, for direct comparison)...")
    m30_src_train, m30_tgt_train, m30_src_val, m30_tgt_val = load_multi30k(num_examples=15000)
    results_multi30k, *_ = run_experiment(
        'multi30k', m30_src_train, m30_tgt_train, m30_src_val, m30_tgt_val, epochs=7, n_bleu_eval=250)

    with open(os.path.join(BASE_OUTPUT_DIR, 'comparison_summary.json'), 'w') as f:
        json.dump({'iwslt2017': results_iwslt, 'multi30k': results_multi30k}, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"IWSLT2017  - mean BLEU (beam-3): {results_iwslt['mean_bleu_beam']:.4f}, "
          f"val acc: {results_iwslt['final_val_accuracy']:.4f}, params: {results_iwslt['params']:,}")
    print(f"Multi30k   - mean BLEU (beam-3): {results_multi30k['mean_bleu_beam']:.4f}, "
          f"val acc: {results_multi30k['final_val_accuracy']:.4f}, params: {results_multi30k['params']:,}")
    print("\nCompare Multi30k number above against Projects 6/7/8's README figures "
          "for the full RNN-bottleneck -> beam-search -> attention -> Transformer progression.")


if __name__ == "__main__":
    main()
