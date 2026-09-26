"""
Exports the actual dataset each project used into that project's own
data/ folder, as plain CSV/text files - so the GitHub repo is self-contained
and reviewable without running any code or needing internet access.

This does NOT change any project's script (they still auto-download via
`datasets`/Keras on first run, which is standard practice) - it just also
writes out a durable, human-readable copy of exactly what was used.

RUN WITH:
    python export_data_samples.py
"""
import os
import csv
import numpy as np
from datasets import load_dataset

BASE = os.path.dirname(__file__)


def project_dir(name):
    for d in os.listdir(BASE):
        if d.startswith(name):
            path = os.path.join(BASE, d, 'data')
            os.makedirs(path, exist_ok=True)
            return path
    raise FileNotFoundError(name)


def write_csv(path, header, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


# ---------- Project 1: make_moons (synthetic, but let's freeze the exact draw used) ----------
def export_project1():
    print("Project 1: make_moons")
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=400, noise=0.2, random_state=42)
    out = project_dir('01_')
    write_csv(os.path.join(out, 'moons.csv'), ['x1', 'x2', 'label'],
              [[x[0], x[1], int(lab)] for x, lab in zip(X, y)])


# ---------- Project 2: Fashion-MNIST (image dataset - export a labeled sample, not the full 12000x784 blob) ----------
def export_project2():
    print("Project 2: Fashion-MNIST sample")
    from tensorflow.keras.datasets import fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    out = project_dir('02_')
    os.makedirs(os.path.join(out, 'sample_images'), exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    for i in range(40):
        fname = f'train_{i:03d}.png'
        plt.imsave(os.path.join(out, 'sample_images', fname), X_train[i], cmap='gray')
        rows.append([fname, int(y_train[i])])
    write_csv(os.path.join(out, 'sample_labels.csv'), ['filename', 'label'], rows)
    with open(os.path.join(out, 'README.txt'), 'w') as f:
        f.write("Fashion-MNIST is a 28x28 grayscale image dataset (60,000 train / 10,000 test), "
                "too large to store as raw pixels here. This folder contains 40 real sample images "
                "(sample_images/) with their true labels (sample_labels.csv) as a concrete illustration. "
                "The full dataset auto-downloads via tensorflow.keras.datasets.fashion_mnist on first run "
                "(cnn_optimizers.py used the first 12,000 train / 2,000 test images).")


# ---------- Project 3 & 4: tiny_shakespeare slice (exact slice used) ----------
def export_shakespeare(project_prefix, n_chars):
    print(f"{project_prefix}: tiny_shakespeare slice ({n_chars} chars)")
    ds = load_dataset('tiny_shakespeare', trust_remote_code=True)
    text = ds['train']['text'][0][:n_chars]
    out = project_dir(project_prefix)
    with open(os.path.join(out, 'shakespeare_slice.txt'), 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"  wrote {os.path.join(out, 'shakespeare_slice.txt')} ({len(text)} chars)")


# ---------- Project 5: CoNLL-2003 NER ----------
def export_project5():
    print("Project 5: CoNLL-2003")
    ds = load_dataset('conll2003', trust_remote_code=True)
    tag_names = ds['train'].features['ner_tags'].feature.names
    out = project_dir('05_')

    def dump(split, fname):
        rows = []
        for sent_id, ex in enumerate(ds[split]):
            for tok, tag_id in zip(ex['tokens'], ex['ner_tags']):
                rows.append([sent_id, tok, tag_names[tag_id]])
        write_csv(os.path.join(out, fname), ['sentence_id', 'token', 'ner_tag'], rows)

    dump('train', 'train.csv')
    dump('validation', 'validation.csv')


# ---------- Projects 6, 7, 8: Multi30k (same 15k-pair subset) ----------
def export_multi30k(project_prefix, num_examples=15000):
    print(f"{project_prefix}: Multi30k ({num_examples} pairs)")
    ds = load_dataset('bentrevett/multi30k')
    train = ds['train'].select(range(num_examples))
    val = ds['validation']
    out = project_dir(project_prefix)
    write_csv(os.path.join(out, 'train_en_de.csv'), ['en', 'de'], list(zip(train['en'], train['de'])))
    write_csv(os.path.join(out, 'validation_en_de.csv'), ['en', 'de'], list(zip(val['en'], val['de'])))


# ---------- Project 9: rotten_tomatoes ----------
def export_project9():
    print("Project 9: rotten_tomatoes")
    ds = load_dataset('rotten_tomatoes')
    out = project_dir('09_')
    write_csv(os.path.join(out, 'train.csv'), ['text', 'label'], list(zip(ds['train']['text'], ds['train']['label'])))
    write_csv(os.path.join(out, 'validation.csv'), ['text', 'label'], list(zip(ds['validation']['text'], ds['validation']['label'])))


# ---------- Project 10: IWSLT2017 (15k pairs used) + Multi30k (secondary run) ----------
def export_project10():
    print("Project 10: IWSLT2017 + Multi30k")
    import re
    def tokenize(text):
        text = text.lower().strip()
        text = re.sub(r"([.,!?])", r" \1 ", text)
        text = re.sub(r"[^a-zA-ZäöüÄÖÜß.,!?' ]+", " ", text)
        return text.split()

    ds = load_dataset('iwslt2017', 'iwslt2017-en-de', trust_remote_code=True)
    train = ds['train'].select(range(15000))
    val = ds['validation'].select(range(min(1000, len(ds['validation']))))
    out = project_dir('10_')

    def dump(split, fname, max_len=24):
        rows = []
        for row in split['translation']:
            s, t = tokenize(row['en']), tokenize(row['de'])
            if 1 <= len(s) <= max_len and 1 <= len(t) <= max_len:
                rows.append([row['en'], row['de']])
        write_csv(os.path.join(out, fname), ['en', 'de'], rows)

    dump(train, 'iwslt2017_train_en_de.csv')
    dump(val, 'iwslt2017_validation_en_de.csv')

    export_multi30k_into(out)


def export_multi30k_into(out_dir, num_examples=15000):
    ds = load_dataset('bentrevett/multi30k')
    train = ds['train'].select(range(num_examples))
    val = ds['validation']
    write_csv(os.path.join(out_dir, 'multi30k_train_en_de.csv'), ['en', 'de'], list(zip(train['en'], train['de'])))
    write_csv(os.path.join(out_dir, 'multi30k_validation_en_de.csv'), ['en', 'de'], list(zip(val['en'], val['de'])))


if __name__ == "__main__":
    export_project1()
    export_project2()
    export_shakespeare('03_', 20000)
    export_shakespeare('04_', 20000)
    export_project5()
    export_multi30k('06_', 15000)
    export_multi30k('07_', 15000)
    export_multi30k('08_', 15000)
    export_project9()
    export_project10()
    print("\nAll data exports complete.")
