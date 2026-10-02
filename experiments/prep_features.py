"""Precompute MFCC (DeepSpeech-1) and 80-bin log-mel (Transformer-CTC) features for a LibriSpeech split.

Usage: python experiments/prep_features.py <librispeech_root> <split> <out_dir>
Writes <out_dir>/<split>.{mfcc,fbank}.npy (frames concatenated) and <out_dir>/<split>.index.pt.
"""
import os, sys
from multiprocessing import Pool
import numpy as np
import torch
sys.path.insert(0, ".")
from unsplice.data import list_librispeech, load_utterance, mfcc, fbank


def work(item):
    uid, path, text = item
    audio = load_utterance(path)
    return uid, text, len(audio), mfcc(audio), fbank(audio).astype(np.float16)


if __name__ == "__main__":
    root, split, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    items = list_librispeech(root, split)
    torch.set_num_threads(1)
    with Pool(int(os.environ.get('PREP_PROCS', 40))) as pool:
        res = pool.map(work, items, chunksize=16)
    index, m_off, f_off = [], 0, 0
    for uid, text, n, m, f in res:
        index.append({"id": uid, "text": text, "samples": n, "mfcc": (m_off, len(m)), "fbank": (f_off, len(f))})
        m_off += len(m)
        f_off += len(f)
    np.save(os.path.join(out, f"{split}.mfcc.npy"), np.concatenate([r[3] for r in res]))
    np.save(os.path.join(out, f"{split}.fbank.npy"), np.concatenate([r[4] for r in res]))
    torch.save(index, os.path.join(out, f"{split}.index.pt"))
    print(split, len(index), "utterances", m_off, "mfcc frames", f_off, "fbank frames")
