# chinese-paralinguistic-features-DPLD
Digital Paralinguistic List &amp; Dictionary (DPLD) + code + data for "Integrating Paralinguistic Features with BERT and LDA for Chinese Implicit Aggression Detection" (AP Research 2026).

The folder contains **multiple copies of `DPLD.csv`** inside different subdirectories (`DPLD/`, `LLM annotation/`, and the root). This is intentional: each subdirectory is self‑contained for a specific processing step, so that every step’s input/output can be used independently without moving files. The **root `DPLD.csv`** is the final, complete lexicon used in the paper’s experiments. All other copies are identical and provided for convenience.


## What Is Included vs. What Is Not

| Component | Status | Notes |
|-----------|--------|-------|
| **DPLD lexicon** | ✅ Complete (147 entries) | Full list with primary/secondary functions, intensity (1‑5), trigger conditions, examples |
| **Pseudo‑labeled annotations** | ⚠️ Samples only | Only a representative subset of `intent`/`offensiveness` labels is provided. The full annotated dataset is too large to host. |
| **BERT embeddings** | ❌ Not included | BERT was **not fine‑tuned**. To obtain BERT vectors, run a standard sentence‑transformer (`shibing624/text2vec-base-chinese`) on the raw comments. |
| **LDA topic features** | ✅ 12‑dimensional | The LDA model (12 topics) was trained on the full corpus. The resulting topic distributions are **not** stored per comment due to size, but can be reproduced using the provided `topic_mapping_final.csv` and standard LDA training. |
| **Raw comment data** | ✅ Available (compressed) | `data/Bilibili.zip` contains the original anonymized comments. Extract and preprocess to rebuild full feature vectors. |

## How to Obtain the Complete Dataset (with BERT + LDA)

Due to GitHub’s file size limits, the **complete feature matrix** (BERT embeddings + LDA probabilities + DPLD triggers for all 110k comments) is **not** directly uploaded. To reconstruct it:

1. Download and extract `data/Bilibili.zip` to obtain the raw comment text.
2. Install required packages:
   ```bash
   pip install sentence-transformers scikit-learn pandas numpy

3. Generate BERT embeddings using shibing624/text2vec-base-chinese (no fine‑tuning needed).

4. Train an LDA model with n_components=12 on the tokenized text (use the provided topic_mapping_final.csv for topic interpretation).

5. Apply the DPLD trigger matching logic as implemented in DPLD/dpld_ollama.py or LLM annotation/ollama_annotation.py.

A 15k‑sample preview with LDA features (but no BERT) is available at data/dataset_15ksample_with_lda.csv.

## Reproducibility
Requirements
1. Python 3.10+
2. ollama (for local LLM inference, optional)
3. sentence-transformers, scikit-learn, pandas, numpy, tqdm

Minimal steps to reproduce the paper’s main classification
1. Prepare features
Run the feature extraction pipeline (see code in LLM annotation/ollama_annotation.py).
BERT and LDA need to be computed separately as described above.

2. Train baseline models
Use logistic regression with class‑weighted balancing (as in the paper).
Scripts are not provided in a single file, but the logic is fully described in the paper (Section 5).

3. Evaluate
Compute macro‑F1 on a hold‑out test set.

## License
Code is licensed under the MIT License (see LICENSE).
The DPLD lexicon and data are provided for academic research use only.

## Contact
Open an issue on GitHub for questions or bug reports.
