# chinese-paralinguistic-features-DPLD
Digital Paralinguistic List &amp; Dictionary (DPLD) + code + data for "Integrating Paralinguistic Features with BERT and LDA for Chinese Implicit Aggression Detection" (AP Research 2026).

## Key Resources

| File | Description | Use in Paper |
|------|-------------|---------------|
| `DPLD/DPLD.csv` | Full DPLD lexicon (147 terms) with primary/secondary function, intensity (1–5), trigger conditions, and example sentences. | Section 3.3: DPLD construction |
| `LLM annotation/fewshot_high_value.json` | 5,000 pseudo‑labeled comments (intent + offensiveness) used as few‑shot demonstrations. | Section 4.2: LLM‑assisted annotation |
| `data/dataset_15ksample_with_lda.csv` | 15,000 annotated samples with BERT embeddings (768d), LDA topic probabilities (12d), and DPLD feature counts. | Section 5: Model training & evaluation |
| `LDA/topic_mapping_final.csv` | Top‑20 keywords per LDA topic, manual topic labels, and risk level (H/M/L). | Section 3.2: LDA topic modeling |

## Reproducibility

### Requirements
- Python 3.10+
- `ollama` (for local LLM inference)
- `sentence-transformers`, `scikit-learn`, `pandas`, `numpy`, `tqdm`

### Steps to reproduce

1. **Set up environment**  
   ```bash
   pip install -r requirements.txt   # see below for dependencies
