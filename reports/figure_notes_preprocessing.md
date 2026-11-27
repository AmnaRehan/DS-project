# Figure notes — preprocessing

- **images/15_normalisation_effect.png** — Normalisation shrinks the average text from 19.2 to 8.7 tokens (54% fewer), keeping the content words and negations.
- **images/16_class_weights.png** — Weights range from 0.50 ('joy') to 4.71 ('surprise'): an error on the rarest class counts 9.5× more than on the most common one.
- **images/17_scaling_before_after.png** — Scaling brings every feature to mean≈0 / std≈1; log1p reduced the skewness of no feature (none exceeded |skew|>1).
- **images/18_tfidf_idf.png** — The TF-IDF matrix is 5,000 features wide and 99.8% sparse; the most common (least informative) term is 'feel'.
- **images/19_sequence_padding.png** — Sequences are padded/truncated to 52 tokens: 0.94% of train texts are truncated and 1.70% / 3.74% of train / test tokens map to <OOV>.
