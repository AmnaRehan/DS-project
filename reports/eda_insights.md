## Key findings from EDA

1. **Size & quality** – after cleaning: 15,923 train / 1,993 val / 2,000 test texts; no missing values, no duplicates, no train/val/test overlap.
2. **Class imbalance** – 'joy' is the largest class (33.5%), 'surprise' the smallest (3.5%); imbalance ratio 9.5:1. The class mix is the same across splits (chi-square p = 0.365).
3. **Text length** – mean 19.2 words (median 17, min 2, max 66), skewness 0.96. 95% of texts have ≤ 41 words and 99% ≤ 52.
4. **Length vs emotion** – mean words per class: joy 19.5, sadness 18.4, anger 19.2, fear 18.9, love 20.8, surprise 20.1 (Kruskal-Wallis p = 3.01e-18).
5. **Vocabulary** – 15,188 unique words in 305,403 tokens; 51% of the vocabulary occurs only once; the top 10,000 words cover 98.30% of tokens.
6. **Template words** – 99% of texts contain a form of 'feel', so these words are shared by (almost) every class and carry little class information.
7. **Likely confusions** – most overlapping vocabularies: 'sadness' & 'anger' (Jaccard 0.45).
8. **Correlations** – strongest feature pair: char_count & word_count (r = 0.98); strongest feature–class correlation r = -0.05 (char_count vs is_sadness) → weak, so the predictive signal is mainly in the words, not in length statistics.

## Implications for modelling (Phase 2)

- **Imbalance** → use stratified splits (already provided), class weights, and report macro-F1 and per-class recall, not accuracy alone.
- **Sequence length** → pad/truncate neural-network inputs to about the 99th percentile (52 words), not to the maximum.
- **Vocabulary** → cap the vocabulary (5k–10k words) with an out-of-vocabulary token; rare words add noise.
- **Features** → TF-IDF (uni+bigrams) is the natural baseline; numeric length features are redundant with each other and only weakly informative.
- **Scaling** → counts are right-skewed, so apply log1p before standardising; fit all scalers/vectorizers on **train only** to avoid leakage.