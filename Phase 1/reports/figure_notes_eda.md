# Figure notes — EDA

- **images/04_class_distribution.png** — 'joy' is the largest class (33.5%) and 'surprise' the smallest (3.5%): imbalance ratio 9.5:1. Class mix is consistent across splits (chi-square p = 0.365).
- **images/05_length_histograms.png** — Texts average 19.2 words (median 17, range 2–66); skewness = 0.96 (right-skewed).
- **images/06_length_by_emotion.png** — Longest on average: 'love' (20.8 words); shortest: 'sadness' (18.4). Kruskal-Wallis p = 3.01e-18 → length differs significantly between emotions.
- **images/07_length_ecdf.png** — 90% of texts have ≤ 35 words, 95% ≤ 41, 99% ≤ 52 → a maximum sequence length near 52 keeps ~99% of texts untruncated.
- **images/08_vocabulary_zipf_coverage.png** — Vocabulary = 15,188 unique words, 51% of them appear only once; the top 10,000 words already cover 98.30% of all tokens → capping the vocabulary at 5,000–10,000 loses almost nothing.
- **images/09_top_words_per_emotion.png** — Top-3 words per class: joy: time, know, make; sadness: know, time, little; anger: time, know, little; fear: little, know, people; love: love, know, loving; surprise: amazed, impressed, overwhelmed.
- **images/10_top_bigrams.png** — Most frequent bigram: 'href http' (149 occurrences); phrases carry extra context beyond single words, supporting n-gram TF-IDF features.
- **images/11_vocabulary_overlap.png** — Most similar vocabularies: 'sadness' & 'anger' (Jaccard 0.45) → expect confusion between them; least similar: 'joy' & 'surprise' (0.32).
- **images/12_feel_template_share.png** — 99% of train texts contain a form of 'feel', ranging from 96% to 100% across classes.
- **images/13_correlation_heatmaps.png** — Most correlated feature pair: char_count & word_count (r = 0.98); strongest feature-label correlation: char_count vs is_sadness (r = -0.05).
- **images/14_split_length_comparison.png** — Length distributions of train/val/test are statistically indistinguishable (KS p-values: train vs val=0.874, train vs test=0.623).
