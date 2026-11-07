# Emotion Classification from Text — Phase 1: Data Cleaning, EDA and Preprocessing

## Project description
This project classifies short English texts (tweets) into one of six emotions: **anger, fear, joy, love, sadness, surprise**. Phase 1 covers all the work on the data before modelling: cleaning, exploratory data analysis (EDA), insights and preprocessing/scaling. Phase 2 will build the ML, deep learning and transformer models on top of the files produced here.

## Problem statement
Given a short text, predict the emotion it expresses (single-label, 6-class text classification). In this phase we want to answer: is the data clean and trustworthy, how are the classes distributed, how long are the texts, which words belong to which emotion, and how should the text be converted to numbers without data leakage?

## Dataset
| | |
|---|---|
| **Name** | Emotions dataset for NLP (CARER emotion dataset, Saravia et al., 2018) |
| **Source** | Hugging Face: https://huggingface.co/datasets/dair-ai/emotion <br> Kaggle: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp |
| **Size** | 20,000 texts: 16,000 train / 2,000 validation / 2,000 test |
| **Format** | `text;label` (semicolon separated, no header row) |
| **Features** | `text` (the tweet). From it we created 5 numeric features: character count, word count, average word length, unique-word ratio and stop-word ratio |
| **Target variable** | `label`, one of the 6 emotions |

The raw files are small (about 2 MB), so they are included in `data/raw/`. The download links above can be used to get the original dataset.

## Repository structure
```
├── data/
│   ├── raw/                  original train, val and test files
│   └── processed/            cleaned and preprocessed data
├── images/                   all graphs saved as PNG
├── reports/                  cleaning log, EDA insights and figure notes
├── 01_data_cleaning_final.ipynb
├── 02_eda.ipynb
├── 03_preprocessing.ipynb
├── README.md
└── requirements.txt
```

## How to run
1. Install the libraries: `pip install -r requirements.txt`
2. Make sure `train`, `val` and `test` (`.csv`) are inside `data/raw/`.
3. Open the notebooks in Jupyter and run all cells **in this order**: `01_data_cleaning_final` → `02_eda` → `03_preprocessing`. Each notebook uses the files saved by the previous one.

The notebooks save the cleaned data and processed data in `data/processed/`, all graphs in `images/` and the written results in `reports/`.

## Summary of cleaning steps
The notebook `01_data_cleaning_final.ipynb` does the following (rows removed are counted at each step):

1. **Loading:** the files have no header row, so they are read with `header=None`. (Reading with pandas' default setting loses one row per file, which gave 15,999 / 1,999 / 1,999 rows in our earlier experiments.) The test file also contained a header line, which is detected and removed.
2. **Missing values:** checked for empty values and blank texts. None were found.
3. **Labels:** stripped and converted to lower case, and checked against the 6 valid emotions. All labels were valid.
4. **Formatting:** unicode normalisation, lower case, removal of HTML tags and URLs, and extra spaces. The raw text was already lower case, ASCII and without punctuation.
5. **Duplicates:** exact duplicates removed (keeping the first copy). Texts that appear with **different labels** are removed completely.
6. **Data leakage:** texts that appear in more than one split are removed from train and val, so the test set stays unchanged.
7. **Outliers:** very long texts are found with the IQR rule on the train set. They are real sentences, so they are **kept and only flagged** in an `is_length_outlier` column.

| Step | train | val | test |
|---|---|---|---|
| Raw rows | 16,000 | 2,000 | 2,000 |
| Missing values | 0 | 0 | 0 |
| Empty text | 0 | 0 | 0 |
| Invalid label | 0 | 0 | 0 |
| Exact duplicates removed | 1 | 0 | 0 |
| Conflicting-label texts removed | 60 | 4 | 0 |
| Leakage removed (overlap with test/val) | 16 | 3 | 0 |
| **Clean rows** | **15,923** | **1,993** | **2,000** |
| Length outliers (kept, flagged) | 375 | 43 | 49 |

Cleaning did not change the class distribution (the share of each class in train moved by less than 0.2 percentage points).

## Key findings
1. **Size and quality:** after cleaning there are 15,923 train, 1,993 val and 2,000 test texts, with no missing values, no duplicates and no overlap between the splits.
2. **Class imbalance:** joy is the largest class (33.5%) and surprise the smallest (3.5%), a ratio of about 9.5 : 1. Train, val and test have the same class mix (chi-square p = 0.365). Because of this, accuracy alone is not enough, so we will also use macro-F1, per-class recall and class weights.
3. **Label noise:** 60 train texts appeared with two different labels, mostly *joy / love* and *fear / surprise*. These emotions are close to each other, so no model can reach 100%.
4. **Text length:** texts are short: mean 19.2 words, median 17, min 2, max 66, skewed to the right (skewness 0.96). 95% of the texts have 41 words or less and 99% have 52 words or less.
5. **Length vs emotion:** love texts are the longest (20.8 words on average) and sadness the shortest (18.4). The Kruskal-Wallis test is significant (p = 3e-18), but the difference is only about 2 words, so length is a weak feature.
6. **Vocabulary:** 15,188 unique words in 305,403 tokens. 51% of the words appear only once. The top 10,000 words cover 98.3% of all tokens, so limiting the vocabulary loses very little.
7. **Template word "feel":** 99% of the texts contain feel, feeling or felt, so these words do not help to separate the emotions.
8. **Words per emotion:** the most frequent words (time, know, little) are the same for most emotions. The words that really separate the classes are emotion words found with the chi-square test, e.g. *terrified, shaken* (fear), *exhausted, miserable* (sadness), *irritable, offended* (anger), *amazed, impressed* (surprise), *caring, loving* (love), *successful, confident* (joy).
9. **Likely confusions:** sadness and anger share the most vocabulary (Jaccard 0.45); joy and surprise share the least (0.32).
10. **Correlations:** characters and words are almost the same feature (r = 0.98). The correlation between any numeric feature and any emotion is very low (strongest r = -0.05), so the information is in the words, not in the length statistics.
11. **Same distribution:** train, val and test have the same text length distribution (KS test p = 0.87 and 0.62).
12. **Remaining noise:** the most frequent bigram is "href http" (149 times), which shows that some texts still contain pieces of web links. They are a small part of the data and were left unchanged.

### What this means for Phase 2
- Use class weights and report macro-F1 and per-class results.
- Use TF-IDF with unigrams and bigrams for the ML models.
- Use a vocabulary of 10,000 words and a maximum length of 52 tokens for the deep learning models.
- Remove stop words for TF-IDF but keep negations (not, no, never), and keep the original text for the neural networks.

## Normalisation and scaling (notebook 03)
- **Text for ML:** lower case, letters only, stop words removed except negations. This reduces the average text from 19.2 to 8.7 tokens.
- **Labels:** encoded alphabetically (anger = 0, fear = 1, joy = 2, love = 3, sadness = 4, surprise = 5). Balanced class weights range from 0.50 (joy) to 4.71 (surprise).
- **Numeric features:** standardised with `StandardScaler`. The skewness of every feature was below 1, so `log1p` was not needed. Train has mean 0 and std 1 after scaling and val/test are very close to 0.
- **TF-IDF:** 5,000 features, unigrams and bigrams (99.8% sparse).
- **Sequences for deep learning:** vocabulary of 10,000 words (0 = padding, 1 = unknown), length 52. 0.94% of train texts are truncated and 3.74% of the test words are out of vocabulary.
- **No leakage:** the scaler, TF-IDF and vocabulary are fitted on train only. The notebook checks this (981 words that only occur in test are not in any vocabulary).

## Graphs and what they show
All graphs are saved in `images/` and are also shown in the notebooks.

| Graph | What it tells us |
|---|---|
| `01_format_issues.png` | Shows how many rows have formatting problems (uppercase, spaces, URLs, punctuation...). The raw text was already consistent, so almost nothing had to be fixed. |
| `02_length_outliers.png` | Boxplot and histogram of text length by split. Most texts are short with a long tail; 375 train texts are above the IQR upper fence and were flagged, not removed. |
| `03_class_share_before_after.png` | Class proportions before and after cleaning are almost identical, so cleaning did not bias the data. |
| `04_class_distribution.png` | Joy (33.5%) and sadness are the big classes, surprise (3.5%) is the smallest. Train/val/test have the same class mix. |
| `05_length_histograms.png` | Texts average 19.2 words (median 17, range 2–66) and are right-skewed. |
| `06_length_by_emotion.png` | Length differs slightly by emotion (love longest, sadness shortest), but the difference is small. |
| `07_length_ecdf.png` | 90% of texts have ≤ 35 words, 95% ≤ 41, 99% ≤ 52, which is used as the maximum sequence length. |
| `08_vocabulary_zipf_coverage.png` | Word frequency follows Zipf's law. The top 10,000 words cover 98.3% of the tokens. |
| `09_top_words_per_emotion.png` | Most frequent words per emotion. Many are generic (time, know) except for love and surprise, which have clear emotion words. |
| `10_top_bigrams.png` | Most frequent word pairs. Shows that some texts still contain link fragments ("href http"). |
| `11_vocabulary_overlap.png` | Sadness and anger share the most vocabulary (0.45), so they may be confused by the models. |
| `12_feel_template_share.png` | 99% of the texts contain "feel", so it is not useful for classification. |
| `13_correlation_heatmaps.png` | Characters and words are almost perfectly correlated (0.98). The correlation of the features with the emotions is very low (max −0.05). |
| `14_split_length_comparison.png` | Train, val and test have the same length distribution. |
| `15_normalisation_effect.png` | Text normalisation reduces the average text from 19.2 to 8.7 tokens. |
| `16_class_weights.png` | Class weights from 0.50 (joy) to 4.71 (surprise): errors on surprise count about 9.5 times more than on joy. |
| `17_scaling_before_after.png` | Features before and after standardisation: all have mean 0 and std 1 after scaling. |
| `18_tfidf_idf.png` | The TF-IDF matrix is 99.8% sparse and the most common term ("feel") gets the lowest weight. |
| `19_sequence_padding.png` | Number of real tokens per padded sequence (length 52) and how the padding looks. |

## Cleaned and processed data
Saved in `data/processed/`:
- `train_clean.csv`, `val_clean.csv`, `test_clean.csv`: cleaned data (text, label, is_length_outlier).
- `train_processed.csv`, `val_processed.csv`, `test_processed.csv`: text, normalised text, label, label id, raw and scaled numeric features.
- `tfidf_*.npz`, `tfidf_vectorizer.joblib`: TF-IDF matrices and the fitted vectoriser.
- `seq_*.npy`, `y_*.npy`, `vocab.json`: padded sequences, labels and vocabulary for the deep learning models.
- `numeric_scaler.joblib`, `label_mapping.json`: fitted scaler, label mapping and class weights.

All of these files can be generated again by running the three notebooks.

## Requirements
The libraries are listed in `requirements.txt` (Python 3.10 or newer is recommended).

## Reference
Saravia et al. (2018). *CARER: Contextualized Affect Representations for Emotion Recognition.* EMNLP.
