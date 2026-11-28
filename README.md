# 🎙️ VoxAid: Speech Emotion Support Assistant

**Live website:** 

VoxAid listens to what you say (or reads what you type), detects the emotion in your words, shows it, and replies with one or two comforting sentences that it can also speak aloud.

```
voice ──► speech-to-text ──► text cleaning ──► emotion model ──► comforting reply ──► text-to-speech
          (Google Speech)    (Phase 1 rules)                     + emotion & confidence     (gTTS)
```

The original proposal planned for an LLM to write the replies. The current app uses fixed, emotion-specific reply templates.

## Problem statement
Given a short English text/audio, predict which of six emotions it expresses: **anger, fear, joy, love, sadness or surprise** (single-label, 6-class text classification).

## Dataset
| | |
|---|---|
| **Name** | CARER emotion dataset (Saravia et al., 2018) |
| **Sources** | [Hugging Face](https://huggingface.co/datasets/dair-ai/emotion), [Kaggle](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) |
| **Raw size** | 20,000 texts: 16,000 train / 2,000 validation / 2,000 test |
| **After cleaning** | 15,923 train / 1,993 validation / 2,000 test |

| joy | sadness | anger | fear | love | surprise |
|---|---|---|---|---|---|
| 33.5% | 29.3% | 13.5% | 12.1% | 8.1% | 3.5% |


## Repository structure
```
├── Phase 1/     data cleaning, EDA and preprocessing
├── Phase 2/     modelling, evaluation and web app
│   ├── app/         app.py, voxaid_core.py, requirements.txt
│   ├── notebooks/   ML, DL, transformer, model comparison
│   ├── models/      ml/, dl/, transformer/
│   ├── results/     metrics (JSON) and best_model.json
│   ├── plots/
│   └── requirements.txt
└── README.md
```

---

## Phase 1: Data cleaning, EDA and preprocessing
Full details: [Phase 1 folder](Phase%201/).

Run the notebooks in order: `01_data_cleaning_final` → `02_eda` → `03_preprocessing`.

### Cleaning
- Checked for missing values, blank texts and invalid labels (none found).
- Normalised text (unicode, lower case, HTML tags, URLs, extra spaces).
- Removed exact duplicates and texts with conflicting labels.
- Removed texts shared between splits from train and validation, leaving test unchanged.
- Flagged long texts (IQR rule) but kept them.

| Step | train | val | test |
|---|---|---|---|
| Raw rows | 16,000 | 2,000 | 2,000 |
| Exact duplicates removed | 1 | 0 | 0 |
| Conflicting-label texts removed | 60 | 4 | 0 |
| Leakage removed | 16 | 3 | 0 |
| **Clean rows** | **15,923** | **1,993** | **2,000** |

### Key findings
- **Imbalance:** joy to surprise is about 9.5 : 1, so macro-F1 and class weights are used.
- **Label noise:** conflicting labels were mostly *joy / love* and *fear / surprise*.
- **Length:** mean 19.2 words, median 17; 99% of texts have 52 words or fewer.
- **Vocabulary:** 15,188 unique words; the top 10,000 cover 98.3% of tokens.
- **"Feel":** 99% of texts contain feel, feeling or felt, so these do not separate emotions.
- **Overlap:** sadness and anger share the most vocabulary (Jaccard 0.45).
- **Numeric features:** very weakly correlated with emotion (strongest r = -0.05).
- **Remaining noise:** some link fragments such as "href http" are left in the text.

### Preprocessing
- **ML text:** letters only, stop words removed but negations kept.
- **TF-IDF:** 5,000 unigram and bigram features.
- **Sequences:** 10,000-word vocabulary, length 52.
- **Labels:** anger = 0, fear = 1, joy = 2, love = 3, sadness = 4, surprise = 5.
- **Class weights:** from 0.50 (joy) to 4.71 (surprise).
- **No leakage:** scaler, TF-IDF and vocabulary are fitted on train only.

---

## Phase 2: Modelling and web app

| Notebook | Models | Input |
|---|---|---|
| `classification_ML.ipynb` | LR baseline, Naive Bayes, Logistic Regression, Linear SVM, Random Forest | TF-IDF |
| `classification_DL.ipynb` | Feedforward NN, Bidirectional LSTM | Padded sequences |
| `transformer.ipynb` | Fine-tuned DistilBERT | Clean text, own tokenizer |
| `model_comparison.ipynb` | Compares models and picks the app's default | Saved results |

- ML models are tuned with stratified cross-validation on train + validation (5-fold; 3-fold for Random Forest).
- DL and transformer models use the validation set for early stopping or best-epoch selection.
- The 2,000-row test set is held out.
- Class weights are used in every model except the baseline.

### Results (test set)
| Model | Family | Accuracy | Macro-F1 |
|---|---|---|---|
| **DistilBERT** | Transformer | **0.930** | **0.884** |
| BiLSTM | Deep learning | 0.901 | 0.869 |
| Linear SVM (calibrated) | Classical ML | 0.910 | 0.867 |
| Logistic Regression | Classical ML | 0.901 | 0.866 |
| Random Forest | Classical ML | 0.884 | 0.847 |
| LR baseline (no class weights) | Classical ML | 0.895 | 0.841 |
| Feedforward NN | Deep learning | 0.875 | 0.835 |
| Naive Bayes | Classical ML | 0.880 | 0.832 |

### Analysis
- **Best model:** DistilBERT has the highest accuracy and macro-F1, and is the app's intended default.
- **Trade-off:** DistilBERT is weaker on the rarest class, *surprise* (recall 0.59, F1 0.71), than BiLSTM (recall 0.89, F1 0.79).
- **Lightweight option:** linear models are small and answer in milliseconds on CPU.
- **Class weights (Logistic Regression):** recall rose from 0.55 to 0.91 on *surprise* and from 0.69 to 0.91 on *love*; *joy* recall fell from 0.97 to 0.89.
- **Common confusions:** *joy / love*, *fear / surprise* and *sadness / anger*.

### Web app
- Speak or type, then see the emotion, confidence and all six probabilities.
- Switch between the available models in the sidebar.
- Low-confidence and unclear inputs are handled separately.
- Messages mentioning self-harm show a support message instead of a generic reply.
- No API keys are needed. Voice input and spoken replies need internet.

## How to run the app
1. Clone the repository and go to Phase 2:
   ```bash
   git clone <repository-url>
   cd "DS-project/Phase 2"
   ```
2. Create and activate an environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```
3. Install the libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. For DistilBERT, download [`model.safetensors`](ADD_GOOGLE_DRIVE_LINK) into `models/transformer/distilbert-emotion/`.
5. Start the app:
   ```bash
   streamlit run app/app.py
   ```

### Re-running the notebooks
Copy the Phase 1 processed files into `Phase 2/data/processed/`. Then run `classification_ML` → `classification_DL` → `transformer` (GPU recommended) → `model_comparison`.

## Limitations and future work
- Trained on short English tweets; long, sarcastic, mixed-language or Roman Urdu input is out of scope.
- Speech-to-text errors can affect predictions, and voice tone is not analysed.
- Future work: more data for rare emotions, multi-label output, Urdu support, audio features and LLM-generated replies with safety checks.

## Disclaimer
VoxAid is a student project, not a medical or mental-health service.

## Reference
Saravia et al. (2018). *CARER: Contextualized Affect Representations for Emotion Recognition.* EMNLP.
