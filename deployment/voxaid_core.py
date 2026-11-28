"""
VoxAid core logic: text preprocessing, model loading, prediction and responses.

The preprocessing here reproduces Phase 1 exactly, so text typed or spoken in the
app reaches the models in the same form as the training data:

* clean text   -> lower case, apostrophes dropped ("didn't" -> "didnt"),
                  everything except a-z replaced by a space, spaces collapsed
* text_ml      -> clean text with scikit-learn's English stop words removed,
                  but negations kept (not, no, never, cannot, without, ...)
* sequences    -> word ids from data/processed/vocab.json (0 = pad, 1 = unknown),
                  pre-padded / pre-truncated to length 52

Used by app/app.py and by the "sanity check" cells in the notebooks.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

LABELS = ["anger", "fear", "joy", "love", "sadness", "surprise"]

# ---------------------------------------------------------------------------
# Preprocessing (same rules as Phase 1)
# ---------------------------------------------------------------------------
NEGATIONS = {"no", "not", "nor", "never", "none", "nobody", "nothing",
             "nowhere", "neither", "cannot", "without"}


def _stop_words() -> frozenset:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return frozenset(ENGLISH_STOP_WORDS - NEGATIONS)


STOP_WORDS = _stop_words()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text) 
    text = re.sub(r"['’`]", "", text)               
    text = re.sub(r"[^a-z\s]", " ", text)              
    return re.sub(r"\s+", " ", text).strip()


def to_ml_text(clean: str) -> str:
    return " ".join(w for w in clean.split() if w not in STOP_WORDS)


_vocab_cache: dict | None = None


def load_vocab(path: Path | None = None) -> dict:
    global _vocab_cache
    if _vocab_cache is None:
        path = path or (MODELS_DIR / "dl" / "vocab.json")
        if not path.exists():
            path = DATA_DIR / "vocab.json"
        with open(path) as f:
            _vocab_cache = json.load(f)
    return _vocab_cache


def to_sequence(clean: str) -> np.ndarray:
    info = load_vocab()
    vocab, max_len = info["vocab"], info["max_len"]
    oov, pad = info.get("oov_id", 1), info.get("pad_id", 0)
    ids = [vocab.get(w, oov) for w in clean.split()]
    ids = ids[-max_len:]                                  
    ids = [pad] * (max_len - len(ids)) + ids              
    return np.array([ids], dtype=np.int32)


@dataclass
class EmotionModel:
    name: str
    kind: str            
    path: Path
    _model: object = field(default=None, repr=False)
    _tokenizer: object = field(default=None, repr=False)

    def load(self):
        if self._model is not None:
            return self
        if self.kind == "ml":
            import joblib
            self._model = joblib.load(self.path)
        elif self.kind == "dl":
            import keras
            self._model = keras.models.load_model(self.path)
        else:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.path)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.path)
            self._model.eval()
        return self

    def predict_proba(self, text: str) -> dict[str, float]:
        self.load()
        clean = clean_text(text)
        if self.kind == "ml":
            x = [to_ml_text(clean)]
            clf = self._model
            if hasattr(clf, "predict_proba"):
                p = clf.predict_proba(x)[0]
            else:                                        
                s = clf.decision_function(x)[0]
                p = np.exp(s - s.max()); p = p / p.sum()
            classes = list(clf.classes_)
        elif self.kind == "dl":
            p = self._model.predict(to_sequence(clean), verbose=0)[0]
            classes = LABELS
        else:
            import torch
            enc = self._tokenizer(clean, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                logits = self._model(**enc).logits[0]
            p = torch.softmax(logits, dim=-1).numpy()
            id2label = self._model.config.id2label
            classes = [id2label[i] for i in range(len(p))]
        probs = {c: float(v) for c, v in zip(classes, p)}
        return dict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))


ML_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest",
}


def available_models() -> list[EmotionModel]:
    """Every trained model found under models/, best/fastest first."""
    found: list[EmotionModel] = []

    tdir = MODELS_DIR / "transformer" / "distilbert-emotion"
    if (tdir / "config.json").exists():
        try:
            import transformers  
            import torch  
            found.append(EmotionModel("DistilBERT (transformer)", "transformer", tdir))
        except ImportError:
            pass

    for fname, label in [("lstm.keras", "BiLSTM (deep learning)"),
                         ("ffnn.keras", "Feedforward NN (deep learning)")]:
        p = MODELS_DIR / "dl" / fname
        if p.exists():
            try:
                import keras  
                found.append(EmotionModel(label, "dl", p))
            except ImportError:
                pass

    for slug, label in ML_DISPLAY.items():
        p = MODELS_DIR / "ml" / f"{slug}.joblib"
        if p.exists():
            found.append(EmotionModel(label, "ml", p))

    best = _best_model_name()
    if best:
        found.sort(key=lambda m: 0 if m.name.startswith(best) else 1)
    return found


def _best_model_name() -> str | None:
    path = ROOT / "results" / "best_model.json"
    if path.exists():
        with open(path) as f:
            return json.load(f).get("display_prefix")
    return None


EMOJI = {"anger": "😠", "fear": "😨", "joy": "😊", "love": "❤️",
         "sadness": "😢", "surprise": "😮"}

ACKNOWLEDGE = {
    "anger": "It sounds like you're feeling angry.",
    "fear": "It sounds like you're feeling scared or anxious.",
    "joy": "It sounds like you're feeling happy.",
    "love": "It sounds like you're feeling love and warmth.",
    "sadness": "It sounds like you're feeling sad.",
    "surprise": "It sounds like something surprised you.",
}

COMFORT = {
    "anger": [
        "Your feelings are valid, and it's okay to be upset.",
        "Try taking a few slow, deep breaths before you decide what to do next.",
        "Stepping away for a few minutes can make the situation feel clearer.",
        "Anger often tells us something matters to us, and that's worth listening to.",
        "You don't have to react right now; giving it some time is a strong choice.",
    ],
    "fear": [
        "You're not alone, and this feeling will pass.",
        "Try to focus on your breathing: in for four seconds, out for six.",
        "It's okay to feel afraid; it means you care about what happens.",
        "Take things one small step at a time; you don't have to solve everything at once.",
        "Talking to someone you trust can make a worry feel much lighter.",
    ],
    "joy": [
        "That's wonderful to hear, enjoy this moment!",
        "Moments like this are worth holding on to.",
        "Your happiness is well deserved.",
        "Maybe share this good news with someone close to you.",
        "Keep doing what makes you feel this way.",
    ],
    "love": [
        "That's a beautiful feeling to have.",
        "The people who make you feel this way are lucky to have you too.",
        "Cherish these connections; they make life brighter.",
        "Letting the people you care about know how you feel can mean a lot to them.",
        "It's lovely that you have so much warmth to give.",
    ],
    "sadness": [
        "I'm sorry you're going through this; your feelings matter.",
        "It's okay to not be okay, and it's okay to take things slowly.",
        "Be gentle with yourself today.",
        "Reaching out to a friend or family member could help you feel less alone.",
        "Hard days don't last forever, even when it feels that way.",
    ],
    "surprise": [
        "Unexpected moments can be a lot to take in, so give yourself a moment.",
        "Take a breath and let it sink in.",
        "Surprises can open doors we never expected.",
        "It's okay to need a little time to figure out how you feel about it.",
        "Life has a way of keeping things interesting!",
    ],
}

# Simple safety net: words that should never get only a generic reply.
CRISIS_PATTERNS = re.compile(
    r"\b(kill myself|killing myself|suicide|suicidal|end my life|want to die|"
    r"self harm|selfharm|hurt myself|no reason to live)\b"
)

CRISIS_MESSAGE = (
    "It sounds like you're going through something really painful, and you deserve "
    "support right now. Please reach out to someone you trust, or contact a crisis "
    "helpline or your local emergency services. You can find a free helpline in your "
    "country at findahelpline.com."
)


UNCLEAR_MESSAGE = ("I couldn't quite tell how you're feeling from that. "
                   "Could you tell me a little more, in a full sentence?")


def has_known_words(text: str) -> bool:
    """True if the text has at least one content word the models saw in training."""
    vocab = load_vocab()["vocab"]
    return any(w in vocab for w in to_ml_text(clean_text(text)).split())


@dataclass
class VoxAidReply:
    emotion: str
    confidence: float
    probabilities: dict[str, float]
    text: str
    crisis: bool = False
    unclear: bool = False


def build_reply(probs: dict[str, float], user_text: str,
                n_sentences: int = 2, low_conf: float = 0.40,
                rng: random.Random | None = None) -> VoxAidReply:
    rng = rng or random.Random()
    emotion, conf = next(iter(probs.items()))
    if CRISIS_PATTERNS.search(clean_text(user_text)):
        return VoxAidReply(emotion, conf, probs, CRISIS_MESSAGE, crisis=True)
    if not has_known_words(user_text):
        return VoxAidReply(emotion, conf, probs, UNCLEAR_MESSAGE, unclear=True)

    opener = ACKNOWLEDGE[emotion]
    if conf < low_conf:
        opener = "I'm not completely sure, but " + opener[0].lower() + opener[1:]
    lines = rng.sample(COMFORT[emotion], k=max(1, min(n_sentences, 2)))
    return VoxAidReply(emotion, conf, probs, " ".join([opener, *lines]))
