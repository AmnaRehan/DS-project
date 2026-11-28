import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voxaid_core import (EMOJI, LABELS, ROOT, available_models,  # noqa: E402
                         build_reply)

st.set_page_config(page_title="VoxAid", page_icon="🎙️", layout="centered")

COLORS = {"anger": "#E4572E", "fear": "#7B5EA7", "joy": "#F2B134",
          "love": "#E56399", "sadness": "#3E78B2", "surprise": "#17A398"}


@st.cache_resource(show_spinner="Loading emotion model...")
def get_model(name: str):
    for m in available_models():
        if m.name == name:
            return m.load()
    raise ValueError(name)


@st.cache_data(show_spinner=False)
def model_names() -> list[str]:
    return [m.name for m in available_models()]


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame | None:
    rows = []
    for fname in ["ml_metrics.json", "dl_metrics.json", "transformer_metrics.json"]:
        p = ROOT / "results" / fname
        if p.exists():
            for name, m in json.load(open(p)).items():
                rows.append({"model": name, "accuracy": m["accuracy"], "macro F1": m["macro_f1"]})
    return pd.DataFrame(rows).set_index("model").sort_values("macro F1", ascending=False) if rows else None


def speech_to_text(wav_bytes: bytes) -> tuple[str | None, str | None]:
    """Returns (text, error). Uses Google's free Web Speech API (needs internet)."""
    try:
        import speech_recognition as sr
    except ImportError:
        return None, "The SpeechRecognition package is not installed (pip install SpeechRecognition)."
    r = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            r.adjust_for_ambient_noise(source, duration=0.3)
            audio = r.record(source)
        return r.recognize_google(audio, language="en-US"), None
    except sr.UnknownValueError:
        return None, "Sorry, I couldn't make out any words. Please try again, a little closer to the microphone."
    except sr.RequestError as e:
        return None, f"The speech service could not be reached ({e}). Check your internet connection, or type instead."
    except Exception as e:  # malformed audio etc.
        return None, f"Could not process the recording: {e}"


@st.cache_data(show_spinner=False, max_entries=50)
def text_to_speech(text: str, slow: bool = False) -> bytes | None:
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang="en", slow=slow).write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


names = model_names()
with st.sidebar:
    st.header("⚙️ Settings")
    if not names:
        st.error("No trained models found in `models/`. Run `notebooks/classification_ML.ipynb` first.")
        st.stop()
    model_name = st.selectbox("Emotion model", names, index=0,
                              help="The first model is the best one by macro-F1 on the test set.")
    n_sentences = st.radio("Comforting sentences", [1, 2], index=1, horizontal=True)
    speak_back = st.toggle("Speak the reply out loud", value=True)
    slow_voice = st.toggle("Slower voice", value=False)

    results = load_results()
    if results is not None:
        st.divider()
        st.caption("Test-set results (2,000 sentences)")
        st.dataframe(results.style.format("{:.3f}"))

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.session_state.last_audio = None

st.session_state.setdefault("history", [])
st.session_state.setdefault("last_audio", None)

st.title("🎙️ VoxAid")
st.write("Tell me how you're feeling. I'll recognise the emotion in what you say "
         "and answer with a few kind words.")


def handle(text: str, source: str):
    model = get_model(model_name)
    probs = model.predict_proba(text)
    reply = build_reply(probs, text, n_sentences=n_sentences)
    audio = text_to_speech(reply.text, slow_voice) if speak_back else None
    st.session_state.history.insert(0, {"source": source, "text": text, "reply": reply,
                                        "audio": audio, "model": model_name})


tab_speak, tab_type = st.tabs(["🎤 Speak", "⌨️ Type"])

with tab_speak:
    wav = st.audio_input("Press the microphone, speak in English, then press stop")
    if wav is not None:
        data = wav.getvalue()
        digest = hashlib.md5(data).hexdigest()
        if digest != st.session_state.last_audio:      
            st.session_state.last_audio = digest
            with st.spinner("Listening..."):
                text, err = speech_to_text(data)
            if err:
                st.warning(err)
            else:
                handle(text, "voice")

with tab_type:
    with st.form("type_form", clear_on_submit=True):
        typed = st.text_area("How are you feeling?", placeholder="e.g. I've been so stressed about my exams lately",
                             height=90)
        if st.form_submit_button("Send", type="primary") and typed.strip():
            handle(typed.strip(), "text")


def render(entry: dict, latest: bool):
    reply = entry["reply"]
    emo = reply.emotion
    with st.container(border=True):
        st.markdown(f"**You said** ({'🎤' if entry['source'] == 'voice' else '⌨️'}): *\"{entry['text']}\"*")

        if reply.crisis:
            st.error(reply.text, icon="💛")
        elif reply.unclear:
            st.info(reply.text, icon="🤔")
        else:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(
                    f"<div style='text-align:center;padding:10px;border-radius:12px;"
                    f"background:{COLORS[emo]}22;border:2px solid {COLORS[emo]}'>"
                    f"<div style='font-size:42px'>{EMOJI[emo]}</div>"
                    f"<div style='font-size:20px;font-weight:600;color:{COLORS[emo]}'>{emo.capitalize()}</div>"
                    f"<div style='font-size:13px;opacity:.8'>{reply.confidence:.0%} confident</div></div>",
                    unsafe_allow_html=True)
            with c2:
                for label in sorted(LABELS, key=lambda l: -reply.probabilities.get(l, 0)):
                    p = reply.probabilities.get(label, 0.0)
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0;font-size:13px'>"
                        f"<span style='width:78px'>{EMOJI[label]} {label}</span>"
                        f"<div style='flex:1;background:#8881;border-radius:4px;height:12px'>"
                        f"<div style='width:{p * 100:.1f}%;background:{COLORS[label]};height:12px;border-radius:4px'></div></div>"
                        f"<span style='width:40px;text-align:right'>{p:.0%}</span></div>",
                        unsafe_allow_html=True)
            st.info(reply.text, icon="💬")

        if entry["audio"]:
            st.audio(entry["audio"], format="audio/mp3", autoplay=latest)
        elif speak_back and latest:
            st.caption("🔇 Voice reply unavailable (the text-to-speech service needs internet).")
        st.caption(f"Model: {entry['model']}")


if st.session_state.history:
    st.subheader("Response")
    render(st.session_state.history[0], latest=True)
    if len(st.session_state.history) > 1:
        with st.expander(f"Earlier ({len(st.session_state.history) - 1})"):
            for e in st.session_state.history[1:]:
                render(e, latest=False)

st.divider()
st.caption("VoxAid is a student project, not a medical or mental-health service. "
           "If you are struggling, please talk to someone you trust or a professional. "
           "Voice input uses Google's speech recognition and the spoken reply uses Google text-to-speech, "
           "so both need an internet connection.")
