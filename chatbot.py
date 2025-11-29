# chatbot.py
import os, json, logging, requests
from typing import Optional, List, Dict

logger = logging.getLogger("medguardian.chatbot")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
REST_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95

def _get_api_key():
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets.get("GEMINI_API_KEY"):
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    k = os.environ.get("GEMINI_API_KEY")
    if k:
        return k
    raise RuntimeError("GEMINI_API_KEY not found in environment or Streamlit secrets.")

def _build_system_prompt(style="concise"):
    base = ("You are MedGuardian, a careful medical assistant. Provide accurate, concise medical information, suggest tests and lifestyle changes, "
            "and recommend when to see a professional. Do not give prescriptions or definitive diagnoses. For emergencies advise immediate care.")
    if style=="detailed":
        base += " Provide step-by-step checks, red flags, and suggested next steps."
    return base

def _call_gemini_rest(prompt: str, max_tokens:int=DEFAULT_MAX_TOKENS, temperature:float=DEFAULT_TEMPERATURE, top_p:float=DEFAULT_TOP_P) -> str:
    api_key = _get_api_key()
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p
        }
    }
    headers = {"Content-Type":"application/json", "x-goog-api-key": api_key}
    resp = requests.post(REST_ENDPOINT, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        logger.error("Gemini REST failed: %s %s", resp.status_code, resp.text)
        raise RuntimeError(f"Gemini REST error {resp.status_code}: {resp.text}")
    data = resp.json()
    cands = data.get("candidates", [])
    if cands:
        c = cands[0]
        cont = c.get("content", {})
        parts = cont.get("parts", [])
        if parts:
            return "".join([p.get("text","") for p in parts if isinstance(p, dict)])
    return data.get("text", json.dumps(data))

def doctor_chatbot(user_input: str, user_id: Optional[int]=None, use_gemini: bool=True,
                   style: str="concise", max_tokens:int=DEFAULT_MAX_TOKENS,
                   temperature:float=DEFAULT_TEMPERATURE, top_p:float=DEFAULT_TOP_P) -> str:
    if not user_input or not isinstance(user_input, str):
        return "Please type your question."

    q = user_input.strip()
    ql = q.lower()

    # emergency & quick rules
    if any(x in ql for x in ["chest pain", "severe chest", "heart attack", "difficulty breathing"]):
        return "If you have sudden severe chest pain or difficulty breathing, seek emergency care immediately."
    if "diabetes" in ql:
        return "Quick: get HbA1c and fasting glucose tests. Reduce sugars/refined carbs, exercise, and consult a doctor."
    if "blood pressure" in ql or ql=="bp":
        return "Monitor BP, reduce salt, exercise, and consult your doctor if consistently high."
    if "kidney" in ql:
        return "Stay hydrated, avoid unnecessary NSAIDs, check creatinine and urine tests, and see a nephrologist for abnormal results."

    if not use_gemini:
        return "Short local tip: enable Gemini for detailed answers."

    # build prompt
    system = _build_system_prompt(style=style)
    pieces = [system, "\nConversation:", f"User: {q}", "Assistant:"]
    prompt_text = "\n".join(pieces)

    try:
        reply = _call_gemini_rest(prompt_text, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    except Exception as e:
        logger.exception("Gemini call failed")
        return f"Sorry — the AI service is unavailable right now ({e}). Try again later, or ask about diabetes/heart/bp/kidney for quick tips."

    return reply.strip()
