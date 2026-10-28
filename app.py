"""
Food Verse AI - Multimodal Food Intelligence Platform (pure Streamlit app)

Features (one tab each):
  - Dashboard        : analysis history, averages and alerts
  - Food Recognition : identify dishes and ingredients from a photo
  - Nutrition        : calories, macros and a health score (photo and/or text)
  - Freshness        : OpenCV colour/texture analysis + Gemini visual check
  - Hygiene          : kitchen video -> glove / hair-cap / handling / cleanliness audit
  - Cooking          : recipes, substitutions and step-by-step guidance
  - Meal Plan        : personalized, adaptive meal plans
  - Multimodal       : image + voice + text combined into one report

Reasoning engine: Google Gemini (vision, audio and text) via the google-genai SDK.
API key:          st.secrets["GEMINI_API_KEY"]  (.streamlit/secrets.toml locally,
                  App settings -> Secrets on Streamlit Community Cloud).

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Food Verse AI", page_icon="🍽️", layout="wide")

GEMINI_MODEL = "gemini-2.5-flash"  # any Gemini model with image/audio input works
MAX_VIDEO_MB = 50  # Streamlit Cloud's default upload limit is 200 MB; keep videos small
HYGIENE_FRAMES = 8

SYSTEM_BASE = (
    "You are Food Verse AI, a food intelligence assistant for nutrition, food safety, "
    "hygiene and cooking. Be precise and honest about uncertainty. Nutrition values are "
    "estimates, not medical advice. Never claim certainty about food safety from a photo alone."
)

if "history" not in st.session_state:
    st.session_state["history"] = []


def record(kind: str, summary: Dict[str, Any]) -> None:
    st.session_state["history"].append({"type": kind, "time": time.strftime("%H:%M:%S"), "summary": summary})


# --------------------------------------------------------------------------- #
# Gemini reasoning engine
# --------------------------------------------------------------------------- #
class AIError(Exception):
    """Raised for any failure talking to the AI model (shown to the user via st.error)."""


def get_api_key() -> str:
    key = None
    try:
        key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        key = None
    key = key or os.getenv("GEMINI_API_KEY")
    if not key or "paste-your" in str(key):
        raise AIError(
            "GEMINI_API_KEY is missing. Add it under **App settings -> Secrets** on Streamlit Cloud "
            '(`GEMINI_API_KEY = "your-key"`), or in `.streamlit/secrets.toml` when running locally. '
            "Get a free key at https://aistudio.google.com/apikey"
        )
    return str(key)


@st.cache_resource(show_spinner=False)
def _client(api_key: str):
    from google import genai

    return genai.Client(api_key=api_key)


def parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise AIError("The AI model returned a response that was not valid JSON. Please try again.")


def ask_llm(
    prompt: str,
    images: Optional[List[bytes]] = None,
    audio: Optional[Tuple[bytes, str]] = None,
    system: str = SYSTEM_BASE,
    max_tokens: int = 2000,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Send text (+ JPEG images, + optional audio) to Gemini and return parsed JSON."""
    from google.genai import types

    parts: List[Any] = []
    for img in images or []:
        parts.append(types.Part.from_bytes(data=img, mime_type="image/jpeg"))
    if audio:
        parts.append(types.Part.from_bytes(data=audio[0], mime_type=audio[1]))
    parts.append(prompt + "\n\nRespond with a single valid JSON object only. No markdown, no commentary.")

    try:
        resp = _client(get_api_key()).models.generate_content(
            model=GEMINI_MODEL,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens + 4096,  # headroom for Gemini 2.5 internal thinking
                response_mime_type="application/json",
            ),
        )
    except AIError:
        raise
    except Exception as exc:
        raise AIError(f"Gemini request failed. Check your API key, quota and model name. Details: {exc}")
    if not resp.text:
        raise AIError("Gemini returned an empty response (it may have been blocked by safety filters).")
    return parse_json(resp.text)


# --------------------------------------------------------------------------- #
# Image / video helpers (OpenCV)
# --------------------------------------------------------------------------- #
def shrink_image(data: bytes, max_side: int = 1280) -> bytes:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise AIError("Could not read that image. Please use a JPG, PNG or WEBP file.")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise AIError("Image re-encoding failed.")
    return buf.tobytes()


def image_input(key: str) -> Optional[bytes]:
    """Upload or camera capture -> shrunken JPEG bytes (or None)."""
    up = st.file_uploader("Upload a food photo", type=["jpg", "jpeg", "png", "webp"], key=f"{key}_up")
    with st.expander("📷 Or take a photo with your camera"):
        cam = st.camera_input("Camera", key=f"{key}_cam", label_visibility="collapsed")
    src = up or cam
    if not src:
        return None
    try:
        data = shrink_image(src.getvalue())
    except AIError as exc:
        st.error(str(exc))
        return None
    st.image(data, width=320)
    return data


def freshness_features(image_bytes: bytes) -> Dict[str, float]:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (400, max(1, int(400 * img.shape[0] / img.shape[1]))))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(hsv)
    total = float(h.size)

    dark_ratio = float(np.count_nonzero(v < 50)) / total
    brown = (h >= 8) & (h <= 25) & (s > 60) & (v >= 40) & (v < 130)
    brown_ratio = float(np.count_nonzero(brown)) / total
    mold_white = (s < 40) & (v > 170)
    mold_gb = (h >= 35) & (h <= 110) & (s > 25) & (s < 110) & (v < 160)
    mold_ratio = float(np.count_nonzero(mold_white | mold_gb)) / total
    return {
        "dark_patch_ratio": round(dark_ratio, 4),
        "brown_discoloration_ratio": round(brown_ratio, 4),
        "mold_like_ratio": round(mold_ratio, 4),
        "texture_variance": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
        "mean_saturation": round(float(s.mean()), 2),
        "hue_std": round(float(h.std()), 2),
    }


def label_for(score: float) -> str:
    return "Fresh" if score >= 72 else "Moderately Fresh" if score >= 45 else "Spoiled"


def heuristic_freshness(f: Dict[str, float]) -> Dict[str, Any]:
    penalty = (
        f["dark_patch_ratio"] * 160
        + f["brown_discoloration_ratio"] * 110
        + f["mold_like_ratio"] * 60
        + min(f["texture_variance"] / 1500, 1) * 15
        + (10 if f["mean_saturation"] < 40 else 0)
    )
    score = int(max(0, min(100, 100 - penalty)))
    return {"score": score, "label": label_for(score)}


def sample_video_frames(path: str, n: int) -> List[bytes]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise AIError("Could not open that video. Try an MP4, MOV, AVI or WEBM file.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    selected = []
    if total > 0:
        for i in np.linspace(0, total - 1, min(n, total)).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok:
                selected.append(fr)
    else:  # unknown length: read everything and pick evenly
        buf = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            buf.append(fr)
        if buf:
            selected = [buf[i] for i in np.linspace(0, len(buf) - 1, min(n, len(buf))).astype(int)]
    cap.release()

    frames: List[bytes] = []
    for fr in selected:
        h, w = fr.shape[:2]
        scale = 1024 / max(h, w)
        if scale < 1:
            fr = cv2.resize(fr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, enc = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            frames.append(enc.tobytes())
    if not frames:
        raise AIError("Could not extract any frames from the video.")
    return frames


# --------------------------------------------------------------------------- #
# Meal-plan maths
# --------------------------------------------------------------------------- #
def estimate_targets(age, sex, height_cm, weight_kg, activity, goal) -> Dict[str, Any]:
    """Mifflin-St Jeor BMR -> TDEE -> goal-adjusted calories and macro targets."""
    s = 5 if sex == "male" else -161 if sex == "female" else -78
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + s
    mult = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "very_active": 1.9}[activity]
    tdee = bmr * mult
    adj = {"lose_weight": -400, "maintain": 0, "gain_muscle": 300, "eat_healthier": 0}[goal]
    kcal = max(1200, tdee + adj)
    protein = weight_kg * (2.0 if goal == "gain_muscle" else 1.6 if goal == "lose_weight" else 1.2)
    fat = kcal * 0.28 / 9
    carbs = max(0.0, (kcal - protein * 4 - fat * 9) / 4)
    return {
        "bmi": round(weight_kg / ((height_cm / 100) ** 2), 1),
        "bmr_kcal": round(bmr),
        "tdee_kcal": round(tdee),
        "target_calories_kcal": round(kcal),
        "target_protein_g": round(protein),
        "target_carbs_g": round(carbs),
        "target_fat_g": round(fat),
    }


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def run(label: str, fn, state_key: str) -> None:
    """Run fn() with a spinner, store the result in session state, surface errors nicely."""
    try:
        with st.spinner(label):
            st.session_state[state_key] = fn()
    except AIError as exc:
        st.session_state.pop(state_key, None)
        st.error(str(exc))
    except Exception as exc:  # unexpected
        st.session_state.pop(state_key, None)
        st.error(f"Something went wrong: {exc}")


def bullets(title: str, items: Optional[List[str]], icon: str = "•") -> None:
    if items:
        st.markdown(f"**{title}**")
        for it in items:
            st.markdown(f"{icon} {it}")


def raw_expander(data: Dict[str, Any]) -> None:
    with st.expander("Raw AI output (JSON)"):
        st.json(data)


def severity_box(sev: str, msg: str) -> None:
    {"critical": st.error, "warning": st.warning}.get(sev, st.info)(msg)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🍽️ Food Verse AI")
    st.caption("Multimodal food intelligence: vision, video, audio and LLM reasoning.")
    try:
        get_api_key()
        st.success("Gemini API key found", icon="🔑")
    except AIError as exc:
        st.error(str(exc))
    st.caption(f"Model: `{GEMINI_MODEL}`")
    st.divider()
    st.caption(
        "Estimates only. Nutrition values are approximate and freshness/hygiene checks from images "
        "cannot replace lab tests or official inspections."
    )

st.title("Food Verse AI")
st.caption("Recognize food, analyse nutrition, check freshness, audit kitchen hygiene, cook smarter and plan meals.")

tabs = st.tabs(
    ["📊 Dashboard", "🍛 Recognition", "🥗 Nutrition", "🍎 Freshness", "🧑‍🍳 Hygiene", "📖 Cooking", "🗓️ Meal Plan", "🎙️ Multimodal"]
)

# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
with tabs[0]:
    hist = st.session_state["history"]
    st.subheader("Insights dashboard")
    if not hist:
        st.info("No analyses yet. Use the other tabs and your results will be summarised here.")
    else:
        cals = [h["summary"]["calories"] for h in hist if h["type"] == "nutrition" and h["summary"].get("calories")]
        hs = [h["summary"]["health_score"] for h in hist if h["type"] == "nutrition" and h["summary"].get("health_score")]
        fresh = [h["summary"]["classification"] for h in hist if h["type"] == "freshness"]
        hyg = [h["summary"]["score"] for h in hist if h["type"] == "hygiene" and h["summary"].get("score") is not None]
        avg = lambda xs: round(sum(xs) / len(xs), 1) if xs else "–"

        c = st.columns(4)
        c[0].metric("Analyses run", len(hist))
        c[1].metric("Avg calories / meal", avg(cals))
        c[2].metric("Avg health score", avg(hs))
        c[3].metric("Avg hygiene score", avg(hyg))

        if fresh.count("Spoiled"):
            st.warning(f"{fresh.count('Spoiled')} item(s) were classified as spoiled.")
        if hyg and min(hyg) < 60:
            st.error(f"A hygiene audit scored as low as {min(hyg)}/100.")
        if hs and sum(hs) / len(hs) < 50:
            st.info("Average meal health score is below 50. Consider lighter, fresher meals.")

        by_type: Dict[str, int] = {}
        for h in hist:
            by_type[h["type"]] = by_type.get(h["type"], 0) + 1
        st.markdown("**Analyses by type**")
        st.bar_chart(by_type)

        st.markdown("**Recent activity**")
        st.dataframe(
            [{"time": h["time"], "type": h["type"], "summary": json.dumps(h["summary"])} for h in hist[::-1][:15]],
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()

# --------------------------------------------------------------------------- #
# Food recognition
# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader("Food recognition")
    img = image_input("rec")
    ctx = st.text_input("Optional context", placeholder="e.g. lunch at a South Indian restaurant", key="rec_ctx")

    def do_recognize():
        prompt = f"""Identify every distinct dish and food item visible in the image.
Optional user context: "{ctx}"

Return JSON:
{{
  "dishes": [{{"name": str, "cuisine": str, "confidence": float(0-1), "description": str}}],
  "ingredients_visible": [str],
  "estimated_portion": str,
  "is_food": bool,
  "notes": str
}}
If the image contains no food, set is_food=false and dishes=[]."""
        res = ask_llm(prompt, [img])
        record("recognition", {"dishes": [d.get("name") for d in res.get("dishes", [])]})
        return res

    if st.button("Identify food", type="primary", disabled=img is None, key="rec_btn"):
        run("Identifying food...", do_recognize, "res_rec")

    r = st.session_state.get("res_rec")
    if r:
        if not r.get("is_food", True):
            st.warning("No food was detected in this image.")
        for d in r.get("dishes", []):
            st.markdown(f"### {d.get('name')}  \n*{d.get('cuisine', '')}* · confidence {float(d.get('confidence', 0)):.0%}")
            st.write(d.get("description", ""))
        if r.get("ingredients_visible"):
            st.markdown("**Visible ingredients:** " + ", ".join(r["ingredients_visible"]))
        if r.get("estimated_portion"):
            st.caption(f"Estimated portion: {r['estimated_portion']}")
        if r.get("notes"):
            st.caption(r["notes"])
        raw_expander(r)

# --------------------------------------------------------------------------- #
# Nutrition
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.subheader("Nutrition analysis")
    img = image_input("nut")
    desc = st.text_area("Or describe the meal", placeholder="2 rotis with dal and mixed veg sabzi", key="nut_desc")
    goal = st.selectbox(
        "Your goal", ["maintain", "lose_weight", "gain_muscle", "eat_healthier"], key="nut_goal",
        format_func=lambda g: g.replace("_", " ").title(),
    )

    def do_nutrition():
        prompt = f"""Estimate the nutrition of the food {'in the image ' if img else ''}{('and described as: ' + desc) if desc.strip() else ''}.
User goal: {goal}.

Return JSON:
{{
  "items": [{{"name": str, "portion": str, "calories_kcal": number, "protein_g": number, "carbs_g": number, "fat_g": number, "fiber_g": number, "sugar_g": number, "sodium_mg": number}}],
  "totals": {{"calories_kcal": number, "protein_g": number, "carbs_g": number, "fat_g": number, "fiber_g": number, "sugar_g": number, "sodium_mg": number}},
  "health_score": integer(0-100),
  "health_score_reasoning": str,
  "positives": [str],
  "concerns": [str],
  "healthier_swaps": [str],
  "goal_fit": str,
  "confidence": float(0-1)
}}"""
        res = ask_llm(prompt, [img] if img else None)
        record("nutrition", {"calories": res.get("totals", {}).get("calories_kcal"), "health_score": res.get("health_score")})
        return res

    if st.button("Analyze nutrition", type="primary", disabled=(img is None and not desc.strip()), key="nut_btn"):
        run("Analyzing nutrition...", do_nutrition, "res_nut")

    r = st.session_state.get("res_nut")
    if r:
        t = r.get("totals", {})
        c = st.columns(5)
        c[0].metric("Calories", f"{t.get('calories_kcal', '–')} kcal")
        c[1].metric("Protein", f"{t.get('protein_g', '–')} g")
        c[2].metric("Carbs", f"{t.get('carbs_g', '–')} g")
        c[3].metric("Fat", f"{t.get('fat_g', '–')} g")
        c[4].metric("Health score", f"{r.get('health_score', '–')}/100")
        try:
            st.progress(max(0, min(100, int(r.get("health_score", 0)))) / 100)
        except Exception:
            pass
        st.write(r.get("health_score_reasoning", ""))
        if r.get("items"):
            st.dataframe(r["items"], use_container_width=True, hide_index=True)
        col1, col2 = st.columns(2)
        with col1:
            bullets("Positives", r.get("positives"), "✅")
            bullets("Healthier swaps", r.get("healthier_swaps"), "🔁")
        with col2:
            bullets("Concerns", r.get("concerns"), "⚠️")
            if r.get("goal_fit"):
                st.markdown("**Fit with your goal**")
                st.write(r["goal_fit"])
        st.caption("Estimates only; not medical advice.")
        raw_expander(r)

# --------------------------------------------------------------------------- #
# Freshness
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Freshness detection")
    img = image_input("fresh")
    ftype = st.text_input("Food type (optional)", placeholder="e.g. tomatoes, chicken, bread", key="fresh_type")

    def do_freshness():
        features = freshness_features(img)
        heur = heuristic_freshness(features)
        llm, err = None, None
        try:
            prompt = f"""Assess the freshness of the food in this image (food type hint: "{ftype or 'unknown'}").
Look at color, texture, discoloration, wilting, bruising, mold, slime and surface patterns.
Objective image measurements from a computer-vision pre-check: {json.dumps(features)}
Pre-check heuristic result: {json.dumps(heur)}

Return JSON:
{{
  "food_identified": str,
  "classification": "Fresh" | "Moderately Fresh" | "Spoiled",
  "freshness_score": integer(0-100),
  "visual_indicators": [{{"indicator": str, "observation": str, "severity": "none"|"low"|"medium"|"high"}}],
  "estimated_remaining_shelf_life": str,
  "safe_to_eat": "yes" | "caution" | "no",
  "storage_advice": [str],
  "confidence": float(0-1),
  "limitations": str
}}"""
            llm = ask_llm(prompt, [img])
        except AIError as exc:
            err = str(exc)
        if llm:
            score = int(round(0.6 * float(llm.get("freshness_score", heur["score"])) + 0.4 * heur["score"]))
        else:
            score = heur["score"]
        res = {"score": score, "label": label_for(score), "features": features, "heuristic": heur, "ai": llm, "ai_error": err}
        record("freshness", {"classification": res["label"], "score": score})
        return res

    if st.button("Check freshness", type="primary", disabled=img is None, key="fresh_btn"):
        run("Analyzing colour, texture and surface patterns...", do_freshness, "res_fresh")

    r = st.session_state.get("res_fresh")
    if r:
        c = st.columns(2)
        c[0].metric("Classification", r["label"])
        c[1].metric("Freshness score", f"{r['score']}/100")
        st.progress(r["score"] / 100)
        {"Fresh": st.success, "Moderately Fresh": st.warning, "Spoiled": st.error}[r["label"]](
            "Visual analysis cannot detect all spoilage. When in doubt, throw it out."
        )
        if r["ai_error"]:
            st.info(f"AI check unavailable, showing the local computer-vision result only. ({r['ai_error']})")
        ai = r["ai"]
        if ai:
            st.write(f"**Identified:** {ai.get('food_identified', '–')} · **Safe to eat:** {ai.get('safe_to_eat', '–')} · "
                     f"**Shelf life:** {ai.get('estimated_remaining_shelf_life', '–')}")
            if ai.get("visual_indicators"):
                st.dataframe(ai["visual_indicators"], use_container_width=True, hide_index=True)
            bullets("Storage advice", ai.get("storage_advice"), "🧊")
            if ai.get("limitations"):
                st.caption(ai["limitations"])
        with st.expander("Computer-vision measurements"):
            st.json({"metrics": r["features"], "heuristic": r["heuristic"]})

# --------------------------------------------------------------------------- #
# Hygiene (video)
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.subheader("Restaurant hygiene monitoring")
    st.caption(f"Upload a short kitchen video (up to {MAX_VIDEO_MB} MB). {HYGIENE_FRAMES} frames are sampled and audited.")
    vid = st.file_uploader("Kitchen video", type=["mp4", "mov", "avi", "webm", "mkv"], key="hyg_vid")
    loc = st.text_input("Location", value="Restaurant kitchen", key="hyg_loc")
    if vid:
        st.video(vid)

    def do_hygiene():
        if vid.size > MAX_VIDEO_MB * 1024 * 1024:
            raise AIError(f"Video is too large (max {MAX_VIDEO_MB} MB).")
        suffix = os.path.splitext(vid.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(vid.getvalue())
            path = tmp.name
        try:
            frames = sample_video_frames(path, HYGIENE_FRAMES)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        prompt = f"""These {len(frames)} frames are sampled in time order from a video of: {loc}.
Evaluate food-safety hygiene practices of the visible staff and environment.

Check each indicator and report status "compliant" | "violation" | "not_visible":
- glove_usage, hair_cap_or_hairnet, apron_or_clean_uniform, hand_washing_or_sanitizing,
  raw_vs_cooked_separation (cross-contamination), food_handling_technique (bare-hand contact with ready-to-eat food, touching face/phone),
  surface_and_equipment_cleanliness, floor_and_waste_management, pest_signs, food_storage_and_labelling

Return JSON:
{{
  "overall_hygiene_score": integer(0-100),
  "risk_level": "low" | "medium" | "high",
  "indicators": [{{"name": str, "status": "compliant"|"violation"|"not_visible", "evidence": str}}],
  "alerts": [{{"severity": "info"|"warning"|"critical", "message": str}}],
  "staff_observed": integer,
  "recommendations": [str],
  "summary": str,
  "confidence": float(0-1)
}}
Only report violations you can actually see; use "not_visible" instead of guessing."""
        res = ask_llm(prompt, frames, system=SYSTEM_BASE + " You are acting as a food-safety hygiene auditor (HACCP-aware).", max_tokens=2500)
        res["_frames"] = len(frames)
        record("hygiene", {"score": res.get("overall_hygiene_score"), "risk": res.get("risk_level")})
        return res

    if st.button("Analyze hygiene", type="primary", disabled=vid is None, key="hyg_btn"):
        run("Sampling frames and auditing hygiene...", do_hygiene, "res_hyg")

    r = st.session_state.get("res_hyg")
    if r:
        c = st.columns(3)
        c[0].metric("Hygiene score", f"{r.get('overall_hygiene_score', '–')}/100")
        c[1].metric("Risk level", str(r.get("risk_level", "–")).title())
        c[2].metric("Staff observed", r.get("staff_observed", "–"))
        st.write(r.get("summary", ""))
        for a in r.get("alerts", []):
            severity_box(a.get("severity", "info"), a.get("message", ""))
        if r.get("indicators"):
            icons = {"compliant": "✅ compliant", "violation": "❌ violation", "not_visible": "➖ not visible"}
            st.dataframe(
                [{"indicator": i.get("name"), "status": icons.get(i.get("status"), i.get("status")), "evidence": i.get("evidence")}
                 for i in r["indicators"]],
                use_container_width=True, hide_index=True,
            )
        bullets("Recommendations", r.get("recommendations"), "🛠️")
        st.caption(f"Based on {r.get('_frames')} sampled frames. Not a substitute for an official inspection.")
        raw_expander(r)

# --------------------------------------------------------------------------- #
# Cooking assistant
# --------------------------------------------------------------------------- #
with tabs[5]:
    st.subheader("Smart cooking assistant")
    with st.form("cook_form"):
        query = st.text_input("What do you want to cook?", placeholder="Paneer butter masala, or 'something quick with eggs'")
        c1, c2, c3 = st.columns(3)
        servings = c1.number_input("Servings", 1, 50, 2)
        max_time = c2.number_input("Max time (min, 0 = no limit)", 0, 600, 0)
        skill = c3.selectbox("Skill level", ["beginner", "intermediate", "advanced"])
        c4, c5 = st.columns(2)
        diet = c4.text_input("Diet", value="none", help="vegetarian, vegan, keto, jain, halal...")
        allergies = c5.text_input("Allergies (comma separated)")
        have = st.text_input("Ingredients you already have (comma separated)")
        c6, c7 = st.columns(2)
        cur_step = c6.number_input("I'm on step (0 = not started)", 0, 50, 0)
        question = c7.text_input("Question about the cooking process (optional)")
        go = st.form_submit_button("Get recipe & guidance", type="primary")

    if go:
        if len(query.strip()) < 2:
            st.warning("Tell me what you'd like to cook.")
        else:
            def do_cook():
                prompt = f"""User request: "{query}"
Servings: {servings}; skill: {skill}; diet: {diet}; allergies: {allergies or 'none'}
Ingredients on hand: {have or 'not specified'}
Max time: {max_time or 'no limit'} minutes
Currently on step: {cur_step or 'not started'}
Follow-up question: {question or 'none'}

Provide a complete recipe plus contextual guidance. Never include listed allergens; respect the diet.
Return JSON:
{{
  "dish": str, "prep_time_min": integer, "cook_time_min": integer, "difficulty": str,
  "ingredients": [{{"item": str, "quantity": str, "have_it": bool}}],
  "missing_ingredients": [str],
  "substitutions": [{{"missing": str, "substitute": str}}],
  "equipment": [str],
  "steps": [{{"step": integer, "instruction": str, "time_min": number|null, "tip": str, "doneness_cue": str}}],
  "food_safety_tips": [str],
  "estimated_nutrition_per_serving": {{"calories_kcal": number, "protein_g": number, "carbs_g": number, "fat_g": number}},
  "contextual_guidance": str,
  "answer_to_question": str|null
}}"""
                res = ask_llm(prompt, max_tokens=3000, temperature=0.5)
                record("cooking", {"dish": res.get("dish")})
                return res

            run("Writing your recipe...", do_cook, "res_cook")

    r = st.session_state.get("res_cook")
    if r:
        st.markdown(f"## {r.get('dish', 'Recipe')}")
        c = st.columns(3)
        c[0].metric("Prep", f"{r.get('prep_time_min', '–')} min")
        c[1].metric("Cook", f"{r.get('cook_time_min', '–')} min")
        c[2].metric("Difficulty", r.get("difficulty", "–"))
        if r.get("contextual_guidance"):
            st.info(r["contextual_guidance"])
        if r.get("answer_to_question"):
            st.success(f"**Your question:** {r['answer_to_question']}")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Ingredients**")
            st.dataframe(
                [{"item": i.get("item"), "qty": i.get("quantity"), "have": "✅" if i.get("have_it") else "🛒"} for i in r.get("ingredients", [])],
                use_container_width=True, hide_index=True,
            )
            bullets("Equipment", r.get("equipment"), "🍳")
            if r.get("substitutions"):
                st.markdown("**Substitutions**")
                for s in r["substitutions"]:
                    st.markdown(f"• {s.get('missing')} → {s.get('substitute')}")
        with col2:
            st.markdown("**Steps**")
            for s in r.get("steps", []):
                t = f" ({s['time_min']} min)" if s.get("time_min") else ""
                st.markdown(f"**{s.get('step')}.** {s.get('instruction')}{t}")
                if s.get("tip"):
                    st.caption(f"Tip: {s['tip']}")
                if s.get("doneness_cue"):
                    st.caption(f"Done when: {s['doneness_cue']}")
        bullets("Food safety tips", r.get("food_safety_tips"), "🧼")
        n = r.get("estimated_nutrition_per_serving")
        if n:
            st.caption(f"Per serving (estimate): {n.get('calories_kcal')} kcal · P {n.get('protein_g')} g · C {n.get('carbs_g')} g · F {n.get('fat_g')} g")
        raw_expander(r)

# --------------------------------------------------------------------------- #
# Meal plan
# --------------------------------------------------------------------------- #
with tabs[6]:
    st.subheader("Personalized meal plan")
    with st.form("plan_form"):
        c = st.columns(4)
        age = c[0].number_input("Age", 5, 100, 25)
        sex = c[1].selectbox("Sex", ["male", "female", "other"])
        height = c[2].number_input("Height (cm)", 100.0, 230.0, 170.0)
        weight = c[3].number_input("Weight (kg)", 25.0, 250.0, 70.0)
        c = st.columns(3)
        activity = c[0].selectbox("Activity", ["sedentary", "light", "moderate", "active", "very_active"], index=2)
        pgoal = c[1].selectbox("Goal", ["maintain", "lose_weight", "gain_muscle", "eat_healthier"], format_func=lambda g: g.replace("_", " ").title())
        pdiet = c[2].text_input("Diet", value="none", help="vegetarian, vegan, keto, jain, halal...")
        c = st.columns(2)
        pallergies = c[0].text_input("Allergies (comma separated)")
        pdislikes = c[1].text_input("Dislikes (comma separated)")
        cuisines = st.text_input("Preferred cuisines (comma separated)", placeholder="North Indian, Mediterranean")
        medical = st.text_input("Health notes (optional, informational)", placeholder="e.g. type 2 diabetes")
        c = st.columns(3)
        days = c[0].slider("Days", 1, 14, 7)
        mpd = c[1].slider("Meals per day", 2, 6, 3)
        budget = c[2].selectbox("Budget", ["low", "medium", "high"], index=1)
        feedback = st.text_input("Feedback to adapt the plan (optional)", placeholder="too much rice, more protein")
        go = st.form_submit_button("Generate meal plan", type="primary")

    if go:
        def do_plan():
            targets = estimate_targets(age, sex, height, weight, activity, pgoal)
            profile = {
                "age": age, "sex": sex, "height_cm": height, "weight_kg": weight, "activity_level": activity,
                "goal": pgoal, "diet": pdiet, "allergies": pallergies, "dislikes": pdislikes,
                "cuisine_preferences": cuisines, "health_notes": medical,
            }
            prompt = f"""Create a {days}-day meal plan with {mpd} meals per day.
Profile: {json.dumps(profile)}
Calculated targets: {json.dumps(targets)}
Budget: {budget}
Adaptive feedback from user: {feedback or 'none'}

Strictly exclude allergens and disliked foods, and respect the diet. Vary meals; favour listed cuisines.
Return JSON:
{{
  "daily_targets": {{"calories_kcal": number, "protein_g": number, "carbs_g": number, "fat_g": number}},
  "days": [{{
    "day": integer,
    "meals": [{{"meal": str, "name": str, "ingredients": [str], "calories_kcal": number, "protein_g": number, "carbs_g": number, "fat_g": number, "prep_time_min": integer}}],
    "day_totals": {{"calories_kcal": number, "protein_g": number, "carbs_g": number, "fat_g": number}}
  }}],
  "grocery_list": [{{"category": str, "items": [str]}}],
  "hydration_tip": str,
  "adaptation_notes": str,
  "guidance": [str]
}}"""
            res = ask_llm(prompt, max_tokens=6000, temperature=0.6)
            res["_targets"] = targets
            record("mealplan", {"days": days, "goal": pgoal})
            return res

        run("Building your meal plan (this can take a little while)...", do_plan, "res_plan")

    r = st.session_state.get("res_plan")
    if r:
        tg = r.get("_targets", {})
        c = st.columns(5)
        c[0].metric("BMI", tg.get("bmi", "–"))
        c[1].metric("Target kcal", tg.get("target_calories_kcal", "–"))
        c[2].metric("Protein", f"{tg.get('target_protein_g', '–')} g")
        c[3].metric("Carbs", f"{tg.get('target_carbs_g', '–')} g")
        c[4].metric("Fat", f"{tg.get('target_fat_g', '–')} g")
        if r.get("adaptation_notes"):
            st.info(r["adaptation_notes"])
        for d in r.get("days", []):
            tot = d.get("day_totals", {})
            with st.expander(f"Day {d.get('day')} · {tot.get('calories_kcal', '–')} kcal · P {tot.get('protein_g', '–')} g", expanded=d.get("day") == 1):
                for m in d.get("meals", []):
                    st.markdown(f"**{m.get('meal')}: {m.get('name')}** · {m.get('calories_kcal')} kcal · P {m.get('protein_g')} g · C {m.get('carbs_g')} g · F {m.get('fat_g')} g")
                    st.caption(", ".join(m.get("ingredients", [])) + f" · {m.get('prep_time_min', '–')} min")
        if r.get("grocery_list"):
            st.markdown("**🛒 Grocery list**")
            for g in r["grocery_list"]:
                st.markdown(f"**{g.get('category')}:** " + ", ".join(g.get("items", [])))
        if r.get("hydration_tip"):
            st.markdown(f"💧 {r['hydration_tip']}")
        bullets("Guidance", r.get("guidance"), "🧭")
        st.caption("General wellness guidance. Consult a doctor or dietitian for medical conditions.")
        raw_expander({k: v for k, v in r.items() if not k.startswith("_")})

# --------------------------------------------------------------------------- #
# Multimodal
# --------------------------------------------------------------------------- #
with tabs[7]:
    st.subheader("Multimodal analysis: image + voice + text")
    st.caption("Combine a photo, a spoken question and typed notes into one actionable report.")
    img = image_input("mm")
    c1, c2 = st.columns(2)
    with c1:
        mm_text = st.text_area("Text / question", placeholder="Is this okay to eat for dinner? I'm trying to cut carbs.", key="mm_text")
    with c2:
        rec = st.audio_input("Record a voice question", key="mm_rec")
        up_audio = st.file_uploader("...or upload audio", type=["wav", "mp3", "m4a", "ogg", "flac", "aac"], key="mm_audio")
    mm_goal = st.selectbox(
        "Goal", ["maintain", "lose_weight", "gain_muscle", "eat_healthier"], key="mm_goal",
        format_func=lambda g: g.replace("_", " ").title(),
    )
    audio_file = rec or up_audio

    def do_multimodal():
        audio = None
        if audio_file:
            audio = (audio_file.getvalue(), audio_file.type or "audio/wav")
        features = freshness_features(img) if img else None
        prompt = f"""Combine all inputs into ONE actionable food report.
User text: "{mm_text.strip() or 'none'}"
Spoken audio provided: {bool(audio)} (listen to it and treat what is said as part of the user's request; include a short transcript)
Goal: {mm_goal}
Image provided: {bool(img)}
CV freshness metrics (if image): {json.dumps(features) if features else 'n/a'}

Return JSON:
{{
  "headline": str,
  "transcript": str|null,
  "recognition": {{"dishes": [str], "confidence": float}},
  "nutrition": {{"calories_kcal": number, "protein_g": number, "carbs_g": number, "fat_g": number, "health_score": integer}},
  "freshness": {{"classification": "Fresh"|"Moderately Fresh"|"Spoiled"|"Not applicable", "notes": str}},
  "alerts": [{{"severity": "info"|"warning"|"critical", "message": str}}],
  "safety_recommendations": [str],
  "personalized_guidance": [str],
  "answer_to_user_question": str|null,
  "suggested_next_actions": [str]
}}"""
        res = ask_llm(prompt, [img] if img else None, audio=audio, max_tokens=2500)
        record("multimodal", {"headline": res.get("headline")})
        return res

    can_go = bool(img or audio_file or mm_text.strip())
    if st.button("Generate report", type="primary", disabled=not can_go, key="mm_btn"):
        run("Combining image, audio and text...", do_multimodal, "res_mm")

    r = st.session_state.get("res_mm")
    if r:
        st.markdown(f"### {r.get('headline', 'Your food report')}")
        if r.get("transcript"):
            st.caption(f"🎙️ Heard: {r['transcript']}")
        if r.get("answer_to_user_question"):
            st.success(r["answer_to_user_question"])
        for a in r.get("alerts", []):
            severity_box(a.get("severity", "info"), a.get("message", ""))
        n = r.get("nutrition", {})
        if n:
            c = st.columns(5)
            c[0].metric("Calories", n.get("calories_kcal", "–"))
            c[1].metric("Protein", f"{n.get('protein_g', '–')} g")
            c[2].metric("Carbs", f"{n.get('carbs_g', '–')} g")
            c[3].metric("Fat", f"{n.get('fat_g', '–')} g")
            c[4].metric("Health score", n.get("health_score", "–"))
        fr = r.get("freshness", {})
        if fr:
            st.write(f"**Freshness:** {fr.get('classification', '–')}. {fr.get('notes', '')}")
        rec_d = r.get("recognition", {})
        if rec_d.get("dishes"):
            st.write("**Detected:** " + ", ".join(rec_d["dishes"]))
        bullets("Safety recommendations", r.get("safety_recommendations"), "🛡️")
        bullets("Personalized guidance", r.get("personalized_guidance"), "🧭")
        bullets("Suggested next actions", r.get("suggested_next_actions"), "➡️")
        raw_expander(r)
