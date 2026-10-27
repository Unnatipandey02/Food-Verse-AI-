"""
Food Verse AI - Multimodal Food Intelligence Platform (FastAPI backend)

Capabilities (one service section per feature):
  1. Food recognition            POST /api/food/recognize
  2. Nutrition analysis          POST /api/nutrition/analyze
  3. Freshness detection         POST /api/freshness/detect
  4. Restaurant hygiene (video)  POST /api/hygiene/analyze
  5. Smart cooking assistant     POST /api/cooking/assist
  6. Personalized meal plans     POST /api/mealplan/generate
  7. Audio transcription         POST /api/audio/transcribe
  8. Unified multimodal analysis POST /api/multimodal/analyze
  9. Dashboard insights          GET  /api/insights/dashboard

Reasoning engine: Anthropic Claude on Amazon Bedrock (vision + text) via boto3.
Computer vision:  OpenCV / NumPy (local freshness heuristics + video frame sampling).
Audio:            optional faster-whisper (local speech-to-text).

Run:
    uvicorn app:app --reload --port 8000
Docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
import uuid
from collections import deque
from typing import Any, Deque, Dict, List, Literal, Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
# Any Bedrock model that supports the Converse API with image input.
# Use an inference-profile ID (e.g. "us.anthropic.claude-...") if your region requires it.
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"
)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "8"))
MAX_VIDEO_MB = int(os.getenv("MAX_VIDEO_MB", "100"))
HYGIENE_FRAMES = min(int(os.getenv("HYGIENE_FRAMES", "8")), 16)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("foodverse")

app = FastAPI(
    title="Food Verse AI",
    description="Multimodal food intelligence: recognition, nutrition, freshness, hygiene, cooking and meal planning.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory analysis history that feeds the dashboard (swap for DynamoDB/Postgres in production).
HISTORY: Deque[Dict[str, Any]] = deque(maxlen=500)


def record(kind: str, summary: Dict[str, Any]) -> None:
    HISTORY.append({"id": str(uuid.uuid4()), "type": kind, "timestamp": time.time(), "summary": summary})


# --------------------------------------------------------------------------- #
# LLM reasoning engine (Amazon Bedrock Converse API)
# --------------------------------------------------------------------------- #
_bedrock_client = None


def bedrock():
    global _bedrock_client
    if _bedrock_client is None:
        try:
            import boto3

            _bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(503, f"AWS Bedrock client unavailable: {exc}")
    return _bedrock_client


def _image_format(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    raise HTTPException(415, "Unsupported image type. Use JPEG, PNG, WEBP or GIF.")


def parse_json(text: str) -> Dict[str, Any]:
    """Extract a JSON object from an LLM reply, tolerating code fences and chatter."""
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
        raise HTTPException(502, "The AI model returned a response that was not valid JSON.")


def ask_llm(
    system: str,
    prompt: str,
    images: Optional[List[bytes]] = None,
    max_tokens: int = 2000,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Send text (+ optional images) to the model and return parsed JSON."""
    content: List[Dict[str, Any]] = []
    for img in images or []:
        content.append({"image": {"format": _image_format(img), "source": {"bytes": img}}})
    content.append({"text": prompt + "\n\nRespond with a single valid JSON object only. No markdown, no commentary."})

    try:
        resp = bedrock().converse(
            modelId=BEDROCK_MODEL_ID,
            system=[{"text": system}],
            messages=[{"role": "user", "content": content}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Bedrock call failed")
        raise HTTPException(
            503,
            "AI reasoning engine unavailable. Check AWS credentials (`aws configure`), "
            f"region and Bedrock model access. Details: {exc}",
        )
    text = "".join(p.get("text", "") for p in resp["output"]["message"]["content"])
    return parse_json(text)


SYSTEM_BASE = (
    "You are Food Verse AI, a food intelligence assistant for nutrition, food safety, "
    "hygiene and cooking. Be precise and honest about uncertainty. Nutrition values are "
    "estimates, not medical advice. Never claim certainty about food safety from a photo alone."
)


# --------------------------------------------------------------------------- #
# Helpers: file handling
# --------------------------------------------------------------------------- #
async def read_image(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty image upload.")
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(413, f"Image too large (max {MAX_IMAGE_MB} MB).")
    _image_format(data)  # validates type
    return data


def shrink_image(data: bytes, max_side: int = 1280) -> bytes:
    """Downscale large photos so requests stay fast and within model limits."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image.")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise HTTPException(500, "Image re-encoding failed.")
    return buf.tobytes()


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class UserProfile(BaseModel):
    age: Optional[int] = Field(None, ge=1, le=120)
    sex: Optional[Literal["male", "female", "other"]] = None
    height_cm: Optional[float] = Field(None, gt=0)
    weight_kg: Optional[float] = Field(None, gt=0)
    activity_level: Literal["sedentary", "light", "moderate", "active", "very_active"] = "moderate"
    goal: Literal["lose_weight", "maintain", "gain_muscle", "eat_healthier"] = "maintain"
    diet: str = Field("none", description="e.g. vegetarian, vegan, keto, jain, halal, none")
    allergies: List[str] = []
    dislikes: List[str] = []
    cuisine_preferences: List[str] = []
    medical_notes: Optional[str] = Field(None, description="e.g. diabetes, hypertension (informational only)")


class CookingRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Dish name or free-text request")
    ingredients_on_hand: List[str] = []
    servings: int = Field(2, ge=1, le=50)
    max_time_minutes: Optional[int] = Field(None, ge=5, le=600)
    skill_level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    diet: str = "none"
    allergies: List[str] = []
    current_step: Optional[int] = Field(None, ge=1, description="Step the user is on, for contextual guidance")
    question: Optional[str] = Field(None, description="Follow-up question about the cooking process")


class MealPlanRequest(BaseModel):
    profile: UserProfile
    days: int = Field(7, ge=1, le=14)
    meals_per_day: int = Field(3, ge=2, le=6)
    budget_level: Literal["low", "medium", "high"] = "medium"
    feedback: Optional[str] = Field(None, description="Adaptive feedback, e.g. 'too much rice, more protein'")
    previous_plan_summary: Optional[str] = None


# --------------------------------------------------------------------------- #
# Health / meta
# --------------------------------------------------------------------------- #
@app.get("/")
def root():
    return {"name": "Food Verse AI", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "model": BEDROCK_MODEL_ID, "region": AWS_REGION}


# --------------------------------------------------------------------------- #
# 1. Food recognition
# --------------------------------------------------------------------------- #
@app.post("/api/food/recognize", tags=["Food Recognition"])
async def recognize_food(image: UploadFile = File(...), context: str = Form("")):
    img = shrink_image(await read_image(image))
    prompt = f"""Identify every distinct dish and food item visible in the image.
Optional user context: "{context}"

Return JSON:
{{
  "dishes": [{{"name": str, "cuisine": str, "confidence": float(0-1), "description": str}}],
  "ingredients_visible": [str],
  "estimated_portion": str,
  "is_food": bool,
  "notes": str
}}
If the image contains no food, set is_food=false and dishes=[]."""
    result = ask_llm(SYSTEM_BASE, prompt, [img])
    record("recognition", {"dishes": [d.get("name") for d in result.get("dishes", [])]})
    return result


# --------------------------------------------------------------------------- #
# 2. Nutrition analysis
# --------------------------------------------------------------------------- #
@app.post("/api/nutrition/analyze", tags=["Nutrition"])
async def analyze_nutrition(
    image: Optional[UploadFile] = File(None),
    description: str = Form("", description="Text description, e.g. '2 rotis with dal and sabzi'"),
    diet_goal: str = Form("maintain"),
):
    if image is None and not description.strip():
        raise HTTPException(400, "Provide a food image, a text description, or both.")
    imgs = [shrink_image(await read_image(image))] if image else []
    prompt = f"""Estimate the nutrition of the food {'in the image ' if imgs else ''}{('and described as: ' + description) if description else ''}.
User goal: {diet_goal}.

Return JSON:
{{
  "items": [{{
    "name": str, "portion": str,
    "calories_kcal": number, "protein_g": number, "carbs_g": number, "fat_g": number,
    "fiber_g": number, "sugar_g": number, "sodium_mg": number
  }}],
  "totals": {{"calories_kcal": number, "protein_g": number, "carbs_g": number, "fat_g": number, "fiber_g": number, "sugar_g": number, "sodium_mg": number}},
  "health_score": integer(0-100),
  "health_score_reasoning": str,
  "macro_split_percent": {{"protein": number, "carbs": number, "fat": number}},
  "positives": [str],
  "concerns": [str],
  "healthier_swaps": [str],
  "goal_fit": str,
  "confidence": float(0-1),
  "disclaimer": "Estimates only; not medical advice."
}}"""
    result = ask_llm(SYSTEM_BASE, prompt, imgs)
    record(
        "nutrition",
        {"calories": result.get("totals", {}).get("calories_kcal"), "health_score": result.get("health_score")},
    )
    return result


# --------------------------------------------------------------------------- #
# 3. Freshness detection (local CV heuristics + LLM visual verification)
# --------------------------------------------------------------------------- #
def freshness_features(image_bytes: bytes) -> Dict[str, float]:
    """Compute scientific-ish visual indicators of spoilage using OpenCV."""
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image.")
    img = cv2.resize(img, (400, int(400 * img.shape[0] / img.shape[1])))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(hsv)
    total = float(h.size)

    # Dark / rotten patches (very low brightness)
    dark_ratio = float(np.count_nonzero(v < 50)) / total
    # Brown discoloration: orange-brown hues with low-mid value
    brown_mask = (h >= 8) & (h <= 25) & (s > 60) & (v >= 40) & (v < 130)
    brown_ratio = float(np.count_nonzero(brown_mask)) / total
    # Mold-like: whitish/grey-blue or green fuzz with low saturation
    mold_white = (s < 40) & (v > 170)
    mold_green_blue = (h >= 35) & (h <= 110) & (s > 25) & (s < 110) & (v < 160)
    mold_ratio = float(np.count_nonzero(mold_white | mold_green_blue)) / total
    # Texture irregularity (wrinkles, slime, patches)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Vividness
    mean_sat = float(s.mean())
    # Colour uniformity
    hue_std = float(h.std())

    return {
        "dark_patch_ratio": round(dark_ratio, 4),
        "brown_discoloration_ratio": round(brown_ratio, 4),
        "mold_like_ratio": round(mold_ratio, 4),
        "texture_variance": round(lap_var, 2),
        "mean_saturation": round(mean_sat, 2),
        "hue_std": round(hue_std, 2),
    }


def heuristic_freshness(f: Dict[str, float]) -> Dict[str, Any]:
    penalty = (
        f["dark_patch_ratio"] * 160
        + f["brown_discoloration_ratio"] * 110
        + f["mold_like_ratio"] * 60
        + min(f["texture_variance"] / 1500, 1) * 15
        + (10 if f["mean_saturation"] < 40 else 0)
    )
    score = int(max(0, min(100, 100 - penalty)))
    label = "Fresh" if score >= 72 else "Moderately Fresh" if score >= 45 else "Spoiled"
    return {"score": score, "label": label}


@app.post("/api/freshness/detect", tags=["Freshness"])
async def detect_freshness(image: UploadFile = File(...), food_type: str = Form("")):
    raw = await read_image(image)
    img = shrink_image(raw)
    features = freshness_features(img)
    heuristic = heuristic_freshness(features)

    llm: Optional[Dict[str, Any]] = None
    llm_error: Optional[str] = None
    try:
        prompt = f"""Assess the freshness of the food in this image (food type hint: "{food_type or 'unknown'}").
Look at color, texture, discoloration, wilting, bruising, mold, slime and surface patterns.
Objective image measurements from a computer-vision pre-check: {json.dumps(features)}
Pre-check heuristic result: {json.dumps(heuristic)}

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
        llm = ask_llm(SYSTEM_BASE, prompt, [img])
    except HTTPException as exc:
        llm_error = str(exc.detail)

    if llm:
        # Blend the LLM visual judgement with the measurable CV score.
        blended = int(round(0.6 * float(llm.get("freshness_score", heuristic["score"])) + 0.4 * heuristic["score"]))
        label = "Fresh" if blended >= 72 else "Moderately Fresh" if blended >= 45 else "Spoiled"
        result = {
            "classification": label,
            "freshness_score": blended,
            "cv_metrics": features,
            "cv_heuristic": heuristic,
            "ai_assessment": llm,
            "warning": "Visual analysis cannot detect all spoilage. When in doubt, throw it out.",
        }
    else:
        result = {
            "classification": heuristic["label"],
            "freshness_score": heuristic["score"],
            "cv_metrics": features,
            "cv_heuristic": heuristic,
            "ai_assessment": None,
            "note": f"AI reasoning unavailable; showing local computer-vision result only. ({llm_error})",
            "warning": "Visual analysis cannot detect all spoilage. When in doubt, throw it out.",
        }
    record("freshness", {"classification": result["classification"], "score": result["freshness_score"]})
    return result


# --------------------------------------------------------------------------- #
# 4. Restaurant hygiene monitoring (video)
# --------------------------------------------------------------------------- #
def sample_video_frames(path: str, n: int) -> List[bytes]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(400, "Could not open video. Try MP4/MOV/AVI/WEBM.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: List[bytes] = []
    if total <= 0:
        # Fallback for streams with unknown length: read sequentially, keep every k-th.
        buf = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            buf.append(fr)
        cap.release()
        if not buf:
            raise HTTPException(400, "Video contains no readable frames.")
        idxs = np.linspace(0, len(buf) - 1, min(n, len(buf))).astype(int)
        sel = [buf[i] for i in idxs]
    else:
        idxs = np.linspace(0, total - 1, min(n, total)).astype(int)
        sel = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok:
                sel.append(fr)
        cap.release()
    for fr in sel:
        h, w = fr.shape[:2]
        scale = 1024 / max(h, w)
        if scale < 1:
            fr = cv2.resize(fr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            frames.append(buf.tobytes())
    if not frames:
        raise HTTPException(400, "Could not extract frames from video.")
    log.info("Sampled %d frames (fps=%.1f, total=%d)", len(frames), fps, total)
    return frames


@app.post("/api/hygiene/analyze", tags=["Hygiene"])
async def analyze_hygiene(video: UploadFile = File(...), location: str = Form("Kitchen")):
    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        size = 0
        while chunk := await video.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_VIDEO_MB * 1024 * 1024:
                raise HTTPException(413, f"Video too large (max {MAX_VIDEO_MB} MB).")
            tmp.write(chunk)
        tmp.close()
        frames = sample_video_frames(tmp.name, HYGIENE_FRAMES)
    finally:
        try:
            tmp.close()
            os.unlink(tmp.name)
        except OSError:
            pass

    prompt = f"""These {len(frames)} frames are sampled in time order from a video of: {location}.
Evaluate food-safety hygiene practices of the visible staff and environment.

Check each indicator and report status "compliant" | "violation" | "not_visible":
- glove_usage, hair_cap_or_hairnet, apron_or_clean_uniform, hand_washing_or_sanitizing,
  raw_vs_cooked_separation (cross-contamination), food_handling_technique (bare-hand contact with ready-to-eat food, touching face/phone),
  surface_and_equipment_cleanliness, floor_and_waste_management, pest_signs, food_storage_and_labelling

Return JSON:
{{
  "overall_hygiene_score": integer(0-100),
  "risk_level": "low" | "medium" | "high",
  "indicators": [{{"name": str, "status": "compliant"|"violation"|"not_visible", "evidence": str, "frames": [int]}}],
  "alerts": [{{"severity": "info"|"warning"|"critical", "message": str, "frame": int|null}}],
  "staff_observed": integer,
  "recommendations": [str],
  "summary": str,
  "confidence": float(0-1),
  "limitations": "Based on sampled frames only; not a substitute for official inspection."
}}
Only report violations you can actually see; use "not_visible" instead of guessing."""
    result = ask_llm(
        SYSTEM_BASE + " You are acting as a food-safety hygiene auditor (HACCP-aware).",
        prompt,
        frames,
        max_tokens=2500,
    )
    result["frames_analyzed"] = len(frames)
    record("hygiene", {"score": result.get("overall_hygiene_score"), "risk": result.get("risk_level")})
    return result


# --------------------------------------------------------------------------- #
# 5. Smart cooking assistance
# --------------------------------------------------------------------------- #
@app.post("/api/cooking/assist", tags=["Cooking"])
def cooking_assist(req: CookingRequest):
    prompt = f"""User request: "{req.query}"
Servings: {req.servings}; skill: {req.skill_level}; diet: {req.diet}; allergies: {req.allergies or 'none'}
Ingredients on hand: {req.ingredients_on_hand or 'not specified'}
Max time: {req.max_time_minutes or 'no limit'} minutes
Currently on step: {req.current_step or 'not started'}
Follow-up question: {req.question or 'none'}

Provide a complete recipe plus contextual guidance. Never include listed allergens; respect the diet.
Return JSON:
{{
  "dish": str,
  "prep_time_min": integer, "cook_time_min": integer, "difficulty": str,
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
    result = ask_llm(SYSTEM_BASE, prompt, max_tokens=3000, temperature=0.5)
    record("cooking", {"dish": result.get("dish")})
    return result


# --------------------------------------------------------------------------- #
# 6. Personalized meal plans
# --------------------------------------------------------------------------- #
def estimate_targets(p: UserProfile) -> Dict[str, Any]:
    """Mifflin-St Jeor BMR -> TDEE -> goal-adjusted calories and macro targets."""
    if not (p.age and p.height_cm and p.weight_kg):
        return {"note": "Provide age, height and weight for calculated targets; the AI will estimate sensibly."}
    s = 5 if p.sex == "male" else -161 if p.sex == "female" else -78
    bmr = 10 * p.weight_kg + 6.25 * p.height_cm - 5 * p.age + s
    mult = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "very_active": 1.9}[p.activity_level]
    tdee = bmr * mult
    adj = {"lose_weight": -400, "maintain": 0, "gain_muscle": 300, "eat_healthier": 0}[p.goal]
    kcal = max(1200, tdee + adj)
    protein_per_kg = 2.0 if p.goal == "gain_muscle" else 1.6 if p.goal == "lose_weight" else 1.2
    protein = p.weight_kg * protein_per_kg
    fat = kcal * 0.28 / 9
    carbs = max(0.0, (kcal - protein * 4 - fat * 9) / 4)
    bmi = p.weight_kg / ((p.height_cm / 100) ** 2)
    return {
        "bmi": round(bmi, 1),
        "bmr_kcal": round(bmr),
        "tdee_kcal": round(tdee),
        "target_calories_kcal": round(kcal),
        "target_protein_g": round(protein),
        "target_carbs_g": round(carbs),
        "target_fat_g": round(fat),
    }


@app.post("/api/mealplan/generate", tags=["Meal Plan"])
def generate_meal_plan(req: MealPlanRequest):
    targets = estimate_targets(req.profile)
    prompt = f"""Create a {req.days}-day meal plan with {req.meals_per_day} meals per day.
Profile: {req.profile.model_dump_json()}
Calculated targets: {json.dumps(targets)}
Budget: {req.budget_level}
Adaptive feedback from user: {req.feedback or 'none'}
Previous plan summary: {req.previous_plan_summary or 'none'}

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
  "guidance": [str],
  "disclaimer": "General wellness guidance; consult a doctor or dietitian for medical conditions."
}}"""
    result = ask_llm(SYSTEM_BASE, prompt, max_tokens=6000, temperature=0.6)
    result["calculated_targets"] = targets
    record("mealplan", {"days": req.days, "goal": req.profile.goal})
    return result


# --------------------------------------------------------------------------- #
# 7. Audio processing (optional local Whisper)
# --------------------------------------------------------------------------- #
_whisper = None


def transcribe_file(path: str) -> str:
    global _whisper
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise HTTPException(
            501, "Audio support not installed. Uncomment `faster-whisper` in requirements.txt and install ffmpeg."
        )
    if _whisper is None:
        _whisper = WhisperModel(WHISPER_MODEL, compute_type="int8")
    segments, _ = _whisper.transcribe(path, vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


async def transcribe_upload(audio: UploadFile) -> str:
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        path = tmp.name
    try:
        return transcribe_file(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@app.post("/api/audio/transcribe", tags=["Audio"])
async def audio_transcribe(audio: UploadFile = File(...)):
    text = await transcribe_upload(audio)
    return {"transcript": text}


# --------------------------------------------------------------------------- #
# 8. Unified multimodal analysis (image + text + audio -> one insight report)
# --------------------------------------------------------------------------- #
@app.post("/api/multimodal/analyze", tags=["Multimodal"])
async def multimodal_analyze(
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    text: str = Form(""),
    diet_goal: str = Form("maintain"),
):
    transcript = await transcribe_upload(audio) if audio else ""
    user_text = " ".join(t for t in (text.strip(), transcript) if t)
    if image is None and not user_text:
        raise HTTPException(400, "Provide at least an image, audio, or text.")
    imgs = [shrink_image(await read_image(image))] if image else []
    features = freshness_features(imgs[0]) if imgs else None

    prompt = f"""Combine all inputs into ONE actionable food report.
User text/voice: "{user_text or 'none'}"
Goal: {diet_goal}
Image provided: {bool(imgs)}
CV freshness metrics (if image): {json.dumps(features) if features else 'n/a'}

Return JSON:
{{
  "headline": str,
  "recognition": {{"dishes": [str], "confidence": float}},
  "nutrition": {{"calories_kcal": number, "protein_g": number, "carbs_g": number, "fat_g": number, "health_score": integer}},
  "freshness": {{"classification": "Fresh"|"Moderately Fresh"|"Spoiled"|"Not applicable", "notes": str}},
  "alerts": [{{"severity": "info"|"warning"|"critical", "message": str}}],
  "safety_recommendations": [str],
  "personalized_guidance": [str],
  "answer_to_user_question": str|null,
  "suggested_next_actions": [str]
}}"""
    result = ask_llm(SYSTEM_BASE, prompt, imgs, max_tokens=2500)
    result["transcript"] = transcript or None
    record("multimodal", {"headline": result.get("headline")})
    return result


# --------------------------------------------------------------------------- #
# 9. Dashboard insights
# --------------------------------------------------------------------------- #
@app.get("/api/insights/dashboard", tags=["Insights"])
def dashboard(limit: int = 50):
    items = list(HISTORY)[-limit:]
    by_type: Dict[str, int] = {}
    for it in items:
        by_type[it["type"]] = by_type.get(it["type"], 0) + 1

    cals = [i["summary"]["calories"] for i in items if i["type"] == "nutrition" and i["summary"].get("calories")]
    health = [i["summary"]["health_score"] for i in items if i["type"] == "nutrition" and i["summary"].get("health_score")]
    fresh = [i["summary"]["classification"] for i in items if i["type"] == "freshness"]
    hyg = [i["summary"]["score"] for i in items if i["type"] == "hygiene" and i["summary"].get("score") is not None]

    alerts: List[Dict[str, str]] = []
    if fresh.count("Spoiled"):
        alerts.append({"severity": "warning", "message": f"{fresh.count('Spoiled')} item(s) were classified as spoiled."})
    if hyg and min(hyg) < 60:
        alerts.append({"severity": "critical", "message": f"A hygiene audit scored as low as {min(hyg)}/100."})
    if health and sum(health) / len(health) < 50:
        alerts.append({"severity": "info", "message": "Average meal health score is below 50; consider lighter, fresher meals."})

    avg = lambda xs: round(sum(xs) / len(xs), 1) if xs else None
    return {
        "total_analyses": len(items),
        "analyses_by_type": by_type,
        "avg_calories_per_meal": avg(cals),
        "avg_health_score": avg(health),
        "freshness_breakdown": {k: fresh.count(k) for k in ("Fresh", "Moderately Fresh", "Spoiled")},
        "avg_hygiene_score": avg(hyg),
        "alerts": alerts,
        "recent": items[-10:][::-1],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
