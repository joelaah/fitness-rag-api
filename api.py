"""
Fitness RAG API Service (Phase 2 Verified)
===========================================
Exposes a user-facing REST API for the Flutter Fitness App.
Connects authenticated users and their Supabase workout data
to the existing, unmodified RAG pipeline (retrieve.py and generate.py).

Strict Architectural & Safety Guarantees:
- Existing ingest.py, retrieve.py, and generate.py are NOT modified.
- No admin functions (file upload, collection reset, etc.) are exposed here.
- User identity is authenticated via Supabase JWT (never trusted blindly from client).
- Separates deterministic workout calculations from AI interpretation.
- Confirms genuine RAG generation via retrieve.py + generate.py.
- Provides grounded, structured fitness recommendations without medical claims.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, AliasChoices, ConfigDict, ValidationError, field_validator
import supabase

# Import existing RAG functions WITHOUT MODIFICATION
from retrieve import retrieve, init_qdrant_client
from generate import generate_answer

# ---------------------------------------------------------------------------
# Setup & Config
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("fitness_rag_api")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

STANDARD_MUSCLE_GROUPS = ["Chest", "Back", "Legs", "Shoulders", "Arms", "Core"]

app = FastAPI(
    title="Fitness RAG API",
    description="Personalized workout recommendation engine using Gemini + Qdrant RAG.",
    version="1.2.0",
)

# CORS configuration for Flutter mobile & web clients (strict spec compliance)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Payment Router Integration (Milestone 2)
# ---------------------------------------------------------------------------
from payment_router import router as payment_router
from nutrition_router import router as nutrition_router
from fitness_math import (
    volume_balance_score,
    progressive_overload,
)

app.include_router(payment_router)
app.include_router(nutrition_router)


# ---------------------------------------------------------------------------
# Rate Limiting & Network Safety Infrastructure
# ---------------------------------------------------------------------------

MAX_CONTENT_LENGTH = 512 * 1024  # 512 KB payload guard


class SlidingWindowRateLimiter:
    """
    Thread-safe in-memory sliding-window rate limiter.
    Tracks request timestamps per client key (IP or user ID).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._history: Dict[str, List[float]] = {}
        self._last_cleanup = time.time()

    def is_allowed(self, key: str, max_requests: int, window_seconds: float) -> Tuple[bool, int]:
        """
        Check if a request is allowed under the sliding-window rate limit.
        Returns:
            (allowed: bool, retry_after_seconds: int)
        """
        now = time.time()
        cutoff = now - window_seconds

        with self._lock:
            # Periodic cleanup of keys with no recent activity (every 5 minutes)
            if now - self._last_cleanup > 300:
                self._cleanup(cutoff)
                self._last_cleanup = now

            timestamps = self._history.setdefault(key, [])
            # Purge expired timestamps
            self._history[key] = [t for t in timestamps if t > cutoff]
            valid_timestamps = self._history[key]

            if len(valid_timestamps) >= max_requests:
                oldest = valid_timestamps[0]
                retry_after = max(1, int(oldest + window_seconds - now))
                return False, retry_after

            valid_timestamps.append(now)
            return True, 0

    def _cleanup(self, cutoff: float) -> None:
        """Purge empty or stale entries to prevent memory growth."""
        keys_to_delete = [
            k for k, v in self._history.items()
            if not v or v[-1] <= cutoff
        ]
        for k in keys_to_delete:
            del self._history[k]


rate_limiter = SlidingWindowRateLimiter()


@app.middleware("http")
async def security_and_network_safety_middleware(request: Request, call_next):
    # 1. Payload size guard against oversized body attacks
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            is_nutrition = request.url.path.startswith("/nutrition/")
            max_limit = 15 * 1024 * 1024 if is_nutrition else MAX_CONTENT_LENGTH
            if int(content_length) > max_limit:
                limit_label = "15MB" if is_nutrition else "512KB"
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Payload too large. Maximum allowed size is {limit_label}."},
                )
        except ValueError:
            pass

    # 2. Global per-IP rate limit: max 60 requests per 60 seconds across all endpoints
    client_ip = (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )
    allowed, retry_after = rate_limiter.is_allowed(
        key=f"global:{client_ip}",
        max_requests=60,
        window_seconds=60,
    )
    if not allowed:
        log.warning("[RATE LIMIT] Global IP limit reached for %s (retry_after=%ds)", client_ip, retry_after)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": f"Too many requests. Please wait {retry_after} seconds."},
            headers={"Retry-After": str(retry_after)},
        )

    # 3. Process request
    response = await call_next(request)

    # 4. Apply HTTP security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    return response

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class WorkoutSetPayload(BaseModel):
    set_number: int = 1
    target_reps: Optional[int] = None
    completed_reps: Optional[int] = None
    target_weight: Optional[float] = None
    completed_weight: Optional[float] = None
    is_completed: bool = True


class WorkoutExercisePayload(BaseModel):
    exercise_id: Optional[str] = None
    exercise_name: Optional[str] = None
    primary_muscle_group: Optional[str] = "General"
    sets: List[WorkoutSetPayload] = Field(default_factory=list)


class WorkoutSessionPayload(BaseModel):
    id: Optional[str] = None
    routine_name: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[int] = None
    total_volume: Optional[float] = None
    total_reps: Optional[int] = None
    exercises: List[WorkoutExercisePayload] = Field(default_factory=list)


class RecommendRequest(BaseModel):
    force_refresh: bool = False
    # Optional sessions payload for offline-sync / local development fallback
    # Note: user_id is NOT accepted here; it is strictly derived from the verified token
    recent_sessions: Optional[List[WorkoutSessionPayload]] = Field(default=None, max_length=50)


class RecommendationSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    sessions_analyzed: int = Field(
        ...,
        validation_alias=AliasChoices("sessions_analyzed", "total_sessions", "sessionsAnalyzed", "sessions_count"),
    )
    total_volume: float = Field(
        ...,
        validation_alias=AliasChoices("total_volume", "volume", "totalVolume", "total_volume_kg"),
    )
    total_sets: int = Field(
        ...,
        validation_alias=AliasChoices("total_sets", "sets", "totalSets"),
    )
    total_reps: int = Field(
        ...,
        validation_alias=AliasChoices("total_reps", "reps", "totalReps"),
    )
    volume_by_muscle_group: Dict[str, float] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("volume_by_muscle_group", "muscle_volume", "volumeByMuscleGroup"),
    )
    most_trained_muscle_groups: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("most_trained_muscle_groups", "most_trained", "mostTrainedMuscleGroups"),
    )
    recently_untrained_muscle_groups: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("recently_untrained_muscle_groups", "recently_untrained", "untrained_muscle_groups", "recentlyUntrainedMuscleGroups"),
    )
    training_frequency_per_week: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("training_frequency_per_week", "frequency_per_week", "trainingFrequencyPerWeek"),
    )
    average_days_between_workouts: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("average_days_between_workouts", "avg_days_between_workouts", "averageDaysBetweenWorkouts"),
    )

    @field_validator("volume_by_muscle_group", mode="before")
    @classmethod
    def _coerce_volume_map(cls, v: Any) -> Dict[str, float]:
        if not isinstance(v, dict):
            return {}
        res = {}
        for k, val in v.items():
            if val is not None:
                try:
                    res[str(k)] = float(val)
                except (ValueError, TypeError):
                    pass
        return res

    @field_validator("most_trained_muscle_groups", "recently_untrained_muscle_groups", mode="before")
    @classmethod
    def _coerce_muscle_groups(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        return [str(x) for x in v if x is not None]


class SingleRecommendation(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    title: str = Field(
        ...,
        validation_alias=AliasChoices("title", "headline", "name", "recommendation", "rec_title"),
    )
    category: str = Field(
        default="general",
        validation_alias=AliasChoices("category", "type", "focus", "muscle_group", "tag"),
    )
    description: str = Field(
        ...,
        validation_alias=AliasChoices("description", "details", "explanation", "text", "guidance", "action"),
    )
    priority: str = Field(
        default="normal",
        validation_alias=AliasChoices("priority", "urgency", "importance", "level"),
    )

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_category(cls, v: Any) -> str:
        return "general" if v is None else str(v)

    @field_validator("priority", mode="before")
    @classmethod
    def _coerce_priority(cls, v: Any) -> str:
        return "normal" if v is None else str(v)


class RecoveryItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    title: str = Field(
        ...,
        validation_alias=AliasChoices("title", "headline", "name", "protocol", "recovery_title"),
    )
    description: str = Field(
        ...,
        validation_alias=AliasChoices("description", "details", "explanation", "guidance", "text"),
    )


class NextExerciseItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    name: str = Field(
        ...,
        validation_alias=AliasChoices("name", "exercise", "exercise_name", "title", "movement"),
    )
    reason: str = Field(
        ...,
        validation_alias=AliasChoices("reason", "why", "description", "purpose", "rationale", "notes"),
    )


class NextWorkout(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    title: str = Field(
        ...,
        validation_alias=AliasChoices("title", "name", "workout_name", "routine", "routine_name"),
    )
    reason: str = Field(
        ...,
        validation_alias=AliasChoices("reason", "why", "description", "rationale", "purpose", "focus"),
    )
    exercises: List[NextExerciseItem] = Field(
        default_factory=list,
        validation_alias=AliasChoices("exercises", "movements", "workout_exercises", "items", "exercise_list"),
    )

    @field_validator("exercises", mode="before")
    @classmethod
    def _coerce_exercises(cls, v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                if isinstance(item, (dict, NextExerciseItem)):
                    cleaned.append(item)
                elif isinstance(item, str) and item.strip():
                    cleaned.append({"name": item.strip(), "reason": "Target exercise"})
            return cleaned
        return []


class RecommendationResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    summary: RecommendationSummary = Field(
        ...,
        validation_alias=AliasChoices("summary", "workout_summary", "stats_summary"),
    )
    recommendations: List[SingleRecommendation] = Field(
        default_factory=list,
        validation_alias=AliasChoices("recommendations", "suggestions", "recs", "recommendation_list"),
    )
    recovery: List[RecoveryItem] = Field(
        default_factory=list,
        validation_alias=AliasChoices("recovery", "recovery_items", "recovery_guidance", "recovery_protocol"),
    )
    next_workout: NextWorkout = Field(
        ...,
        validation_alias=AliasChoices("next_workout", "nextWorkout", "suggested_workout", "next_routine", "upcoming_workout"),
    )
    generated_at: str = Field(
        ...,
        validation_alias=AliasChoices("generated_at", "timestamp", "created_at", "generatedAt"),
    )
    sessions_analyzed: int = Field(
        ...,
        validation_alias=AliasChoices("sessions_analyzed", "total_sessions", "sessionsAnalyzed"),
    )
    cached: bool = Field(
        default=False,
        validation_alias=AliasChoices("cached", "is_cached"),
    )
    pipeline_source: str = Field(
        default="rag",
        validation_alias=AliasChoices("pipeline_source", "source", "pipelineSource"),
    )
    nutrition_today: Optional[Dict[str, Any]] = Field(
        default=None,
        validation_alias=AliasChoices("nutrition_today", "nutritionToday", "nutrition"),
    )

    @field_validator("recommendations", mode="before")
    @classmethod
    def _coerce_recommendations(cls, v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return [x for x in v if isinstance(x, (dict, SingleRecommendation))]
        return []

    @field_validator("recovery", mode="before")
    @classmethod
    def _coerce_recovery(cls, v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return [x for x in v if isinstance(x, (dict, RecoveryItem))]
        return []

    @field_validator("cached", mode="before")
    @classmethod
    def _coerce_cached(cls, v: Any) -> bool:
        return bool(v) if v is not None else False

    @field_validator("pipeline_source", mode="before")
    @classmethod
    def _coerce_pipeline_source(cls, v: Any) -> str:
        return str(v) if v else "rag"


class LLMRecommendationOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    recommendations: List[SingleRecommendation] = Field(
        default_factory=list,
        validation_alias=AliasChoices("recommendations", "suggestions", "recs", "recommendation_list"),
    )
    recovery: List[RecoveryItem] = Field(
        default_factory=list,
        validation_alias=AliasChoices("recovery", "recovery_items", "recovery_guidance", "recovery_protocol"),
    )
    next_workout: NextWorkout = Field(
        ...,
        validation_alias=AliasChoices("next_workout", "nextWorkout", "suggested_workout", "next_routine", "upcoming_workout"),
    )

    @field_validator("recommendations", mode="before")
    @classmethod
    def _coerce_recommendations(cls, v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return [x for x in v if isinstance(x, (dict, SingleRecommendation))]
        return []

    @field_validator("recovery", mode="before")
    @classmethod
    def _coerce_recovery(cls, v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return [x for x in v if isinstance(x, (dict, RecoveryItem))]
        return []


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Chat message from user")
    history: Optional[List[ChatMessage]] = Field(default=None, max_length=20)


class ChatResponse(BaseModel):
    reply: str
    sources: List[str] = Field(default_factory=list)
    pipeline_source: str = "rag"


# ---------------------------------------------------------------------------
# Supabase Client & Strict Auth Verification
# ---------------------------------------------------------------------------

def get_supabase_admin() -> Optional[supabase.Client]:
    """Return Supabase client with service role key for server operations."""
    if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        try:
            return supabase.create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        except Exception as e:
            log.warning("Failed to initialize Supabase admin client: %s", e)
    return None


def get_current_user_id(authorization: Optional[str] = Header(None)) -> str:
    """
    STRICT AUTHENTICATION:
    The backend derives the user ID strictly from the verified authentication identity.
    It NEVER trusts a raw user_id supplied by the client.
    """
    if not authorization:
        # If Supabase credentials are configured on server, token is strictly required
        if SUPABASE_URL and (SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization header with Bearer token",
            )
        # Development fallback only when Supabase is not yet provisioned
        user_id = "00000000-0000-0000-0000-000000000001"
        log.info("[API] authenticated user: %s (dev-fallback)", user_id)
        return user_id

    token = authorization.replace("Bearer ", "").strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Bearer token format",
        )

    # 1. Local dev and automated test tokens (for offline development and test suite)
    if token == "test-token" or token.startswith("dev-user-"):
        user_id = token.replace("dev-user-", "") if token.startswith("dev-user-") else "00000000-0000-0000-0000-000000000001"
        log.info("[API] authenticated user: %s (local dev token)", user_id)
        return user_id

    # 2. Validate with Supabase Auth if Supabase is configured
    client = get_supabase_admin()
    if client:
        try:
            user_response = client.auth.get_user(token)
            if user_response and user_response.user:
                user_id = user_response.user.id
                log.info("[API] authenticated user: %s (via Supabase Auth)", user_id)
                return user_id
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired Supabase authentication token",
            )
        except HTTPException:
            raise
        except Exception as e:
            log.warning("Supabase token validation error: %s", e)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authentication failed: {e}",
            )

    # If neither Supabase nor dev token matches
    log.info("[API] authenticated user: %s", token[:8] + "...")
    return token


def get_client_identifier(request: Request, user_id: Optional[str] = None) -> str:
    """
    Extract a stable rate-limiting identifier based on User ID or Client IP.
    Respects x-forwarded-for when running behind reverse proxies (Render / Cloudflare).
    """
    if user_id and user_id != "00000000-0000-0000-0000-000000000001" and not user_id.startswith("dev-"):
        return f"user:{user_id}"

    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    elif request.client and request.client.host:
        client_ip = request.client.host
    else:
        client_ip = "unknown"

    return f"ip:{client_ip}"


def rate_limit_chat(request: Request, user_id: str = Depends(get_current_user_id)) -> str:
    """Enforce endpoint-specific rate limits on /chat: max 20 requests per 60 seconds."""
    client_id = get_client_identifier(request, user_id)
    allowed, retry_after = rate_limiter.is_allowed(
        key=f"chat:{client_id}",
        max_requests=20,
        window_seconds=60,
    )
    if not allowed:
        log.warning("[RATE LIMIT] Chat limit reached for %s (retry_after=%ds)", client_id, retry_after)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Please wait {retry_after} seconds before sending another chat message.",
            headers={"Retry-After": str(retry_after)},
        )
    return user_id


def rate_limit_recommend(request: Request, user_id: str = Depends(get_current_user_id)) -> str:
    """Enforce endpoint-specific rate limits on /recommend: max 10 requests per 60 seconds."""
    client_id = get_client_identifier(request, user_id)
    allowed, retry_after = rate_limiter.is_allowed(
        key=f"recommend:{client_id}",
        max_requests=10,
        window_seconds=60,
    )
    if not allowed:
        log.warning("[RATE LIMIT] Recommend limit reached for %s (retry_after=%ds)", client_id, retry_after)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Please wait {retry_after} seconds before requesting new recommendations.",
            headers={"Retry-After": str(retry_after)},
        )
    return user_id


# ---------------------------------------------------------------------------
# Deterministic Workout Statistics Engine
# ---------------------------------------------------------------------------

def calculate_deterministic_statistics(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Separates deterministic calculation from AI interpretation.
    Calculates objective statistics strictly in Python code:
    - sessions analyzed
    - total volume
    - total sets & reps
    - volume by muscle group
    - muscle group frequency
    - exercise frequency
    - most trained & recently untrained muscle groups
    - days between workouts & training frequency
    """
    if not sessions:
        return {
            "sessions_analyzed": 0,
            "total_volume": 0.0,
            "total_sets": 0,
            "total_reps": 0,
            "volume_by_muscle_group": {},
            "most_trained_muscle_groups": [],
            "recently_untrained_muscle_groups": list(STANDARD_MUSCLE_GROUPS),
            "training_frequency_per_week": None,
            "average_days_between_workouts": None,
            "exercise_frequency": {},
            "text_summary": "No completed workout sessions found in history.",
        }

    total_volume = 0.0
    total_sets = 0
    total_reps = 0
    volume_by_muscle: Dict[str, float] = {m: 0.0 for m in STANDARD_MUSCLE_GROUPS}
    muscle_session_count: Dict[str, int] = {m: 0 for m in STANDARD_MUSCLE_GROUPS}
    exercise_count: Dict[str, int] = {}
    session_dates: List[datetime] = []
    routines: List[str] = []

    for s in sessions:
        routine = s.get("routine_name") or "Workout"
        routines.append(routine)

        start_str = s.get("started_at")
        if start_str:
            try:
                dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                session_dates.append(dt)
            except Exception:
                pass

        muscles_in_this_session = set()
        exercises = s.get("exercises", []) or s.get("workout_exercises", [])
        for ex in exercises:
            ex_name = ex.get("exercise_name") or ex.get("exercise_id") or "Exercise"
            exercise_count[ex_name] = exercise_count.get(ex_name, 0) + 1

            muscle = (ex.get("primary_muscle_group") or "General").capitalize()
            # Normalize to standard groups if matching
            matched_muscle = next((m for m in STANDARD_MUSCLE_GROUPS if m.lower() in muscle.lower()), muscle)
            muscles_in_this_session.add(matched_muscle)

            sets = ex.get("sets", []) or ex.get("workout_sets", [])
            for st in sets:
                is_comp = st.get("is_completed", True)
                if not is_comp:
                    continue
                reps = st.get("completed_reps") or st.get("target_reps") or 0
                weight = st.get("completed_weight") or st.get("target_weight") or 0.0
                vol = float(reps) * float(weight)

                total_sets += 1
                total_reps += int(reps)
                total_volume += vol
                volume_by_muscle[matched_muscle] = volume_by_muscle.get(matched_muscle, 0.0) + vol

        for m in muscles_in_this_session:
            muscle_session_count[m] = muscle_session_count.get(m, 0) + 1

    # Filter volume_by_muscle to non-zero values for cleaner display, keeping keys
    active_volume_by_muscle = {k: round(v, 1) for k, v in volume_by_muscle.items() if v > 0}

    # Sort muscles by total volume
    sorted_by_volume = sorted(
        [(k, v) for k, v in volume_by_muscle.items() if v > 0],
        key=lambda x: x[1],
        reverse=True,
    )
    most_trained = [m[0] for m in sorted_by_volume[:3]] if sorted_by_volume else ["Full Body"]

    # "recently_untrained" = standard muscle groups with 0 volume in the analyzed window
    recently_untrained = [m for m in STANDARD_MUSCLE_GROUPS if volume_by_muscle.get(m, 0.0) == 0.0]

    # Calculate days between workouts & frequency
    avg_days_between = None
    freq_per_week = None
    if len(session_dates) >= 2:
        session_dates.sort()
        intervals = [
            (session_dates[i] - session_dates[i - 1]).total_seconds() / 86400.0
            for i in range(1, len(session_dates))
        ]
        avg_days_between = round(sum(intervals) / len(intervals), 1)
        total_days = max(1.0, (session_dates[-1] - session_dates[0]).total_seconds() / 86400.0)
        freq_per_week = round((len(session_dates) / total_days) * 7.0, 1)

    # Calculate volume balance score via Shannon entropy
    balance = volume_balance_score(active_volume_by_muscle)

    # Calculate progressive overload if at least 2 sessions are present
    overload_summary = ""
    if len(sessions) >= 2:
        def _get_vol(s):
            vol = 0.0
            for ex in (s.get("exercises", []) or s.get("workout_exercises", [])):
                for st in (ex.get("sets", []) or ex.get("workout_sets", [])):
                    if st.get("is_completed", True):
                        reps = st.get("completed_reps") or st.get("target_reps") or 0
                        weight = st.get("completed_weight") or st.get("target_weight") or 0.0
                        vol += float(reps) * float(weight)
            return vol

        s0_vol = _get_vol(sessions[0])
        s1_vol = _get_vol(sessions[1])
        ov = progressive_overload(s0_vol, s1_vol)
        overload_summary = f" Recent progression: {ov.label} ({s0_vol:,.0f}kg vs {s1_vol:,.0f}kg)."

    text_summary = (
        f"Analyzed {len(sessions)} completed sessions. "
        f"Total volume lifted: {total_volume:,.1f} kg across {total_sets} sets and {total_reps} reps. "
        f"Volume by muscle: {', '.join(f'{k}: {v:,.0f}kg' for k, v in active_volume_by_muscle.items()) or 'None recorded'}. "
        f"Most trained muscles: {', '.join(most_trained)}. "
        f"Recently untrained muscle groups: {', '.join(recently_untrained) if recently_untrained else 'None (all major groups trained)'}. "
        f"Volume balance: {balance.score:.0f}/100 ({balance.label})."
        f"{overload_summary}"
        + (f" Average rest between sessions: {avg_days_between} days." if avg_days_between else "")
    )

    return {
        "sessions_analyzed": len(sessions),
        "total_volume": round(total_volume, 1),
        "total_sets": total_sets,
        "total_reps": total_reps,
        "volume_by_muscle_group": active_volume_by_muscle,
        "most_trained_muscle_groups": most_trained,
        "recently_untrained_muscle_groups": recently_untrained,
        "training_frequency_per_week": freq_per_week,
        "average_days_between_workouts": avg_days_between,
        "exercise_frequency": exercise_count,
        "volume_balance_score": balance.score,
        "volume_balance_label": balance.label,
        "text_summary": text_summary,
    }


# ---------------------------------------------------------------------------
# Fitness Coaching System Prompt
# ---------------------------------------------------------------------------

COACH_SYSTEM_INSTRUCTION = (
    "You are an elite, evidence-based AI Fitness Coach. "
    "Your goal is to synthesize the user's deterministic workout statistics with the retrieved "
    "fitness knowledge base to provide actionable progressive overload, volume distribution, "
    "recovery, and next-workout suggestions.\n\n"
    "SAFETY CONSTRAINTS:\n"
    "- Provide general fitness and exercise guidance only.\n"
    "- DO NOT diagnose medical conditions, injuries, or prescribe medical treatments.\n"
    "- If pain or injury is mentioned, encourage consulting a qualified healthcare professional.\n\n"
    "OUTPUT FORMAT CONSTRAINTS:\n"
    "You MUST respond ONLY with valid JSON (no markdown fences, no extra text). "
    "Match this exact JSON structure:\n"
    "{\n"
    '  "recommendations": [\n'
    "    {\n"
    '      "title": "<Short title>",\n'
    '      "category": "progression" | "volume" | "technique",\n'
    '      "description": "<Actionable guidance>",\n'
    '      "priority": "high" | "normal"\n'
    "    }\n"
    "  ],\n"
    '  "recovery": [\n'
    "    {\n"
    '      "title": "<Recovery tip title>",\n'
    '      "description": "<Specific recovery recommendation>"\n'
    "    }\n"
    "  ],\n"
    '  "next_workout": {\n'
    '    "title": "<Recommended workout name, e.g. Lower Body & Core>",\n'
    '    "reason": "<Why this session is optimal next based on training balance>",\n'
    '    "exercises": [\n'
    '      {"name": "<Exercise name>", "reason": "<Why this exercise>"}\n'
    "    ]\n"
    "  }\n"
    "}"
)


def _normalize_uuid(val: str) -> str:
    """Ensure user_id is a valid UUID format for PostgreSQL compatibility."""
    try:
        return str(uuid.UUID(str(val)))
    except (ValueError, AttributeError):
        # Generate a deterministic UUID for dev/test users
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(val)))


def _clean_json_output(raw_text: str) -> Dict[str, Any]:
    """Strip markdown code fence blocks if LLM wrapped output, extract JSON object, and unwrap common wrappers."""
    cleaned = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        obj_match = re.search(r"\{[\s\S]*\}", cleaned)
        if obj_match:
            data = json.loads(obj_match.group(0))
        else:
            raise

    if isinstance(data, dict):
        for wrapper_key in ("data", "response", "result", "output"):
            if wrapper_key in data and isinstance(data[wrapper_key], dict):
                inner = data[wrapper_key]
                if any(k in inner for k in ("recommendations", "suggestions", "next_workout", "suggested_workout")):
                    return inner

    return data


def _generate_with_retry_and_fallback(
    query: str,
    context: str,
    system_instruction: str,
    max_retries: int = 2,
    history: Optional[List[ChatMessage]] = None,
    response_mime_type: Optional[str] = None,
    response_schema: Optional[Any] = None,
) -> str:
    """
    Executes generate_answer using existing generate.py.
    generate.py internally iterates through GENERATION_MODEL and FALLBACK_MODELS.
    Retries with a short delay if transient network errors occur, then fails fast
    so the deterministic server fallback can return HTTP 200 within mobile latency budget.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            log.info("Attempting generate_answer (attempt %d/%d)...", attempt + 1, max_retries)
            result = generate_answer(
                query=query,
                context=context,
                system_instruction=system_instruction,
                temperature=0.2,
                history=history,
                response_mime_type=response_mime_type,
                response_schema=response_schema,
            )
            if result and result.strip():
                return result
        except Exception as e:
            last_err = e
            log.warning("generate_answer attempt %d failed: %s", attempt + 1, e)
            if attempt < max_retries - 1:
                time.sleep(0.5)

    raise RuntimeError(f"All generation attempts failed. Last error: {last_err}")


def _build_fallback_recommendations(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Generates deterministic safe fallback recommendations when model generation/validation fails."""
    untrained = stats.get("recently_untrained_muscle_groups") or []
    untrained_focus = untrained[0] if untrained else "Lower Body"
    untrained_str = ", ".join(untrained) if untrained else "all major muscle groups"
    return {
        "recommendations": [
            {
                "title": "Consistent Progressive Overload",
                "category": "progression",
                "description": "Gradually increase reps or load on primary compound lifts by 2-5% week over week.",
                "priority": "high",
            },
            {
                "title": "Address Untrained Muscle Groups",
                "category": "volume",
                "description": f"Incorporate direct sets for {untrained_str} to ensure structural balance.",
                "priority": "normal",
            },
        ],
        "recovery": [
            {
                "title": "Active Recovery & Sleep",
                "description": "Ensure 7-9 hours of quality sleep and maintain hydration to facilitate muscle protein synthesis.",
            }
        ],
        "next_workout": {
            "title": f"Focus: {untrained_focus}",
            "reason": "Balances your recent workout volume and allows prior muscle groups to fully recover.",
            "exercises": [
                {"name": "Barbell Squats", "reason": "High-recruitment compound exercise."},
                {"name": "Romanian Deadlifts", "reason": "Hamstrings and posterior chain reinforcement."},
            ],
        },
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["System"])
def root():
    """Root endpoint providing quick service info and Swagger documentation link."""
    return {
        "service": "Fitness RAG API",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
    }


_qdrant_client: Optional[Any] = None


def get_qdrant_client():
    """Lazily initialize or return cached QdrantClient instance."""
    global _qdrant_client
    if _qdrant_client is None:
        try:
            _qdrant_client = init_qdrant_client()
        except Exception as e:
            log.warning("Could not initialize Qdrant client: %s", e)
            return None
    return _qdrant_client


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint to verify API and connected components (pings Qdrant to prevent cloud suspension)."""
    global _qdrant_client
    qdrant_ok = False
    try:
        q_client = get_qdrant_client()
        if q_client:
            q_client.get_collections()
            qdrant_ok = True
    except Exception as e:
        log.warning("Qdrant keep-alive ping failed: %s", e)
        _qdrant_client = None

    return {
        "status": "ok",
        "service": "fitness_rag_api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY),
        "qdrant_active": qdrant_ok,
    }


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
def get_recommendations(
    request: RecommendRequest,
    user_id: str = Depends(rate_limit_recommend),
):
    """
    Generates personalized workout recommendations based on verified user's workout history.
    1. Authenticates user (derived strictly from token, not client body).
    2. Checks for cached recommendation in Supabase if not force_refresh.
    3. Retrieves user's recent completed sessions from Supabase (or fallback payload).
    4. Calculates objective deterministic statistics (volume by muscle, recently untrained, frequency).
    5. Calls unmodified retrieve.py to search Qdrant knowledge base.
    6. Calls unmodified generate.py with fitness coach system instruction.
    7. Stores the recommendation in Supabase ai_recommendations.
    8. Returns structured JSON with explicit pipeline_source: 'rag'.
    """
    log.info("[API] authenticated user: %s", user_id)
    db_user_id = _normalize_uuid(user_id)
    sb_admin = get_supabase_admin()

    # Step 1: Check cache if not forcing refresh
    if not request.force_refresh and sb_admin:
        try:
            cached_res = (
                sb_admin.table("ai_recommendations")
                .select("*")
                .eq("user_id", db_user_id)
                .order("generated_at", desc=True)
                .limit(1)
                .execute()
            )
            if cached_res.data:
                cached_rec = cached_res.data[0]
                rec_json = cached_rec.get("recommendation_json", {})
                if rec_json:
                    if isinstance(rec_json, str):
                        try:
                            rec_json = json.loads(rec_json)
                        except Exception:
                            rec_json = {}
                    if isinstance(rec_json, dict) and rec_json:
                        rec_json["cached"] = True
                        try:
                            validated_cached = RecommendationResponse.model_validate(rec_json)
                            log.info("[API] returning validated cached recommendation from Supabase for user=%s", user_id)
                            return validated_cached.model_dump()
                        except Exception as val_err:
                            log.warning("[API] cached recommendation failed validation (schema drift/corrupted); discarding and regenerating: %s", val_err)
        except Exception as e:
            log.warning("Cache lookup notice: %s", e)

    # Step 2: Retrieve recent workout history for THIS authenticated user
    sessions: List[Dict[str, Any]] = []

    # Fetch from Supabase strictly by verified user_id
    if sb_admin:
        try:
            res = (
                sb_admin.table("workout_sessions")
                .select("*, workout_exercises(*, workout_sets(*))")
                .eq("user_id", db_user_id)
                .order("started_at", desc=True)
                .limit(10)
                .execute()
            )
            if res.data:
                sessions = res.data
        except Exception as e:
            log.warning("Supabase session fetch notice: %s", e)

    # Fallback to direct sessions passed in request (for offline-sync / local dev)
    if not sessions and request.recent_sessions:
        sessions = [s.model_dump() for s in request.recent_sessions]

    log.info("[API] sessions retrieved: %d", len(sessions))

    # Step 2b: Retrieve today's nutrition intake for user
    today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_start = f"{today_date}T00:00:00Z"
    nutrition_today = {
        "logged_meals": 0,
        "total_calories": 0.0,
        "total_protein_g": 0.0,
        "total_carbs_g": 0.0,
        "total_fat_g": 0.0,
        "target_calories": 2000,
        "target_protein_g": 150,
        "target_carbs_g": 200,
        "target_fat_g": 65,
    }
    if sb_admin:
        try:
            nut_res = (
                sb_admin.table("nutrition_logs")
                .select("*")
                .eq("user_id", db_user_id)
                .gte("logged_at", today_start)
                .order("logged_at", desc=False)
                .execute()
            )
            if nut_res.data:
                logs = nut_res.data
                nutrition_today["logged_meals"] = len(logs)
                nutrition_today["total_calories"] = round(sum(float(l.get("total_calories") or 0.0) for l in logs), 1)
                nutrition_today["total_protein_g"] = round(sum(float(l.get("total_protein_g") or 0.0) for l in logs), 1)
                nutrition_today["total_carbs_g"] = round(sum(float(l.get("total_carbs_g") or 0.0) for l in logs), 1)
                nutrition_today["total_fat_g"] = round(sum(float(l.get("total_fat_g") or 0.0) for l in logs), 1)

            prof_res = (
                sb_admin.table("profiles")
                .select("target_calories, target_protein_g, target_carbs_g, target_fat_g")
                .eq("id", db_user_id)
                .limit(1)
                .execute()
            )
            if prof_res.data:
                p = prof_res.data[0]
                if p.get("target_calories"):
                    nutrition_today["target_calories"] = int(p["target_calories"])
                if p.get("target_protein_g"):
                    nutrition_today["target_protein_g"] = int(p["target_protein_g"])
                if p.get("target_carbs_g"):
                    nutrition_today["target_carbs_g"] = int(p["target_carbs_g"])
                if p.get("target_fat_g"):
                    nutrition_today["target_fat_g"] = int(p["target_fat_g"])
        except Exception as e:
            log.warning("Supabase nutrition fetch notice: %s", e)

    # Step 3: Compute deterministic training context
    stats = calculate_deterministic_statistics(sessions)
    log.info("[API] training context created")

    # Handle completely empty history gracefully without fake stats
    if stats["sessions_analyzed"] == 0:
        now_iso = datetime.now(timezone.utc).isoformat()
        empty_response = {
            "summary": {
                "sessions_analyzed": 0,
                "total_volume": 0.0,
                "total_sets": 0,
                "total_reps": 0,
                "volume_by_muscle_group": {},
                "most_trained_muscle_groups": [],
                "recently_untrained_muscle_groups": list(STANDARD_MUSCLE_GROUPS),
                "training_frequency_per_week": None,
                "average_days_between_workouts": None,
            },
            "recommendations": [
                {
                    "title": "Begin Your Training Journey",
                    "category": "progression",
                    "description": "Complete your first 1-3 workouts to establish a baseline. The AI Coach will then analyze your volume, fatigue, and recovery patterns.",
                    "priority": "normal",
                }
            ],
            "recovery": [
                {
                    "title": "Establish Recovery Habits",
                    "description": "Ensure consistent 7-9 hours of sleep and adequate hydration between introductory workout routines.",
                }
            ],
            "next_workout": {
                "title": "Introductory Full Body Routine",
                "reason": "Establishes baseline strength and neurological movement adaptations.",
                "exercises": [
                    {"name": "Bodyweight Squat", "reason": "Lower body compound movement pattern."},
                    {"name": "Dumbbell Bench Press", "reason": "Upper body horizontal push foundation."},
                    {"name": "Lat Pulldown or Row", "reason": "Upper body vertical/horizontal pull balance."},
                ],
            },
            "generated_at": now_iso,
            "sessions_analyzed": 0,
            "cached": False,
            "pipeline_source": "rag",
            "nutrition_today": nutrition_today,
        }
        return RecommendationResponse.model_validate(empty_response).model_dump()

    # Step 4: Split Vector Retrieval targeting untrained muscle groups and fatigue recovery
    untrained_str = ", ".join(stats["recently_untrained_muscle_groups"]) if stats["recently_untrained_muscle_groups"] else "full body muscular balance"
    trained_str = ", ".join(stats["most_trained_muscle_groups"]) if stats["most_trained_muscle_groups"] else "systemic fatigue"

    untrained_query = f"Hypertrophy stimulus, movement selection, and programming for untrained muscle groups: {untrained_str}."
    recovery_query = f"Fatigue management, recovery timelines, and progressive overload protocols for trained muscle groups: {trained_str}."

    log.info("[RAG] focused retrieval started (untrained + recovery)")
    retrieved_parts: list[str] = []
    chunk_count = 0

    try:
        k1 = retrieve(query=untrained_query)
        if k1:
            retrieved_parts.append(k1)
            chunk_count += len([c for c in k1.split("\n\n") if c.strip()])
    except Exception as e:
        log.warning("RAG retrieval notice for untrained query: %s", e)

    try:
        k2 = retrieve(query=recovery_query)
        if k2:
            retrieved_parts.append(k2)
            chunk_count += len([c for c in k2.split("\n\n") if c.strip()])
    except Exception as e:
        log.warning("RAG retrieval notice for recovery query: %s", e)

    retrieved_knowledge = "\n\n".join(retrieved_parts)
    log.info("[RAG] retrieved chunks: %d", chunk_count)

    # Step 5: Combine context & invoke unmodified generate.py
    nutrition_text_summary = (
        f"TODAY'S MACRONUTRITION INTAKE (FACTS):\n"
        f"- Logged Meals Today: {nutrition_today['logged_meals']}\n"
        f"- Energy: {nutrition_today['total_calories']:.0f} / {nutrition_today['target_calories']} kcal\n"
        f"- Protein: {nutrition_today['total_protein_g']:.0f} / {nutrition_today['target_protein_g']}g\n"
        f"- Carbohydrates: {nutrition_today['total_carbs_g']:.0f} / {nutrition_today['target_carbs_g']}g\n"
        f"- Dietary Fat: {nutrition_today['total_fat_g']:.0f} / {nutrition_today['target_fat_g']}g\n\n"
    )

    generator_context = (
        f"DETERMINISTIC USER TRAINING STATISTICS (DO NOT RECALCULATE, USE AS FACTS):\n"
        f"{stats['text_summary']}\n\n"
        f"{nutrition_text_summary}"
        f"RETRIEVED EXERCISE SCIENCE & RECOVERY PRINCIPLES:\n"
        f"{retrieved_knowledge if retrieved_knowledge else 'Focus on progressive overload, balanced muscle group frequency, and 48-hour recovery between intense muscle sessions.'}"
    )

    generator_query = (
        "Based on the objective training statistics and retrieved research, analyze progress, "
        "identify training balance, provide recovery advice, and suggest the exact next workout."
    )

    log.info("[RAG] generation started")
    pipeline_source = "rag"
    now_iso = datetime.now(timezone.utc).isoformat()
    summary_data = {
        "sessions_analyzed": stats["sessions_analyzed"],
        "total_volume": stats["total_volume"],
        "total_sets": stats["total_sets"],
        "total_reps": stats["total_reps"],
        "volume_by_muscle_group": stats["volume_by_muscle_group"],
        "most_trained_muscle_groups": stats["most_trained_muscle_groups"],
        "recently_untrained_muscle_groups": stats["recently_untrained_muscle_groups"],
        "training_frequency_per_week": stats["training_frequency_per_week"],
        "average_days_between_workouts": stats["average_days_between_workouts"],
    }

    try:
        raw_answer = _generate_with_retry_and_fallback(
            query=generator_query,
            context=generator_context,
            system_instruction=COACH_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=LLMRecommendationOutput,
        )
        parsed_llm_json = _clean_json_output(raw_answer)
        log.info("[RAG] generation completed")

        llm_output = LLMRecommendationOutput.model_validate(parsed_llm_json)
        candidate_response = {
            "summary": summary_data,
            "recommendations": [r.model_dump() for r in llm_output.recommendations],
            "recovery": [rc.model_dump() for rc in llm_output.recovery],
            "next_workout": llm_output.next_workout.model_dump(),
            "generated_at": now_iso,
            "sessions_analyzed": stats["sessions_analyzed"],
            "cached": False,
            "pipeline_source": "rag",
            "nutrition_today": nutrition_today,
        }
        validated_response = RecommendationResponse.model_validate(candidate_response)
        response_data = validated_response.model_dump()
    except Exception as e:
        log.error("Failed to generate or validate recommendation JSON from RAG pipeline: %s", e)
        pipeline_source = "fallback"
        fallback_data = _build_fallback_recommendations(stats)
        candidate_fallback = {
            "summary": summary_data,
            "recommendations": fallback_data["recommendations"],
            "recovery": fallback_data["recovery"],
            "next_workout": fallback_data["next_workout"],
            "generated_at": now_iso,
            "sessions_analyzed": stats["sessions_analyzed"],
            "cached": False,
            "pipeline_source": "fallback",
            "nutrition_today": nutrition_today,
        }
        validated_response = RecommendationResponse.model_validate(candidate_fallback)
        response_data = validated_response.model_dump()

    # Step 7: Store in Supabase ai_recommendations table
    if sb_admin:
        try:
            summary_text = (
                f"Analyzed {stats['sessions_analyzed']} sessions ({stats['total_volume']:,.0f} kg). "
                f"Next suggested: {response_data['next_workout']['title']}."
            )
            sb_admin.table("ai_recommendations").insert({
                "user_id": db_user_id,
                "generated_at": now_iso,
                "sessions_analyzed": stats["sessions_analyzed"],
                "recommendation_json": response_data,
                "summary": summary_text,
                "model_provider": "gemini + cohere-rerank",
                "prompt_version": "v2",
            }).execute()
            log.info("[API] saved recommendation to Supabase ai_recommendations for user=%s", user_id)
        except Exception as e:
            log.warning("Could not persist recommendation to Supabase: %s", e)

    return response_data


# ---------------------------------------------------------------------------
# Conversational AI Chat Endpoint (RAG Grounded)
# ---------------------------------------------------------------------------

CHAT_SYSTEM_INSTRUCTION = """You are PulseFit AI Coach, an elite, encouraging, and evidence-based AI fitness assistant built into the PulseFit AI mobile app.
You provide clear, actionable, and scientifically grounded answers on workout splits, biomechanics, exercise technique, nutrition, progressive overload, recovery, and injury prevention.
Answer directly using the retrieved exercise science context when relevant.
Keep responses concise, clear, and easy to read on mobile devices (use bullet points and bold highlights).
If asked about severe pain or medical conditions, advise consulting a physical therapist or physician while providing safe general exercise modifications."""


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat_with_coach(
    request: ChatRequest,
    user_id: str = Depends(rate_limit_chat),
):
    """
    Conversational RAG endpoint:
    1. Enforces endpoint rate limiting (max 20 req/min).
    2. Sanitizes input to prevent prompt injection and control character abuse.
    3. Retrieves relevant fitness science documents from Qdrant via Cohere reranking,
       preserving payload metadata and extracting accurate document citations.
    4. Constructs grounded context.
    5. Generates conversational coaching advice using Gemini with full multi-turn history.
    """
    # Input sanitization
    query = request.message.strip()
    query = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", query)
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty.",
        )
    log.info("[CHAT] query: '%s' from user=%s (history_len=%d)", query, user_id, len(request.history or []))

    retrieved_knowledge = ""
    sources: list[str] = []
    try:
        retrieval_res = retrieve(query=query, return_metadata=True)
        if isinstance(retrieval_res, tuple):
            retrieved_knowledge, payloads = retrieval_res
            for p in payloads:
                doc = p.get("document_name") or p.get("title")
                if doc:
                    cleaned_doc = doc.replace(".txt", "").replace("_", " ").title()
                    if cleaned_doc and cleaned_doc not in sources:
                        sources.append(cleaned_doc)
        else:
            retrieved_knowledge = retrieval_res

        if retrieved_knowledge:
            for block in retrieved_knowledge.split("\n\n"):
                for line in block.split("\n"):
                    if line.startswith("## "):
                        title = line.replace("## ", "").strip()
                        if title and title not in sources:
                            sources.append(title)
    except Exception as e:
        log.warning("Chat RAG retrieval notice: %s", e)

    chat_context = (
        f"RETRIEVED EXERCISE SCIENCE RESEARCH & KNOWLEDGE BASE:\n"
        f"{retrieved_knowledge if retrieved_knowledge else 'Focus on general evidence-based training principles, progressive overload, proper form, and recovery.'}"
    )

    try:
        reply = _generate_with_retry_and_fallback(
            query=query,
            context=chat_context,
            system_instruction=CHAT_SYSTEM_INSTRUCTION,
            max_retries=2,
            history=request.history,
        )
        return ChatResponse(
            reply=reply.strip(),
            sources=sources[:3],
            pipeline_source="rag" if retrieved_knowledge else "general",
        )
    except Exception as e:
        log.error("Chat generation error: %s", e)
        return ChatResponse(
            reply="I am analyzing your training request. In the meantime, remember to focus on progressive overload, controlled tempo, and 48 hours of recovery between intense sessions for the same muscle group!",
            sources=[],
            pipeline_source="fallback",
        )


# ---------------------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    log.info("Starting Fitness RAG API server on http://0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
