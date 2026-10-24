"""
PulseFit Fitness Mathematics Engine (Python Backend)
=====================================================
Deterministic, scientifically-validated fitness formulas for the RAG API.
Mirrors the Dart frontend engine for server-side calculations and RAG context.

References:
  - Mifflin-St Jeor (1990)   — BMR
  - Epley (1985) / Brzycki    — 1RM
  - ACSM MET Compendium       — Calorie Burn
  - Hodgdon & Beckett (1984)  — Navy Body Fat %
  - Shannon Entropy            — Volume Balance Scoring
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _r1(v: float) -> float:
    """Round to 1 decimal place safely."""
    return 0.0 if not math.isfinite(v) else round(v, 1)


def _log10(x: float) -> float:
    return math.log10(x) if x > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. BMR / TDEE / Auto Macro Splits
# ─────────────────────────────────────────────────────────────────────────────

class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"


class ActivityLevel(str, Enum):
    SEDENTARY = "sedentary"
    LIGHT = "light"
    MODERATE = "moderate"
    ACTIVE = "active"
    ATHLETE = "athlete"


class FitnessGoal(str, Enum):
    FAT_LOSS = "fat_loss"
    MILD_CUT = "mild_cut"
    MAINTAIN = "maintain"
    MILD_BULK = "mild_bulk"
    BULKING = "bulking"


ACTIVITY_MULTIPLIERS = {
    ActivityLevel.SEDENTARY: 1.2,
    ActivityLevel.LIGHT: 1.375,
    ActivityLevel.MODERATE: 1.55,
    ActivityLevel.ACTIVE: 1.725,
    ActivityLevel.ATHLETE: 1.9,
}

GOAL_ADJUSTMENTS = {
    FitnessGoal.FAT_LOSS: -500,
    FitnessGoal.MILD_CUT: -250,
    FitnessGoal.MAINTAIN: 0,
    FitnessGoal.MILD_BULK: 250,
    FitnessGoal.BULKING: 500,
}


@dataclass
class MacroSplit:
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    tdee: float
    bmr: float
    goal_adjustment: int


def calculate_bmr(
    weight_kg: float,
    height_cm: float,
    age_years: int,
    sex: Sex,
) -> float:
    """
    Mifflin-St Jeor BMR (kcal/day) — gold standard, ±10% accuracy.

    Male:   BMR = 10w + 6.25h − 5a + 5
    Female: BMR = 10w + 6.25h − 5a − 161
    """
    base = (10.0 * weight_kg) + (6.25 * height_cm) - (5.0 * age_years)
    return base + 5.0 if sex == Sex.MALE else base - 161.0


def calculate_tdee(
    weight_kg: float,
    height_cm: float,
    age_years: int,
    sex: Sex,
    activity: ActivityLevel,
) -> float:
    """TDEE = BMR × Activity Factor."""
    return calculate_bmr(weight_kg, height_cm, age_years, sex) * ACTIVITY_MULTIPLIERS[activity]


def auto_macros(
    weight_kg: float,
    height_cm: float,
    age_years: int,
    sex: Sex,
    activity: ActivityLevel,
    goal: FitnessGoal = FitnessGoal.MAINTAIN,
    protein_pct: float = 0.30,
    carbs_pct: float = 0.40,
    fat_pct: float = 0.30,
) -> MacroSplit:
    """Full auto-macro calculation: TDEE → goal adjustment → macro split."""
    bmr_val = calculate_bmr(weight_kg, height_cm, age_years, sex)
    tdee_val = bmr_val * ACTIVITY_MULTIPLIERS[activity]
    adj = GOAL_ADJUSTMENTS[goal]
    target_cal = max(1200.0, min(10000.0, tdee_val + adj))

    return MacroSplit(
        calories=_r1(target_cal),
        protein_g=_r1((target_cal * protein_pct) / 4.0),
        carbs_g=_r1((target_cal * carbs_pct) / 4.0),
        fat_g=_r1((target_cal * fat_pct) / 9.0),
        tdee=_r1(tdee_val),
        bmr=_r1(bmr_val),
        goal_adjustment=adj,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. One-Rep Max (1RM)
# ─────────────────────────────────────────────────────────────────────────────

def epley_1rm(weight: float, reps: int) -> float:
    """Epley: 1RM = w × (1 + r/30)"""
    if reps <= 0 or weight <= 0:
        return 0.0
    if reps == 1:
        return weight
    return _r1(weight * (1.0 + reps / 30.0))


def brzycki_1rm(weight: float, reps: int) -> float:
    """Brzycki: 1RM = w / (1.0278 − 0.0278 × r)"""
    if reps <= 0 or weight <= 0:
        return 0.0
    if reps == 1:
        return weight
    denom = 1.0278 - (0.0278 * reps)
    if denom <= 0:
        return epley_1rm(weight, reps)
    return _r1(weight / denom)


def estimate_1rm(weight: float, reps: int) -> float:
    """Best-fit: Brzycki for ≤10 reps, Epley for >10."""
    if reps <= 10:
        return brzycki_1rm(weight, reps)
    return epley_1rm(weight, reps)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Progressive Overload
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OverloadResult:
    current_volume: float
    previous_volume: float
    overload_pct: float
    label: str
    is_first_session: bool = False

    @property
    def is_positive(self) -> bool:
        return not self.is_first_session and self.overload_pct > 0

    @property
    def is_pr(self) -> bool:
        return not self.is_first_session and self.overload_pct >= 5.0


def progressive_overload(current_volume: float, previous_volume: float) -> OverloadResult:
    """Δ% = ((V_current − V_previous) / V_previous) × 100"""
    if previous_volume <= 0:
        return OverloadResult(current_volume, 0, 0.0, "First Session", is_first_session=True)

    delta = ((current_volume - previous_volume) / previous_volume) * 100.0
    rounded = _r1(delta)

    if rounded >= 5.0:
        label = f"+{abs(rounded)}% 🔥 PR"
    elif rounded > 0:
        label = f"+{abs(rounded)}%"
    elif rounded == 0:
        label = "0% (Maintained)"
    else:
        label = f"−{abs(rounded)}%"

    return OverloadResult(current_volume, previous_volume, rounded, label)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Adaptive Rest Timer
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_rest_from_reps(reps: int) -> int:
    """
    NSCA guidelines:
      reps ≥ 15 → 60s  (endurance)
      8–14      → 90s  (hypertrophy)
      4–7       → 150s (strength)
      1–3       → 240s (power)
    """
    if reps >= 15:
        return 60
    if reps >= 8:
        return 90
    if reps >= 4:
        return 150
    return 240


def adaptive_rest_from_intensity(weight: float, reps: int, estimated_1rm: Optional[float] = None) -> int:
    """Rest(s) = 30 + 150 × (w / 1RM)²"""
    if estimated_1rm is None or estimated_1rm <= 0:
        return adaptive_rest_from_reps(reps)
    pct = min(1.0, max(0.0, weight / estimated_1rm))
    rest = 30.0 + 150.0 * pct * pct
    return max(45, min(300, round(rest)))


def training_zone_label(reps: int) -> str:
    if reps >= 15:
        return "Endurance"
    if reps >= 8:
        return "Hypertrophy"
    if reps >= 4:
        return "Strength"
    return "Power"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Streak & Consistency
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConsistencyResult:
    current_streak: int
    workouts_this_week: int
    consistency_pct: float  # 0–100
    last_28_sessions: int
    target_per_week: int

    @property
    def label(self) -> str:
        if self.consistency_pct >= 90:
            return "Elite"
        if self.consistency_pct >= 75:
            return "Strong"
        if self.consistency_pct >= 50:
            return "Decent"
        if self.consistency_pct >= 25:
            return "Building"
        return "Getting Started"


def calculate_consistency(
    session_dates: List[datetime],
    target_per_week: int = 4,
    now: Optional[datetime] = None,
) -> ConsistencyResult:
    """Calculate streak, weekly count, and consistency from session dates."""
    today = now or datetime.now()
    today_date = today.replace(hour=0, minute=0, second=0, microsecond=0)

    if not session_dates:
        return ConsistencyResult(0, 0, 0.0, 0, target_per_week)

    unique_days = sorted(
        {dt.replace(hour=0, minute=0, second=0, microsecond=0) for dt in session_dates},
        reverse=True,
    )

    # ── Streak ──
    streak = 0
    check_date = today_date
    if unique_days and unique_days[0] != today_date:
        yesterday = today_date - timedelta(days=1)
        if unique_days[0] == yesterday:
            check_date = yesterday
    for day in unique_days:
        if day == check_date:
            streak += 1
            check_date -= timedelta(days=1)
        elif day < check_date:
            break

    # ── This week (Monday-based) ──
    days_since_monday = today.weekday()
    monday = today_date - timedelta(days=days_since_monday)
    this_week = sum(1 for d in unique_days if monday <= d <= today_date)

    # ── Last 28 days ──
    cutoff = today_date - timedelta(days=28)
    last_28 = sum(1 for d in unique_days if cutoff <= d <= today_date)

    target_28 = target_per_week * 4
    consistency = min(100.0, (last_28 / target_28 * 100.0)) if target_28 > 0 else 0.0

    return ConsistencyResult(streak, this_week, _r1(consistency), last_28, target_per_week)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Per-Muscle Training Frequency
# ─────────────────────────────────────────────────────────────────────────────

STANDARD_GROUPS = ["Chest", "Back", "Shoulders", "Arms", "Legs", "Core"]


class FrequencyStatus(str, Enum):
    NEGLECTED = "neglected"        # < 1.0×/week
    UNDERTRAINED = "undertrained"  # 1.0–1.49
    OPTIMAL = "optimal"            # 1.5–3.0
    OVERREACHING = "overreaching"  # > 3.0


@dataclass
class MuscleFrequencyResult:
    muscle: str
    sessions_hit: int
    frequency_per_week: float
    status: FrequencyStatus


def analyze_muscle_frequency(
    sessions_with_muscles: List[Tuple[datetime, List[str]]],
    window_days: int = 28,
) -> List[MuscleFrequencyResult]:
    """Analyze per-muscle training frequency over a date window."""
    weeks = max(1.0, window_days / 7.0)
    counts: Dict[str, int] = {g: 0 for g in STANDARD_GROUPS}

    for _date, muscles in sessions_with_muscles:
        for m in set(muscles):
            matched = next((g for g in STANDARD_GROUPS if g.lower() == m.lower()), m)
            counts[matched] = counts.get(matched, 0) + 1

    results = []
    for muscle, count in counts.items():
        freq = _r1(count / weeks)
        if freq < 1.0:
            status = FrequencyStatus.NEGLECTED
        elif freq < 1.5:
            status = FrequencyStatus.UNDERTRAINED
        elif freq <= 3.0:
            status = FrequencyStatus.OPTIMAL
        else:
            status = FrequencyStatus.OVERREACHING
        results.append(MuscleFrequencyResult(muscle, count, freq, status))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. Workout Intensity Score (VLI)
# ─────────────────────────────────────────────────────────────────────────────

class IntensityLevel(str, Enum):
    LIGHT = "light"
    MODERATE = "moderate"
    HARD = "hard"
    BRUTAL = "brutal"


@dataclass
class IntensityResult:
    vli: float
    level: IntensityLevel
    label: str


def workout_intensity(total_volume_kg: float, duration_seconds: int) -> IntensityResult:
    """
    Volume Load Index = Total Volume(kg) / Duration(min).

    | VLI    | Label    |
    |--------|----------|
    | < 50   | Light    |
    | 50–150 | Moderate |
    | 150–300| Hard     |
    | > 300  | Brutal   |
    """
    if duration_seconds <= 0:
        return IntensityResult(0, IntensityLevel.LIGHT, "Light")

    vli = _r1(total_volume_kg / (duration_seconds / 60.0))

    if vli < 50:
        return IntensityResult(vli, IntensityLevel.LIGHT, "Light")
    if vli < 150:
        return IntensityResult(vli, IntensityLevel.MODERATE, "Moderate")
    if vli < 300:
        return IntensityResult(vli, IntensityLevel.HARD, "Hard")
    return IntensityResult(vli, IntensityLevel.BRUTAL, "Brutal 🔥")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Body Composition (BMI + Navy BF%)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BodyCompositionResult:
    bmi: float
    bmi_category: str
    body_fat_pct: float
    fat_mass_kg: float
    lean_mass_kg: float
    body_fat_category: str


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    if height_cm <= 0 or weight_kg <= 0:
        return 0.0
    hm = height_cm / 100.0
    return _r1(weight_kg / (hm * hm))


def navy_body_fat(
    sex: Sex,
    waist_cm: float,
    neck_cm: float,
    height_cm: float,
    hip_cm: Optional[float] = None,
) -> float:
    """
    US Navy Body Fat % — Hodgdon & Beckett (1984).

    Male:   BF% = 86.010 × log₁₀(waist − neck) − 70.041 × log₁₀(height) + 36.76
    Female: BF% = 163.205 × log₁₀(waist + hip − neck) − 97.684 × log₁₀(height) − 78.387
    """
    if height_cm <= 0 or neck_cm <= 0 or waist_cm <= 0:
        return 0.0

    if sex == Sex.MALE:
        diff = waist_cm - neck_cm
        if diff <= 0:
            return 0.0
        bf = 86.010 * _log10(diff) - 70.041 * _log10(height_cm) + 36.76
    else:
        hip = hip_cm or 0
        if hip <= 0:
            return 0.0
        circ = waist_cm + hip - neck_cm
        if circ <= 0:
            return 0.0
        bf = 163.205 * _log10(circ) - 97.684 * _log10(height_cm) - 78.387

    return _r1(max(2.0, min(60.0, bf)))


def body_composition(
    weight_kg: float,
    height_cm: float,
    sex: Sex,
    waist_cm: float,
    neck_cm: float,
    hip_cm: Optional[float] = None,
) -> BodyCompositionResult:
    """Full body composition analysis."""
    bmi_val = calculate_bmi(weight_kg, height_cm)

    # BMI category
    if bmi_val < 18.5:
        bmi_cat = "Underweight"
    elif bmi_val < 25.0:
        bmi_cat = "Normal"
    elif bmi_val < 30.0:
        bmi_cat = "Overweight"
    else:
        bmi_cat = "Obese"

    is_invalid = (
        weight_kg <= 0
        or waist_cm <= 0
        or neck_cm <= 0
        or height_cm <= 0
        or (sex == Sex.MALE and waist_cm <= neck_cm)
        or (sex == Sex.FEMALE and ((hip_cm or 0) <= 0 or (waist_cm + (hip_cm or 0) <= neck_cm)))
    )
    if is_invalid:
        return BodyCompositionResult(
            bmi=bmi_val,
            bmi_category=bmi_cat,
            body_fat_pct=0.0,
            fat_mass_kg=0.0,
            lean_mass_kg=weight_kg,
            body_fat_category="Invalid Measurements",
        )

    bf_pct = navy_body_fat(sex, waist_cm, neck_cm, height_cm, hip_cm)
    fat_mass = _r1(weight_kg * bf_pct / 100.0)
    lean_mass = _r1(weight_kg - fat_mass)

    # BF% category
    if sex == Sex.MALE:
        if bf_pct < 6:
            bf_cat = "Essential"
        elif bf_pct < 14:
            bf_cat = "Athletic"
        elif bf_pct < 18:
            bf_cat = "Fitness"
        elif bf_pct < 25:
            bf_cat = "Average"
        else:
            bf_cat = "Above Average"
    else:
        if bf_pct < 14:
            bf_cat = "Essential"
        elif bf_pct < 21:
            bf_cat = "Athletic"
        elif bf_pct < 25:
            bf_cat = "Fitness"
        elif bf_pct < 32:
            bf_cat = "Average"
        else:
            bf_cat = "Above Average"

    return BodyCompositionResult(bmi_val, bmi_cat, bf_pct, fat_mass, lean_mass, bf_cat)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Calorie Burn (MET-based)
# ─────────────────────────────────────────────────────────────────────────────

MET_WEIGHTLIFTING = 6.0
MET_HIGH_INTENSITY = 8.0
MET_BODYWEIGHT = 5.0
MET_REST = 2.0


@dataclass
class CalorieBurnResult:
    total_kcal: float
    active_kcal: float
    rest_kcal: float
    average_met: float


def estimate_workout_burn(
    weight_kg: float,
    duration_seconds: int,
    active_seconds: Optional[int] = None,
    is_high_intensity: bool = False,
    is_bodyweight: bool = False,
) -> CalorieBurnResult:
    """kcal = MET × bodyWeight(kg) × time(hours)"""
    if duration_seconds <= 0 or weight_kg <= 0:
        return CalorieBurnResult(0, 0, 0, 0)

    total_min = duration_seconds / 60.0
    active_min = min(total_min, max(0.0, active_seconds / 60.0)) if active_seconds is not None else total_min * 0.45
    rest_min = max(0.0, total_min - active_min)

    if is_bodyweight:
        active_met = MET_BODYWEIGHT
    elif is_high_intensity:
        active_met = MET_HIGH_INTENSITY
    else:
        active_met = MET_WEIGHTLIFTING

    active_kcal = _r1(active_met * weight_kg * (active_min / 60.0))
    rest_kcal = _r1(MET_REST * weight_kg * (rest_min / 60.0))
    total_kcal = _r1(active_kcal + rest_kcal)
    avg_met = _r1((active_met * active_min + MET_REST * rest_min) / total_min) if total_min > 0 else 0.0

    return CalorieBurnResult(total_kcal, active_kcal, rest_kcal, avg_met)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Entropy-Based Volume Balance Scoring
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BalanceScoreResult:
    score: float  # 0–100
    label: str
    entropy: float
    max_entropy: float
    dominant_muscle: str
    dominant_pct: float


TOTAL_STANDARD_GROUPS = 6


def volume_balance_score(volume_by_muscle: Dict[str, float]) -> BalanceScoreResult:
    """
    Shannon entropy-based balance score.

    H = −Σ pᵢ × ln(pᵢ)
    Score = (H / ln(TOTAL_STANDARD_GROUPS)) × 100

    100 = perfectly balanced across all 6 standard muscle groups.  0 = all in one group.
    """
    non_zero = {k: v for k, v in volume_by_muscle.items() if v > 0}

    if not non_zero:
        return BalanceScoreResult(0, "No Data", 0, 0, "-", 0)

    if len(non_zero) == 1:
        return BalanceScoreResult(0, "Single Focus", 0, 0, list(non_zero.keys())[0], 100)

    total = sum(non_zero.values())
    max_entropy = math.log(TOTAL_STANDARD_GROUPS)

    entropy = 0.0
    dominant = ""
    dominant_val = 0.0

    for muscle, vol in non_zero.items():
        p = vol / total
        if p > 0:
            entropy -= p * math.log(p)
        if vol > dominant_val:
            dominant_val = vol
            dominant = muscle

    score = _r1(max(0.0, min(100.0, (entropy / max_entropy) * 100.0))) if max_entropy > 0 else 0.0
    dominant_pct = _r1((dominant_val / total) * 100.0)

    if score >= 85:
        label = "Excellent Balance"
    elif score >= 65:
        label = "Good Balance"
    elif score >= 40:
        label = "Moderate Imbalance"
    else:
        label = "Heavily Skewed"

    return BalanceScoreResult(score, label, _r1(entropy), _r1(max_entropy), dominant, dominant_pct)
