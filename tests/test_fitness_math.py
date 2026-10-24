"""Tests for PulseFit Fitness Mathematics Engine (Python)."""

import math
from datetime import datetime, timedelta

from fitness_math import (
    ActivityLevel,
    CalorieBurnResult,
    FitnessGoal,
    Sex,
    adaptive_rest_from_intensity,
    adaptive_rest_from_reps,
    auto_macros,
    body_composition,
    brzycki_1rm,
    calculate_bmi,
    calculate_bmr,
    calculate_consistency,
    calculate_tdee,
    epley_1rm,
    estimate_1rm,
    estimate_workout_burn,
    navy_body_fat,
    progressive_overload,
    analyze_muscle_frequency,
    volume_balance_score,
    workout_intensity,
)


# ─── 1. BMR / TDEE ──────────────────────────────────────────────────────

def test_bmr_male():
    bmr = calculate_bmr(80, 180, 25, Sex.MALE)
    assert bmr == 1805.0, f"Expected 1805.0, got {bmr}"


def test_bmr_female():
    bmr = calculate_bmr(60, 165, 30, Sex.FEMALE)
    assert abs(bmr - 1320.3) < 0.2, f"Expected ~1320.3, got {bmr}"


def test_tdee_moderate():
    tdee = calculate_tdee(80, 180, 25, Sex.MALE, ActivityLevel.MODERATE)
    assert abs(tdee - 2797.8) < 0.5, f"Expected ~2797.8, got {tdee}"


def test_auto_macros_fat_loss():
    result = auto_macros(80, 180, 25, Sex.MALE, ActivityLevel.MODERATE, FitnessGoal.FAT_LOSS)
    assert abs(result.calories - 2297.8) < 2
    assert result.goal_adjustment == -500
    # Verify Atwater invariance
    atwater = result.protein_g * 4 + result.carbs_g * 4 + result.fat_g * 9
    assert abs(atwater - result.calories) < 3, f"Atwater mismatch: {atwater} vs {result.calories}"


# ─── 2. 1RM ─────────────────────────────────────────────────────────────

def test_epley_1rm():
    orm = epley_1rm(100, 10)
    assert abs(orm - 133.3) < 0.2


def test_brzycki_1rm():
    orm = brzycki_1rm(100, 5)
    assert abs(orm - 112.5) < 0.5


def test_estimate_1rm_switches():
    # ≤10 uses Brzycki, >10 uses Epley
    assert estimate_1rm(100, 8) == brzycki_1rm(100, 8)
    assert estimate_1rm(100, 15) == epley_1rm(100, 15)


def test_1rm_edge_cases():
    assert estimate_1rm(140, 1) == 140.0
    assert estimate_1rm(0, 5) == 0.0
    assert estimate_1rm(100, 0) == 0.0


# ─── 3. Progressive Overload ────────────────────────────────────────────

def test_positive_overload():
    r = progressive_overload(5000, 4750)
    assert abs(r.overload_pct - 5.3) < 0.2
    assert "PR" in r.label


def test_negative_overload():
    r = progressive_overload(4500, 5000)
    assert r.overload_pct < 0


def test_first_session():
    r = progressive_overload(3000, 0)
    assert r.is_first_session is True
    assert r.overload_pct == 0.0
    assert r.is_pr is False
    assert r.is_positive is False
    assert r.label == "First Session"


# ─── 4. Adaptive Rest ───────────────────────────────────────────────────

def test_rest_from_reps():
    assert adaptive_rest_from_reps(20) == 60
    assert adaptive_rest_from_reps(10) == 90
    assert adaptive_rest_from_reps(5) == 150
    assert adaptive_rest_from_reps(2) == 240


def test_rest_from_intensity():
    rest = adaptive_rest_from_intensity(95, 1, 100.0)
    assert 160 <= rest <= 170  # 30 + 150×0.9025 ≈ 165


# ─── 5. Streak & Consistency ────────────────────────────────────────────

def test_streak_3_days():
    now = datetime(2026, 10, 24)
    dates = [
        datetime(2026, 10, 24),
        datetime(2026, 10, 23),
        datetime(2026, 10, 22),
        datetime(2026, 10, 19),
    ]
    r = calculate_consistency(dates, now=now)
    assert r.current_streak == 3


def test_streak_from_yesterday():
    now = datetime(2026, 10, 24)
    dates = [datetime(2026, 10, 23), datetime(2026, 10, 22)]
    r = calculate_consistency(dates, now=now)
    assert r.current_streak == 2


def test_consistency_pct():
    now = datetime(2026, 10, 24)
    dates = [now - timedelta(days=i * 2) for i in range(12)]
    r = calculate_consistency(dates, target_per_week=4, now=now)
    assert r.consistency_pct == 75.0


def test_empty_sessions():
    r = calculate_consistency([])
    assert r.current_streak == 0
    assert r.workouts_this_week == 0


# ─── 6. Muscle Frequency ────────────────────────────────────────────────

def test_muscle_frequency():
    sessions = [
        (datetime(2026, 10, 20), ["Chest", "Arms"]),
        (datetime(2026, 10, 23), ["Chest", "Shoulders"]),
    ]
    results = analyze_muscle_frequency(sessions, window_days=7)
    chest = next(r for r in results if r.muscle == "Chest")
    assert chest.frequency_per_week == 2.0
    assert chest.status.value == "optimal"


# ─── 7. Workout Intensity ───────────────────────────────────────────────

def test_intensity_moderate():
    r = workout_intensity(5000, 45 * 60)
    assert abs(r.vli - 111.1) < 0.2
    assert r.level.value == "moderate"


def test_intensity_brutal():
    r = workout_intensity(20000, 50 * 60)
    assert r.vli >= 300
    assert r.level.value == "brutal"


# ─── 8. Body Composition ────────────────────────────────────────────────

def test_bmi():
    bmi = calculate_bmi(80, 180)
    assert abs(bmi - 24.7) < 0.1


def test_navy_bf_male():
    bf = navy_body_fat(Sex.MALE, 85, 38, 180)
    assert 5 < bf < 30  # Reasonable range


def test_body_comp_analysis():
    r = body_composition(80, 180, Sex.MALE, 82, 38)
    assert r.bmi > 20
    assert r.body_fat_pct > 5
    assert abs(r.fat_mass_kg + r.lean_mass_kg - 80) < 0.5


def test_body_comp_invalid_measurements():
    r = body_composition(80, 180, Sex.MALE, 35, 38)
    assert r.body_fat_pct == 0.0
    assert r.body_fat_category == "Invalid Measurements"
    assert r.fat_mass_kg == 0.0
    assert r.lean_mass_kg == 80.0


def test_body_comp_female_invalid_measurements():
    # waist(40) + hip(40) <= neck(80)
    r = body_composition(60, 165, Sex.FEMALE, 40, 80, hip_cm=40)
    assert r.body_fat_pct == 0.0
    assert r.body_fat_category == "Invalid Measurements"
    assert r.fat_mass_kg == 0.0
    assert r.lean_mass_kg == 60.0


def test_body_comp_non_positive_weight():
    r = body_composition(0, 180, Sex.MALE, 80, 38)
    assert r.body_fat_pct == 0.0
    assert r.body_fat_category == "Invalid Measurements"
    assert r.bmi == 0.0


# ─── 9. Calorie Burn ────────────────────────────────────────────────────

def test_calorie_burn():
    r = estimate_workout_burn(80, 60 * 60)
    assert 200 < r.total_kcal < 500
    assert r.active_kcal > r.rest_kcal


def test_calorie_burn_rest_clamping():
    r = estimate_workout_burn(80, 30 * 60, active_seconds=40 * 60)
    assert r.rest_kcal == 0.0
    assert r.total_kcal > 0.0


def test_calorie_burn_negative_active_clamping():
    r = estimate_workout_burn(80, 30 * 60, active_seconds=-120)
    assert r.active_kcal == 0.0
    assert r.rest_kcal > 0.0
    assert r.total_kcal == r.rest_kcal


def test_hi_burns_more():
    hi = estimate_workout_burn(80, 45 * 60, is_high_intensity=True)
    normal = estimate_workout_burn(80, 45 * 60)
    assert hi.total_kcal > normal.total_kcal


# ─── 10. Volume Balance ─────────────────────────────────────────────────

def test_perfect_balance():
    r = volume_balance_score({
        "Chest": 1000, "Back": 1000, "Legs": 1000,
        "Shoulders": 1000, "Arms": 1000, "Core": 1000,
    })
    assert abs(r.score - 100.0) < 0.2


def test_volume_balance_more_than_six_groups():
    r = volume_balance_score({
        "Chest": 1000, "Back": 1000, "Legs": 1000,
        "Shoulders": 1000, "Biceps": 1000, "Triceps": 1000,
        "Core": 1000, "Calves": 1000,
    })
    assert r.score == 100.0
    assert r.label == "Excellent Balance"


def test_heavily_skewed():
    r = volume_balance_score({"Chest": 5000, "Back": 100})
    assert r.score < 30
    assert r.dominant_muscle == "Chest"


def test_single_group():
    r = volume_balance_score({"Chest": 5000})
    assert r.score == 0
    assert r.label == "Single Focus"


def test_empty():
    r = volume_balance_score({})
    assert r.score == 0


# ─── Run all ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed:
        sys.exit(1)
