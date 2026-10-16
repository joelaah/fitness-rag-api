"""
Unit Tests for PulseFit AI RAG Backend: Math Guardrails & Schemas
==================================================================
Tests:
1. Deterministic volume / tonnage calculations (calculate_deterministic_statistics).
2. Pydantic schema validation & serialization contracts.
3. Cohere reranker cross-encoder score sorting and truncation logic.
"""

import sys
import os
from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError

# Ensure D:\Rag is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api import (
    calculate_deterministic_statistics,
    STANDARD_MUSCLE_GROUPS,
    WorkoutSetPayload,
    WorkoutExercisePayload,
    WorkoutSessionPayload,
    RecommendRequest,
    RecommendationSummary,
    SingleRecommendation,
    RecoveryItem,
    NextExerciseItem,
    NextWorkout,
    RecommendationResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
)
from retrieve import rerank_chunks


# ============================================================================
# 1. Deterministic Volume / Tonnage & Muscle Group Guardrails
# ============================================================================

class TestDeterministicMathGuardrails:
    """Verifies that mathematical calculations are strictly computed in Python,
    preventing LLM arithmetic hallucinations."""

    def test_empty_sessions_returns_zero_stats(self):
        stats = calculate_deterministic_statistics([])
        assert stats["sessions_analyzed"] == 0
        assert stats["total_volume"] == 0.0
        assert stats["total_sets"] == 0
        assert stats["total_reps"] == 0
        assert stats["volume_by_muscle_group"] == {}
        assert stats["most_trained_muscle_groups"] == []
        assert set(stats["recently_untrained_muscle_groups"]) == set(STANDARD_MUSCLE_GROUPS)
        assert stats["training_frequency_per_week"] is None
        assert stats["average_days_between_workouts"] is None

    def test_single_session_exact_tonnage_calculation(self):
        sessions = [
            {
                "routine_name": "Chest Hypertrophy",
                "started_at": "2026-09-01T10:00:00Z",
                "exercises": [
                    {
                        "exercise_name": "Incline Dumbbell Press",
                        "primary_muscle_group": "Chest",
                        "sets": [
                            {"completed_reps": 10, "completed_weight": 30.0, "is_completed": True},
                            {"completed_reps": 8, "completed_weight": 32.5, "is_completed": True},
                        ],
                    },
                    {
                        "exercise_name": "Cable Flyes",
                        "primary_muscle_group": "Chest",
                        "sets": [
                            {"completed_reps": 12, "completed_weight": 15.0, "is_completed": True},
                        ],
                    },
                ],
            }
        ]
        stats = calculate_deterministic_statistics(sessions)

        # Expected volume: (10*30.0) + (8*32.5) + (12*15.0) = 300 + 260 + 180 = 740.0 kg
        expected_volume = 740.0
        assert stats["sessions_analyzed"] == 1
        assert stats["total_volume"] == pytest.approx(expected_volume, 0.01)
        assert stats["total_sets"] == 3
        assert stats["total_reps"] == 30
        assert stats["volume_by_muscle_group"]["Chest"] == pytest.approx(expected_volume, 0.01)
        assert "Chest" in stats["most_trained_muscle_groups"]
        # All standard muscles except Chest should be in recently_untrained
        assert "Chest" not in stats["recently_untrained_muscle_groups"]
        assert "Legs" in stats["recently_untrained_muscle_groups"]
        assert "Back" in stats["recently_untrained_muscle_groups"]

    def test_skips_uncompleted_sets_deterministically(self):
        sessions = [
            {
                "routine_name": "Shoulder Session",
                "exercises": [
                    {
                        "primary_muscle_group": "Shoulders",
                        "sets": [
                            {"completed_reps": 10, "completed_weight": 20.0, "is_completed": True},
                            {"completed_reps": 10, "completed_weight": 20.0, "is_completed": False},  # Skipped
                        ],
                    }
                ],
            }
        ]
        stats = calculate_deterministic_statistics(sessions)
        assert stats["total_sets"] == 1
        assert stats["total_reps"] == 10
        assert stats["total_volume"] == pytest.approx(200.0, 0.01)

    def test_multi_muscle_sorting_by_volume_descending(self):
        sessions = [
            {
                "routine_name": "Full Body Day",
                "exercises": [
                    {
                        "primary_muscle_group": "Legs",
                        "sets": [
                            {"completed_reps": 10, "completed_weight": 100.0, "is_completed": True},  # 1000 kg
                        ],
                    },
                    {
                        "primary_muscle_group": "Chest",
                        "sets": [
                            {"completed_reps": 10, "completed_weight": 80.0, "is_completed": True},   # 800 kg
                        ],
                    },
                    {
                        "primary_muscle_group": "Back",
                        "sets": [
                            {"completed_reps": 10, "completed_weight": 60.0, "is_completed": True},   # 600 kg
                        ],
                    },
                ],
            }
        ]
        stats = calculate_deterministic_statistics(sessions)
        assert stats["total_volume"] == pytest.approx(2400.0, 0.01)
        # Most trained must be ordered: Legs (1000) > Chest (800) > Back (600)
        assert stats["most_trained_muscle_groups"] == ["Legs", "Chest", "Back"]
        assert "Legs" not in stats["recently_untrained_muscle_groups"]
        assert "Chest" not in stats["recently_untrained_muscle_groups"]
        assert "Back" not in stats["recently_untrained_muscle_groups"]
        assert "Shoulders" in stats["recently_untrained_muscle_groups"]

    def test_interval_and_frequency_calculations(self):
        sessions = [
            {
                "routine_name": "Day 1",
                "started_at": "2026-09-01T08:00:00Z",
                "exercises": [],
            },
            {
                "routine_name": "Day 2",
                "started_at": "2026-09-03T08:00:00Z",  # 2 days later
                "exercises": [],
            },
            {
                "routine_name": "Day 3",
                "started_at": "2026-09-05T08:00:00Z",  # 2 days later
                "exercises": [],
            },
        ]
        stats = calculate_deterministic_statistics(sessions)
        assert stats["sessions_analyzed"] == 3
        assert stats["average_days_between_workouts"] == 2.0
        assert stats["training_frequency_per_week"] is not None
        assert stats["training_frequency_per_week"] > 0

    def test_relational_supabase_keys_format(self):
        """Verifies calculation works on database schema keys (workout_exercises & workout_sets)."""
        sessions = [
            {
                "routine_name": "Back & Biceps Day",
                "workout_exercises": [
                    {
                        "exercise_name": "Lat Pulldown",
                        "primary_muscle_group": "Back",
                        "workout_sets": [
                            {"completed_reps": 12, "completed_weight": 55.0, "is_completed": True},
                            {"completed_reps": 10, "completed_weight": 60.0, "is_completed": True},
                        ],
                    }
                ],
            }
        ]
        stats = calculate_deterministic_statistics(sessions)
        assert stats["sessions_analyzed"] == 1
        assert stats["total_sets"] == 2
        assert stats["total_reps"] == 22
        # (12*55) + (10*60) = 660 + 600 = 1260.0 kg
        assert stats["total_volume"] == pytest.approx(1260.0, 0.01)
        assert stats["volume_by_muscle_group"]["Back"] == pytest.approx(1260.0, 0.01)

    def test_zero_weight_bodyweight_exercises_handling(self):
        """Verifies that 0kg bodyweight exercises do not cause math errors or zero division."""
        sessions = [
            {
                "routine_name": "Bodyweight Calisthenics",
                "exercises": [
                    {
                        "exercise_name": "Pull-ups",
                        "primary_muscle_group": "Back",
                        "sets": [
                            {"completed_reps": 10, "completed_weight": 0.0, "is_completed": True},
                            {"completed_reps": 8, "completed_weight": 0.0, "is_completed": True},
                        ],
                    }
                ],
            }
        ]
        stats = calculate_deterministic_statistics(sessions)
        assert stats["total_sets"] == 2
        assert stats["total_reps"] == 18
        assert stats["total_volume"] == 0.0

    def test_custom_unmapped_muscle_group(self):
        """Verifies custom muscle groups are normalized to standard groups where applicable or preserved cleanly."""
        sessions = [
            {
                "routine_name": "Arm & Neck Isolation",
                "exercises": [
                    {
                        "exercise_name": "Wrist Curls",
                        "primary_muscle_group": "Forearms",  # Substring matches 'Arms'
                        "sets": [
                            {"completed_reps": 15, "completed_weight": 10.0, "is_completed": True},
                        ],
                    },
                    {
                        "exercise_name": "Neck Extensions",
                        "primary_muscle_group": "Neck",  # Not in standard groups -> preserved as Neck
                        "sets": [
                            {"completed_reps": 10, "completed_weight": 5.0, "is_completed": True},
                        ],
                    },
                ],
            }
        ]
        stats = calculate_deterministic_statistics(sessions)
        assert stats["total_volume"] == pytest.approx(200.0, 0.01)
        # Forearms normalizes to standard 'Arms'
        assert stats["volume_by_muscle_group"]["Arms"] == pytest.approx(150.0, 0.01)
        # Neck is preserved cleanly as custom group
        assert stats["volume_by_muscle_group"]["Neck"] == pytest.approx(50.0, 0.01)
        assert "Arms" in stats["most_trained_muscle_groups"]


# ============================================================================
# 2. Pydantic Schema Contracts & Validation Guardrails
# ============================================================================

class TestPydanticSchemas:
    """Verifies that API boundary schemas properly validate and enforce typed contracts."""

    def test_valid_workout_set_defaults(self):
        set_payload = WorkoutSetPayload(set_number=1, completed_reps=10, completed_weight=50.0)
        assert set_payload.set_number == 1
        assert set_payload.completed_reps == 10
        assert set_payload.completed_weight == 50.0
        assert set_payload.is_completed is True

    def test_valid_recommend_response_schema(self):
        summary = RecommendationSummary(
            sessions_analyzed=3,
            total_volume=4500.0,
            total_sets=12,
            total_reps=120,
            volume_by_muscle_group={"Chest": 2500.0, "Back": 2000.0},
            most_trained_muscle_groups=["Chest", "Back"],
            recently_untrained_muscle_groups=["Legs", "Shoulders"],
            training_frequency_per_week=3.0,
            average_days_between_workouts=2.3,
        )
        recommendations = [
            SingleRecommendation(
                title="Progressive Overload on Bench",
                category="hypertrophy",
                description="Increase dumbbell load by 2kg next week.",
                priority="high",
            )
        ]
        recovery = [
            RecoveryItem(
                title="Hydration & Sleep",
                description="Target 8 hours of sleep for central nervous system repair.",
            )
        ]
        next_workout = NextWorkout(
            title="Lower Body Quad Focus",
            reason="Legs were untrained during the last 7-day microcycle.",
            exercises=[
                NextExerciseItem(name="Barbell Back Squat", reason="Compound quad loading"),
                NextExerciseItem(name="Romanian Deadlift", reason="Hamstring eccentric strength"),
            ],
        )

        resp = RecommendationResponse(
            summary=summary,
            recommendations=recommendations,
            recovery=recovery,
            next_workout=next_workout,
            generated_at="2026-10-09T12:00:00Z",
            sessions_analyzed=3,
            cached=False,
            pipeline_source="rag",
        )

        dumped = resp.model_dump()
        assert dumped["sessions_analyzed"] == 3
        assert dumped["pipeline_source"] == "rag"
        assert dumped["next_workout"]["exercises"][0]["name"] == "Barbell Back Squat"

    def test_chat_message_validation(self):
        chat_req = ChatRequest(message="What is the best rep range for hypertrophy?")
        assert chat_req.message == "What is the best rep range for hypertrophy?"
        assert chat_req.history is None

        # Message cannot be empty
        with pytest.raises(ValidationError):
            ChatRequest(message="")

        chat_resp = ChatResponse(
            reply="The evidence-based rep range for hypertrophy is 6-12 reps near failure (1-3 RIR).",
            sources=["Schoenfeld et al., 2021"],
            pipeline_source="rag",
        )
        assert len(chat_resp.sources) == 1
        assert chat_resp.pipeline_source == "rag"


# ============================================================================
# 3. Two-Stage Reranker Sorting Guardrails
# ============================================================================

class TestRerankerSortingLogic:
    """Verifies cross-encoder precision reranker sorting invariants."""

    def test_rerank_chunks_empty_input(self):
        assert rerank_chunks("test query", []) == []

    def test_rerank_chunks_sorts_descending_by_relevance_score(self):
        chunks = [
            "Doc A: General cardio guidelines",
            "Doc B: High-intensity hypertrophy progressive overload research",
            "Doc C: Plant-based protein synthesis study",
        ]

        # Mock Cohere V2 client and response
        mock_cohere = MagicMock()
        mock_result_0 = MagicMock(index=1, relevance_score=0.95)  # Doc B is most relevant
        mock_result_1 = MagicMock(index=0, relevance_score=0.60)  # Doc A is 2nd
        mock_result_2 = MagicMock(index=2, relevance_score=0.30)  # Doc C is 3rd

        mock_response = MagicMock(results=[mock_result_0, mock_result_1, mock_result_2])
        mock_cohere.rerank.return_value = mock_response

        reranked = rerank_chunks(
            query="hypertrophy overload",
            chunks=chunks,
            co=mock_cohere,
            top_n=3,
        )

        # Expected order: Doc B, Doc A, Doc C
        assert len(reranked) == 3
        assert reranked[0] == chunks[1]
        assert reranked[1] == chunks[0]
        assert reranked[2] == chunks[2]

    def test_rerank_chunks_truncates_to_top_n(self):
        chunks = [f"Doc {i}" for i in range(10)]

        mock_cohere = MagicMock()
        # Top 3 results from 10 candidates
        mock_cohere.rerank.return_value = MagicMock(
            results=[
                MagicMock(index=8, relevance_score=0.99),
                MagicMock(index=2, relevance_score=0.88),
                MagicMock(index=5, relevance_score=0.77),
            ]
        )

        reranked = rerank_chunks(
            query="test query",
            chunks=chunks,
            co=mock_cohere,
            top_n=3,
        )

        assert len(reranked) == 3
        assert reranked[0] == "Doc 8"
        assert reranked[1] == "Doc 2"
        assert reranked[2] == "Doc 5"

    def test_rerank_chunks_top_n_exceeds_chunk_count(self):
        chunks = ["Single candidate chunk"]
        mock_cohere = MagicMock()
        mock_cohere.rerank.return_value = MagicMock(
            results=[MagicMock(index=0, relevance_score=0.91)]
        )

        reranked = rerank_chunks(
            query="test",
            chunks=chunks,
            co=mock_cohere,
            top_n=5,
        )

        assert len(reranked) == 1
        assert reranked[0] == "Single candidate chunk"
