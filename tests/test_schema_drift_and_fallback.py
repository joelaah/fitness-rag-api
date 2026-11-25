"""
Unit & Integration Tests for Schema Drift Resilience, Response Validation, and Deterministic Fallback
====================================================================================================
"""

import json
import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure D:\Rag is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api import (
    app,
    RecommendationResponse,
    RecommendationSummary,
    SingleRecommendation,
    RecoveryItem,
    NextExerciseItem,
    NextWorkout,
    LLMRecommendationOutput,
    _generate_with_retry_and_fallback,
)
from generate import generate_answer


class TestPydanticAliasChoices(unittest.TestCase):
    """Test flexible key alias handling across recommendation response models."""

    def test_single_recommendation_alias_choices(self):
        # Uses aliases: headline -> title, focus -> category, details -> description, urgency -> priority
        data = {
            "headline": "Progressive Overload",
            "focus": "progression",
            "details": "Increase weight by 2.5kg",
            "urgency": "high",
        }
        rec = SingleRecommendation.model_validate(data)
        self.assertEqual(rec.title, "Progressive Overload")
        self.assertEqual(rec.category, "progression")
        self.assertEqual(rec.description, "Increase weight by 2.5kg")
        self.assertEqual(rec.priority, "high")

    def test_recovery_item_alias_choices(self):
        # Uses aliases: headline -> title, guidance -> description
        data = {
            "headline": "Hydration Protocol",
            "guidance": "Drink at least 3 liters of water daily",
        }
        item = RecoveryItem.model_validate(data)
        self.assertEqual(item.title, "Hydration Protocol")
        self.assertEqual(item.description, "Drink at least 3 liters of water daily")

    def test_next_workout_and_exercise_alias_choices(self):
        # Uses aliases: routine -> title, rationale -> reason, movements -> exercises, exercise -> name, why -> reason
        data = {
            "routine": "Push Day Hypertrophy",
            "rationale": "Focus on horizontal pushing volume",
            "movements": [
                {"exercise": "Incline Dumbbell Press", "why": "Upper chest stimulus"}
            ],
        }
        workout = NextWorkout.model_validate(data)
        self.assertEqual(workout.title, "Push Day Hypertrophy")
        self.assertEqual(workout.reason, "Focus on horizontal pushing volume")
        self.assertEqual(len(workout.exercises), 1)
        self.assertEqual(workout.exercises[0].name, "Incline Dumbbell Press")
        self.assertEqual(workout.exercises[0].reason, "Upper chest stimulus")

    def test_recommendation_summary_alias_choices(self):
        data = {
            "total_sessions": 6,
            "volume": 15000.0,
            "sets": 36,
            "reps": 360,
            "muscle_volume": {"Chest": 7500.0, "Back": 7500.0},
            "most_trained": ["Chest", "Back"],
            "untrained_muscle_groups": ["Legs"],
            "frequency_per_week": 4.0,
            "avg_days_between_workouts": 1.75,
        }
        summary = RecommendationSummary.model_validate(data)
        self.assertEqual(summary.sessions_analyzed, 6)
        self.assertEqual(summary.total_volume, 15000.0)
        self.assertEqual(summary.total_sets, 36)
        self.assertEqual(summary.total_reps, 360)
        self.assertEqual(summary.volume_by_muscle_group["Chest"], 7500.0)
        self.assertEqual(summary.most_trained_muscle_groups, ["Chest", "Back"])
        self.assertEqual(summary.recently_untrained_muscle_groups, ["Legs"])
        self.assertEqual(summary.training_frequency_per_week, 4.0)
        self.assertEqual(summary.average_days_between_workouts, 1.75)

    def test_llm_recommendation_output_alias_choices(self):
        data = {
            "suggestions": [
                {"name": "Deload Week", "type": "recovery", "text": "Reduce intensity", "level": "normal"}
            ],
            "recovery_items": [
                {"name": "Sleep", "text": "Sleep 8 hours"}
            ],
            "suggested_workout": {
                "name": "Active Recovery Mobility",
                "purpose": "Promote bloodflow",
                "items": [
                    {"movement": "Cat Cow Stretch", "purpose": "Spinal mobility"}
                ],
            },
        }
        output = LLMRecommendationOutput.model_validate(data)
        self.assertEqual(len(output.recommendations), 1)
        self.assertEqual(output.recommendations[0].title, "Deload Week")
        self.assertEqual(output.recommendations[0].category, "recovery")
        self.assertEqual(output.recovery[0].title, "Sleep")
        self.assertEqual(output.next_workout.title, "Active Recovery Mobility")
        self.assertEqual(output.next_workout.exercises[0].name, "Cat Cow Stretch")

    def test_null_defaulted_fields_and_list_filtering(self):
        """Null values in defaulted fields and non-dict items in lists are safely handled."""
        # SingleRecommendation with category=None, priority=None
        rec = SingleRecommendation.model_validate({
            "title": "Bench Press",
            "description": "Progress load",
            "category": None,
            "priority": None,
        })
        self.assertEqual(rec.category, "general")
        self.assertEqual(rec.priority, "normal")

        # NextWorkout with exercises=None or containing stray non-dict items
        workout = NextWorkout.model_validate({
            "title": "Push",
            "reason": "Chest recovery",
            "exercises": [
                {"name": "Dips", "reason": "Triceps"},
                "Push-ups",  # string item coerced
                None,        # None filtered
                1234,        # non-dict filtered
            ],
        })
        self.assertEqual(len(workout.exercises), 2)
        self.assertEqual(workout.exercises[0].name, "Dips")
        self.assertEqual(workout.exercises[1].name, "Push-ups")

        # LLMRecommendationOutput with null lists and non-dict items
        output = LLMRecommendationOutput.model_validate({
            "recommendations": [
                {"title": "Rec 1", "description": "Desc 1"},
                None,
                "invalid string",
            ],
            "recovery": None,
            "next_workout": {
                "title": "Leg Day",
                "reason": "Lower body",
                "exercises": None,
            },
        })
        self.assertEqual(len(output.recommendations), 1)
        self.assertEqual(output.recovery, [])
        self.assertEqual(output.next_workout.exercises, [])


class TestCleanJsonOutput(unittest.TestCase):
    """Test JSON cleaning, code fence stripping, conversational text removal, and wrapper unwrapping."""

    def test_clean_json_output_with_conversational_text_and_no_fences(self):
        from api import _clean_json_output
        raw = "Certainly! Here is your workout recommendation:\n{\"recommendations\": [], \"recovery\": [], \"next_workout\": {\"title\": \"Session\", \"reason\": \"Recovery\"}}\nEnjoy your workout!"
        parsed = _clean_json_output(raw)
        self.assertIn("next_workout", parsed)
        self.assertEqual(parsed["next_workout"]["title"], "Session")

    def test_clean_json_output_unwraps_single_envelope_key(self):
        from api import _clean_json_output
        raw = '{"data": {"recommendations": [], "next_workout": {"title": "Legs", "reason": "Focus"}}}'
        parsed = _clean_json_output(raw)
        self.assertIn("next_workout", parsed)
        self.assertEqual(parsed["next_workout"]["title"], "Legs")


class TestGenerateAnswerConfig(unittest.TestCase):
    """Test response_mime_type and response_schema wiring in generate_answer."""

    @patch("generate.init_gemini_client")
    def test_passes_json_mime_type_and_schema_to_gemini(self, mock_init_client):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = '{"status": "ok"}'
        mock_client.models.generate_content.return_value = mock_resp
        mock_init_client.return_value = mock_client

        result = generate_answer(
            query="Analyze workout",
            context="Some context",
            gemini=mock_client,
            response_mime_type="application/json",
            response_schema=LLMRecommendationOutput,
        )

        self.assertEqual(result, '{"status": "ok"}')
        call_kwargs = mock_client.models.generate_content.call_args[1]
        config = call_kwargs["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertEqual(config.response_schema, LLMRecommendationOutput)


class TestRecommendationEndpointValidationAndFallback(unittest.TestCase):
    """Test API endpoint behavior under schema drift, validation failures, and cached data."""

    def setUp(self):
        self.client = TestClient(app)

    @patch("api.retrieve", return_value="Knowledge on progressive overload and recovery")
    @patch("api.get_supabase_admin")
    @patch("api.rate_limit_recommend")
    @patch("api._generate_with_retry_and_fallback")
    def test_drifted_llm_keys_parsed_successfully_returns_200_rag(
        self, mock_generate, mock_rate_limit, mock_sb_admin, mock_retrieve
    ):
        """When LLM returns aliased keys, Pydantic parses them cleanly and returns pipeline_source='rag'."""
        mock_rate_limit.return_value = "00000000-0000-0000-0000-000000000001"
        mock_sb_admin.return_value = None  # No Supabase admin

        # LLM returns drifted keys (suggestions, routine, movements, etc.)
        drifted_response = {
            "suggestions": [
                {
                    "headline": "Incorporate Quad Volume",
                    "focus": "volume",
                    "details": "Add 3 sets of squats",
                    "urgency": "high",
                }
            ],
            "recovery_items": [
                {
                    "headline": "Foam Rolling",
                    "details": "Roll thoracic spine",
                }
            ],
            "suggested_workout": {
                "routine": "Leg Hypertrophy",
                "rationale": "Legs are untrained",
                "movements": [
                    {"exercise": "Front Squat", "why": "Quad quad stimulus"}
                ],
            },
        }
        mock_generate.return_value = json.dumps(drifted_response)

        payload = {
            "force_refresh": True,
            "recent_sessions": [
                {
                    "routine_name": "Chest Day",
                    "exercises": [
                        {
                            "exercise_name": "Bench Press",
                            "primary_muscle_group": "Chest",
                            "sets": [
                                {
                                    "target_reps": 10,
                                    "completed_reps": 10,
                                    "target_weight": 100,
                                    "completed_weight": 100,
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        resp = self.client.post("/recommend", json=payload, headers={"Authorization": "Bearer test-token"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["pipeline_source"], "rag")
        self.assertEqual(len(data["recommendations"]), 1)
        self.assertEqual(data["recommendations"][0]["title"], "Incorporate Quad Volume")
        self.assertEqual(data["recommendations"][0]["category"], "volume")
        self.assertEqual(data["next_workout"]["title"], "Leg Hypertrophy")
        self.assertEqual(data["next_workout"]["exercises"][0]["name"], "Front Squat")

    @patch("api.retrieve", return_value="Knowledge on progressive overload and recovery")
    @patch("api.get_supabase_admin")
    @patch("api.rate_limit_recommend")
    @patch("api._generate_with_retry_and_fallback")
    def test_completely_invalid_llm_json_triggers_fallback_returns_200(
        self, mock_generate, mock_rate_limit, mock_sb_admin, mock_retrieve
    ):
        """When LLM completely fails validation (e.g. broken schema), server fallback triggers cleanly with HTTP 200."""
        mock_rate_limit.return_value = "00000000-0000-0000-0000-000000000001"
        mock_sb_admin.return_value = None

        # Return completely invalid schema missing next_workout and recommendations
        mock_generate.return_value = json.dumps({"unexpected_garbage": 12345})

        payload = {
            "force_refresh": True,
            "recent_sessions": [
                {
                    "routine_name": "Upper Body",
                    "exercises": [
                        {
                            "exercise_name": "Pull Up",
                            "primary_muscle_group": "Back",
                            "sets": [{"completed_reps": 8, "completed_weight": 80}],
                        }
                    ],
                }
            ],
        }

        resp = self.client.post("/recommend", json=payload, headers={"Authorization": "Bearer test-token"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        # MUST trigger deterministic fallback, NEVER crash with ResponseValidationError (500)
        self.assertEqual(data["pipeline_source"], "fallback")
        self.assertGreater(len(data["recommendations"]), 0)
        self.assertGreater(len(data["recovery"]), 0)
        self.assertTrue(data["next_workout"]["title"])

    @patch("api.retrieve", return_value="Retrieved principles")
    @patch("api.get_supabase_admin")
    @patch("api.rate_limit_recommend")
    def test_corrupted_cached_recommendation_is_discarded_and_regenerated(
        self, mock_rate_limit, mock_sb_admin, mock_retrieve
    ):
        """Corrupted/drifted cached record in Supabase is validated, discarded, and cleanly regenerated."""
        mock_rate_limit.return_value = "00000000-0000-0000-0000-000000000001"

        mock_supabase = MagicMock()
        mock_sb_admin.return_value = mock_supabase

        # Step 1 cache query returns corrupted/drifted JSON missing required next_workout field
        corrupted_cached_data = {
            "summary": {"sessions_analyzed": 1},
            # Missing next_workout, missing recommendations
            "generated_at": "2026-11-20T00:00:00Z",
        }

        def fake_table(table_name):
            t = MagicMock()
            if table_name == "ai_recommendations":
                t.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(
                    data=[{"recommendation_json": corrupted_cached_data}]
                )
            else:
                t.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
                t.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value = MagicMock(data=[])
                t.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
            return t

        mock_supabase.table.side_effect = fake_table

        # Subsequent fetch for sessions returns empty list
        # So it falls through to empty sessions handler or request payload
        payload = {"force_refresh": False, "recent_sessions": []}

        resp = self.client.post("/recommend", json=payload, headers={"Authorization": "Bearer test-token"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        # Cached was invalid, so response is NOT cached
        self.assertEqual(data["cached"], False)
        # Empty sessions handler returned clean baseline recommendations
        self.assertEqual(data["sessions_analyzed"], 0)
        self.assertEqual(data["recommendations"][0]["title"], "Begin Your Training Journey")

    @patch("api.generate_answer", side_effect=RuntimeError("503 Service Unavailable"))
    def test_generate_with_retry_fails_fast_on_api_outage(self, mock_gen):
        """Verify _generate_with_retry_and_fallback does not loop endlessly and fails fast."""
        with self.assertRaises(RuntimeError):
            _generate_with_retry_and_fallback(
                query="Query",
                context="Context",
                system_instruction="Instruction",
                max_retries=2,
            )
        # Should attempt only max_retries times, not 4 models * 2 = 8 times
        self.assertEqual(mock_gen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
