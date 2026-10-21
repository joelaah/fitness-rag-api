"""
Unit and Integration Tests for PulseFit AI Nutrition & Macro Scanner Router
============================================================================
Tests:
1. Deterministic Atwater energy calculation (4P + 4C + 9F).
2. Food items macro guardrails and summation integrity.
3. Image analysis endpoint with multipart image upload.
4. Bad media types and corrupted file error handling.
5. In-memory sliding-window rate limiter enforcement.
6. Meal logging and daily summary endpoints.
"""

import io
import json
import time
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from api import app
from nutrition_router import (
    calculate_atwater_energy,
    apply_atwater_guardrails,
    NutritionRateLimiter,
    FoodItem,
    NutritionAnalysisResponse,
)

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_dummy_image_bytes(format: str = "JPEG", size=(100, 100), color=(200, 100, 50)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()

# ---------------------------------------------------------------------------
# 1. Atwater Energy Factor Unit Tests
# ---------------------------------------------------------------------------

class TestAtwaterEnergyCalculations:
    def test_zero_macros(self):
        cal = calculate_atwater_energy(0.0, 0.0, 0.0)
        assert cal == 0.0

    def test_pure_protein(self):
        # 50g protein * 4 = 200 kcal
        assert calculate_atwater_energy(50.0, 0.0, 0.0) == 200.0

    def test_pure_carbohydrates(self):
        # 75g carbs * 4 = 300 kcal
        assert calculate_atwater_energy(0.0, 75.0, 0.0) == 300.0

    def test_pure_fat(self):
        # 20g fat * 9 = 180 kcal
        assert calculate_atwater_energy(0.0, 0.0, 20.0) == 180.0

    def test_mixed_macronutrient_meal(self):
        # 40g Protein (160), 50g Carbs (200), 15g Fat (135) => 495 kcal
        assert calculate_atwater_energy(40.0, 50.0, 15.0) == 495.0

    def test_rounding_precision(self):
        # 33.3g P (133.2) + 44.4g C (177.6) + 12.2g F (109.8) = 420.6 kcal
        assert calculate_atwater_energy(33.3, 44.4, 12.2) == 420.6

# ---------------------------------------------------------------------------
# 2. Atwater Guardrails and Food Item Normalization
# ---------------------------------------------------------------------------

class TestFoodItemGuardrails:
    def test_sanitize_and_compute_item_totals(self):
        raw_items = [
            {
                "name": "Grilled Chicken",
                "portion_g": 150,
                "protein_g": 46.5,
                "carbs_g": 0.0,
                "fat_g": 5.5,
                "confidence": 0.95,
                "category": "protein",
            },
            {
                "name": "White Rice",
                "portion_g": 200,
                "protein_g": 5.0,
                "carbs_g": 56.0,
                "fat_g": 0.6,
                "confidence": 0.90,
                "category": "carb",
            },
        ]
        items, tot_cal, tot_p, tot_c, tot_f, atwater_cal = apply_atwater_guardrails(raw_items, "Chicken & Rice")

        assert len(items) == 2
        assert tot_p == 51.5  # 46.5 + 5.0
        assert tot_c == 56.0  # 0.0 + 56.0
        assert tot_f == 6.1   # 5.5 + 0.6
        # Expected Atwater = 51.5*4 + 56.0*4 + 6.1*9 = 206 + 224 + 54.9 = 484.9
        assert atwater_cal == 484.9
        assert tot_cal == atwater_cal
        # Item 1 calories = 46.5*4 + 5.5*9 = 186 + 49.5 = 235.5
        assert items[0].calories == 235.5
        # Item 2 calories = 5*4 + 56*4 + 0.6*9 = 20 + 224 + 5.4 = 249.4
        assert items[1].calories == 249.4

    def test_handles_missing_or_negative_fields_safely(self):
        raw_items = [
            {"name": "Mystery Dish", "portion_g": -50, "protein_g": -10, "carbs_g": 20},
        ]
        items, tot_cal, tot_p, tot_c, tot_f, atwater_cal = apply_atwater_guardrails(raw_items)
        assert len(items) == 1
        assert items[0].portion_g == 0.0
        assert items[0].protein_g == 0.0
        assert items[0].carbs_g == 20.0
        assert items[0].fat_g == 0.0
        assert tot_cal == 80.0  # 20 * 4

# ---------------------------------------------------------------------------
# 3. Nutrition Rate Limiter Tests
# ---------------------------------------------------------------------------

class TestNutritionRateLimiter:
    def test_sliding_window_rate_limiting(self):
        limiter = NutritionRateLimiter()
        key = "test_user_key"

        # Allowed up to 3 requests in 10-second window
        for _ in range(3):
            allowed, retry = limiter.is_allowed(key, max_requests=3, window_seconds=10.0)
            assert allowed is True
            assert retry == 0

        # 4th request must be rejected
        allowed, retry = limiter.is_allowed(key, max_requests=3, window_seconds=10.0)
        assert allowed is False
        assert retry > 0

# ---------------------------------------------------------------------------
# 4. FastAPI Integration Tests (/nutrition endpoints)
# ---------------------------------------------------------------------------

class TestNutritionEndpoints:
    @patch("nutrition_router.genai.Client")
    def test_analyze_image_success(self, mock_client_cls):
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({
            "meal_name": "Grilled Chicken and Broccoli",
            "items": [
                {
                    "name": "Grilled Chicken Breast",
                    "portion_g": 180.0,
                    "protein_g": 55.0,
                    "carbs_g": 0.0,
                    "fat_g": 6.0,
                    "confidence": 0.95,
                    "category": "protein",
                },
                {
                    "name": "Steamed Broccoli",
                    "portion_g": 120.0,
                    "protein_g": 3.0,
                    "carbs_g": 8.0,
                    "fat_g": 0.5,
                    "confidence": 0.92,
                    "category": "vegetable",
                },
            ],
            "health_score": 95,
            "nutritional_summary": "Lean protein and high fiber meal.",
            "scientific_notes": "Volumetric density standard applied.",
        })
        mock_instance.models.generate_content.return_value = mock_resp

        img_bytes = create_dummy_image_bytes("JPEG")
        response = client.post(
            "/nutrition/analyze-image",
            files={"file": ("meal.jpg", img_bytes, "image/jpeg")},
            data={"meal_hint": "Healthy lunch bowl"},
            headers={"Authorization": "Bearer dev-user-nutrition-test"},
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["meal_name"] == "Grilled Chicken and Broccoli"
        assert len(data["items"]) == 2
        assert data["total_protein_g"] == 58.0
        assert data["total_carbs_g"] == 8.0
        assert data["total_fat_g"] == 6.5
        # 58*4 + 8*4 + 6.5*9 = 232 + 32 + 58.5 = 322.5
        assert data["atwater_calories"] == 322.5
        assert data["total_calories"] == 322.5
        assert data["health_score"] == 95

    def test_analyze_image_png_format_fallback(self):
        # Without mock / with fallback
        img_bytes = create_dummy_image_bytes("PNG")
        with patch("nutrition_router.GEMINI_API_KEY", ""):
            response = client.post(
                "/nutrition/analyze-image",
                files={"file": ("plate.png", img_bytes, "image/png")},
                headers={"Authorization": "Bearer dev-user-nutrition-test"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["total_calories"] > 0
            assert len(data["items"]) > 0

    def test_analyze_image_rejects_corrupted_file(self):
        corrupted_bytes = b"not-a-valid-image-stream-content"
        response = client.post(
            "/nutrition/analyze-image",
            files={"file": ("corrupted.jpg", corrupted_bytes, "image/jpeg")},
            headers={"Authorization": "Bearer dev-user-nutrition-test"},
        )
        assert response.status_code == 400
        assert "corrupted" in response.json()["detail"].lower()

    def test_analyze_image_rejects_unsupported_mime(self):
        response = client.post(
            "/nutrition/analyze-image",
            files={"file": ("notes.txt", b"some text", "text/plain")},
            headers={"Authorization": "Bearer dev-user-nutrition-test"},
        )
        assert response.status_code == 415

    def test_log_meal_and_fetch_today(self):
        log_payload = {
            "meal_name": "Post-Workout Fuel Bowl",
            "meal_type": "lunch",
            "total_calories": 520.0,
            "total_protein_g": 48.0,
            "total_carbs_g": 55.0,
            "total_fat_g": 12.0,
            "health_score": 90,
            "notes": "Scanned with PulseFit AI scanner",
        }
        log_resp = client.post(
            "/nutrition/log",
            json=log_payload,
            headers={"Authorization": "Bearer dev-user-nutrition-test"},
        )
        assert log_resp.status_code == 200
        assert log_resp.json()["status"] == "ok"

        # Fetch today's summary
        today_resp = client.get(
            "/nutrition/today",
            headers={"Authorization": "Bearer dev-user-nutrition-test"},
        )
        assert today_resp.status_code == 200
        data = today_resp.json()
        assert "date" in data
        assert "target_calories" in data
        assert data["target_calories"] == 2000
