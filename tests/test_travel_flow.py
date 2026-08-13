import unittest

from fastapi.testclient import TestClient

from app.api import create_app
from app.graph.runner import run_travel_graph


class TravelFlowTests(unittest.TestCase):

    def test_homepage_renders(self) -> None:
        """The root endpoint should return 200 and contain 'Travel Planner'."""
        app = create_app()
        client = TestClient(app)
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Travel Planner", response.content)

    def test_plan_endpoint_returns_itinerary(self) -> None:
        """POST /plan should return a JSON body that includes 'itinerary'."""
        app = create_app()
        client = TestClient(app)
        response = client.post(
            "/plan?user_id=test-user",
            json={
                "destination": "Paris",
                "origin": "London",
                "budget": "1500",
                "currency": "EUR",
                "start_date": "2026-09-01",
                "end_date": "2026-09-05",
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("itinerary", data)
        self.assertIsInstance(data["itinerary"], str)
        self.assertGreater(len(data["itinerary"]), 0)
