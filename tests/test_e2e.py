"""
End-to-end tests for the complete luxury translation workflow
"""

import pytest


class TestEndToEnd:
    """End-to-end workflow tests"""

    def test_api_health(self, client):
        """Test API health endpoint"""
        response = client.get("/")

        assert response.status_code == 200

        result = response.json()
        assert result["service"] == "Pydantic AI - Luxury Fashion Translation API"
        assert result["status"] == "healthy"
        assert "agents" in result

    @pytest.mark.slow
    def test_complete_translation_workflow(self, client):
        """Test complete translation workflow from input to output"""

        content = """
        <h1>Venetian Artisan - Luxury Collection</h1>
        <p>Discover our exclusive Italian leather handbags.</p>
        """

        payload = {
            "content": content,
            "source_language": "en",
            "target_language": "fr",
            "brand_name": "Venetian Artisan",
            "brand_voice": "Sophisticated Italian luxury",
            "target_market": "French luxury consumers",
            "brand_glossary": {"luxury": "luxe", "heritage": "patrimoine"},
            "quality_requirements": {
                "accuracy": 0.85,
                "fluency": 0.80,
                "brand_consistency": 0.85,
                "cultural_appropriateness": 0.80,
                "luxury_positioning": 0.85,
            },
        }

        response = client.post("/translate/pydantic-ai", json=payload)

        assert response.status_code == 200

        result = response.json()

        # Verify workflow execution
        assert "task_id" in result
        assert result["workflow_status"] in ["completed", "reviewing"]
        assert "agents_used" in result
        assert len(result["agents_used"]) > 0

        # Verify translation results
        if result["workflow_status"] == "completed":
            assert "translated_content" in result
            assert "segment_results" in result

    def test_agent_status(self, client):
        """Test agent status endpoint"""
        response = client.get("/agents/status")

        assert response.status_code == 200

        result = response.json()
        assert "agents" in result
        assert "workflow_orchestrator" in result
        assert "system_health" in result

    def test_segmentation_then_translation_flow(self, client):
        """Test segmentation followed by translation"""

        content = "<h1>Luxury Handbags</h1><p>Handcrafted Italian leather goods.</p>"

        # Step 1: Segment content
        seg_payload = {
            "content": content,
            "segmentation_method": "basic",
            "brand_name": "Venetian Artisan",
        }

        seg_response = client.post("/content/segment", json=seg_payload)
        assert seg_response.status_code == 200

        seg_result = seg_response.json()
        assert seg_result["total_segments"] > 0

        # Step 2: Translate using the same content
        trans_payload = {
            "content": content,
            "source_language": "en",
            "target_language": "fr",
            "brand_name": "Venetian Artisan",
            "brand_voice": "Sophisticated luxury",
            "target_market": "French consumers",
        }

        trans_response = client.post("/translate/pydantic-ai", json=trans_payload)
        assert trans_response.status_code == 200

        trans_result = trans_response.json()
        assert "task_id" in trans_result
