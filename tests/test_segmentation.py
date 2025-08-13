"""
Tests for content segmentation functionality
"""

import pytest

from src.app import ContentSegmentationService


class TestSegmentationService:
    """Unit tests for ContentSegmentationService"""

    def test_html_segmentation(self):
        """Test HTML content segmentation"""
        html_content = "<h1>Luxury Collection</h1><p>Premium handbags for discerning customers.</p>"

        segments = ContentSegmentationService.auto_segment_content(html_content)

        assert len(segments) >= 2
        assert any("Luxury Collection" in seg.text for seg in segments)
        assert any("Premium handbags" in seg.text for seg in segments)

        # Check priorities (headers should have priority 1)
        header_seg = next(seg for seg in segments if "Luxury Collection" in seg.text)
        assert header_seg.priority == 1
        assert header_seg.content_type == "title"

    def test_markdown_segmentation(self):
        """Test Markdown content segmentation"""
        md_content = "# Venetian Artisan\n\nExperience luxury craftsmanship."

        segments = ContentSegmentationService.auto_segment_content(md_content)

        assert len(segments) >= 2
        assert any("Venetian Artisan" in seg.text for seg in segments)

    def test_empty_content(self):
        """Test empty content handling"""
        segments = ContentSegmentationService.auto_segment_content("")
        assert len(segments) == 0


class TestSegmentationAPI:
    """Integration tests for segmentation API endpoint"""

    def test_basic_segmentation_api(self, client):
        """Test basic segmentation via API"""
        payload = {
            "content": "<h1>Luxury Handbag Collection</h1><p>Discover our exclusive Italian leather handbags.</p>",
            "segmentation_method": "basic",
        }

        response = client.post("/content/segment", json=payload)

        assert response.status_code == 200

        result = response.json()
        assert result["segmentation_method"] == "basic"
        assert result["total_segments"] > 0
        assert len(result["segments"]) > 0

        # Check segment structure
        for segment in result["segments"]:
            assert "id" in segment
            assert "text" in segment
            assert "priority" in segment
            assert "content_type" in segment

    @pytest.mark.slow
    def test_agent_segmentation_api(self, client):
        """Test AI agent segmentation via API"""
        payload = {
            "content": "<h1>Venetian Artisan Collection</h1><p>Experience luxury craftsmanship.</p>",
            "segmentation_method": "agent",
            "brand_name": "Venetian Artisan",
            "brand_voice": "Sophisticated Italian luxury",
            "target_market": "French luxury consumers",
        }

        response = client.post("/content/segment", json=payload)

        assert response.status_code == 200

        result = response.json()
        assert result["segmentation_method"] == "agent"
        assert result["total_segments"] > 0

        # Agent segmentation should include human review flags
        for segment in result["segments"]:
            assert "requires_human_review" in segment
