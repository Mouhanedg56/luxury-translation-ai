"""
Pytest configuration and fixtures
"""

import os

import pytest
from dotenv import load_dotenv


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may require API calls)"
    )


@pytest.fixture(scope="session", autouse=True)
def load_test_env():
    """Load environment variables for testing"""
    # Load .env file
    load_dotenv()

    # Ensure we have required API keys
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set - skipping tests requiring API")


@pytest.fixture
def client():
    """FastAPI test client with proper env setup"""
    from fastapi.testclient import TestClient

    from src.app import app

    with TestClient(app) as client:
        yield client
