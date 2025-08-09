"""
Shared test configuration and fixtures.
"""

import os
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def assets_dir():
    """Return the path to the test assets directory."""
    return Path(__file__).parent / "assets"


@pytest.fixture(scope="session")
def has_openai_api_key():
    """Check if OpenAI API key is available for integration tests."""
    api_key = os.getenv('OPENAI_API_KEY')
    return api_key is not None and api_key != 'test-key-for-unit-tests'


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "requires_api_key: test requires OpenAI API key")
    config.addinivalue_line("markers", "integration: integration test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark tests that require API keys
        if "real_llm" in item.name or "api_key" in str(item.function):
            item.add_marker(pytest.mark.requires_api_key)
            item.add_marker(pytest.mark.integration)
