# tests/conftest.py
import pytest
import os
from pathlib import Path
from typing import Dict, Any

# Mock the logger functions to avoid the import error
import sys
from unittest.mock import MagicMock

# Create mocks for the logging functions
sys.modules['src.agentic_stocks_trading.infrastructure.monitoring.logger_factory'] = MagicMock()
sys.modules['src.agentic_stocks_trading.infrastructure.monitoring.logger_factory'].get_logger = MagicMock(
    return_value=MagicMock())
sys.modules['src.agentic_stocks_trading.infrastructure.monitoring.logger_factory'].setup_logging = MagicMock()

# Now import your modules
from src.agentic_stocks_trading.config import Settings, TradingConfig


@pytest.fixture
def temp_env_file(tmp_path):
    """Create a temporary .env file for testing."""
    env_file = tmp_path / ".env"
    env_file.write_text("LOG_LEVEL=INFO\nDEBUG=true\n")
    return env_file


@pytest.fixture
def mock_trading_config_dict() -> Dict[str, Any]:
    """Return a sample trading config dictionary for testing."""
    return {
        "results_dir": "./test_results",
        "llm_provider": "test-provider",
        "deep_think_llm": "test-model",
        "quick_think_llm": "test-quick-model",
        "max_debate_rounds": 4,
        "max_risk_discuss_rounds": 2,
        "max_recur_limit": 50,
        "online_tools": False,
        "data_cache_dir": "./test_cache"
    }
