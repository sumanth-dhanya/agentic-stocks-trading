import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.agentic_stocks_trading.config import (
    DebateConfig,
    LLMConfig,
    LogConfig,
    ToolConfig,
    TradingConfig,
    get_settings,
)


class TestLogConfig:
    def test_default_values(self):
        config = LogConfig()
        assert config.log_level == "INFO"
        assert config.log_to_console is True
        assert config.log_to_file is True
        assert str(config.log_file).endswith("logs/service.log")
        assert config.intercept_modules == ["uvicorn", "sqlalchemy"]

    def test_intercept_modules_validator_with_string(self):
        config = LogConfig(intercept_modules="uvicorn,sqlalchemy,fastapi")
        assert config.intercept_modules == ["uvicorn", "sqlalchemy", "fastapi"]

    def test_intercept_modules_validator_with_list(self):
        config = LogConfig(intercept_modules=["uvicorn", "sqlalchemy", "fastapi"])
        assert config.intercept_modules == ["uvicorn", "sqlalchemy", "fastapi"]

    def test_intercept_modules_validator_with_invalid_type(self):
        with pytest.raises(ValueError, match="intercept_modules must be a string or list"):
            LogConfig(intercept_modules=123)

    def test_intercept_modules_validator_with_invalid_list_items(self):
        with pytest.raises(ValueError, match="All items in intercept_modules must be strings"):
            LogConfig(intercept_modules=["uvicorn", 123, "sqlalchemy"])

    def test_log_file_validator_creates_directory(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_file = str(log_dir / "test.log")

        # Directory should not exist yet
        assert not log_dir.exists()

        config = LogConfig(log_file=log_file)
        # Directory should be created
        assert log_dir.exists()
        # Also test the config object itself
        assert config.log_file == log_file
        # File should not be created
        assert not Path(log_file).exists()


class TestLLMConfig:
    def test_default_values(self):
        config = LLMConfig()
        assert config.llm_provider == "openai"
        assert config.deep_think_llm == "gpt-4o"
        assert config.quick_think_llm == "gpt-4o-mini"
        assert config.backend_url == "https://api.openai.com/v1"


class TestDebateConfig:
    def test_default_values(self):
        config = DebateConfig()
        assert config.max_debate_rounds == 2
        assert config.max_risk_discuss_rounds == 1
        assert config.max_recur_limit == 100

    def test_validation_for_positive_values(self):
        with pytest.raises(ValueError):
            DebateConfig(max_debate_rounds=0)

        with pytest.raises(ValueError):
            DebateConfig(max_risk_discuss_rounds=0)

        with pytest.raises(ValueError):
            DebateConfig(max_recur_limit=0)


class TestToolConfig:
    def test_default_values(self):
        config = ToolConfig()
        assert config.online_tools is True
        assert "data_cache" in config.data_cache_dir

    def test_data_cache_dir_validator_creates_directory(self, tmp_path):
        cache_dir = str(tmp_path / "cache")

        # Directory should not exist yet
        assert not Path(cache_dir).exists()

        config = ToolConfig(data_cache_dir=cache_dir)
        assert config.data_cache_dir == cache_dir

        # Directory should be created
        assert Path(cache_dir).exists()


class TestTradingConfig:
    def test_default_values(self):
        config = TradingConfig()
        assert "results" in config.results_dir
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.debate, DebateConfig)
        assert isinstance(config.tools, ToolConfig)

    def test_results_dir_validator_creates_directory(self, tmp_path):
        results_dir = str(tmp_path / "results")

        # Directory should not exist yet
        assert not Path(results_dir).exists()

        config = TradingConfig(results_dir=results_dir)
        assert config.results_dir == results_dir

        # Directory should be created
        assert Path(results_dir).exists()


class TestSettings:
    def test_default_values(self):
        settings = get_settings()
        assert settings.app_version == "0.1.0"
        assert settings.debug is True
        assert settings.environment == "development"
        assert settings.service_name == "agentic-stocks-trading"
        assert settings.OPENAI_API_KEY != ""
        assert isinstance(settings.log, LogConfig)
        assert isinstance(settings.trading, TradingConfig)

    @patch.dict(os.environ, {"APP_VERSION": "1.0.0", "DEBUG": "false"})
    def test_env_variables_override(self):
        settings = get_settings()
        assert settings.app_version == "1.0.0"
        assert settings.debug is False

    @patch.dict(os.environ, {"LOG__LOG_LEVEL": "INFO", "LOG__LOG_TO_CONSOLE": "false"})
    def test_nested_env_variables(self):
        settings = get_settings()
        assert settings.log.log_level == "INFO"
        assert settings.log.log_to_console is False

    @patch.dict(os.environ, {"TRADING__LLM__LLM_PROVIDER": "anthropic", "TRADING__DEBATE__MAX_DEBATE_ROUNDS": "3"})
    def test_deeply_nested_env_variables(self):
        settings = get_settings()
        assert settings.trading.llm.llm_provider == "anthropic"
        assert settings.trading.debate.max_debate_rounds == 3


def test_settings():
    settings = get_settings()
    assert settings is not None
    assert hasattr(settings, "app_version")
    assert hasattr(settings, "log")
    assert hasattr(settings, "trading")
    assert settings.service_name == "agentic-stocks-trading"
