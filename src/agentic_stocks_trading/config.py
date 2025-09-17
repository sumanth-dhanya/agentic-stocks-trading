import os
from pathlib import Path
from typing import Any, Literal, dict

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.agentic_stocks_trading.infrastructure.monitoring.logger_factory import get_logger, setup_logging

PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class BaseConfigSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        env_file_encoding="utf-8",
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False,
    )


class LogConfig(BaseConfigSettings):
    log_level: str = Field("DEBUG", description="Minimum log level")
    log_to_console: bool = Field(True, description="Enable stderr logging")
    log_to_file: bool = Field(True, description="Enable file logging")
    log_file: str = Field("application.log", description="Path for log output")
    intercept_modules: list[str] | str = Field(default_factory=lambda: ["uvicorn", "sqlalchemy"])

    @field_validator("intercept_modules", mode="before")
    @classmethod
    def parse_intercept_modules(cls, value: Any) -> list[str]:
        if not isinstance(value, 'str | list'):
            raise ValueError(f"intercept_modules must be a string or list, got {type(value).__name__}")
        if isinstance(value, list):
            if not all(isinstance(item, str) for item in value):
                raise ValueError("All items in intercept_modules must be strings")
            return value
        return [item.strip() for item in value.split(",")]

    @field_validator("log_file")
    @classmethod
    def validate_log_dir(cls, v: str) -> str:
        # Get the parent directory of the log file
        parent_dir = Path(v).parent

        # If there's a parent directory (not just a filename),
        # make sure it exists
        if parent_dir != Path("."):
            os.makedirs(parent_dir, exist_ok=True)

        return v


class LLMConfig(BaseConfigSettings):
    """Configuration for LLM settings."""

    llm_provider: str = Field("openai", description="LLM provider (e.g., openai, anthropic)")
    deep_think_llm: str = Field("gpt-4o", description="Powerful model for complex reasoning")
    quick_think_llm: str = Field("gpt-4o-mini", description="Fast, cheaper model for data processing")
    backend_url: str = Field("https://api.openai.com/v1", description="API backend URL")


class DebateConfig(BaseConfigSettings):
    """Configuration for debate and discussion settings."""

    max_debate_rounds: int = Field(2, description="Number of rounds for Bull vs. Bear debate", ge=1)
    max_risk_discuss_rounds: int = Field(1, description="Number of rounds for risk team debate", ge=1)
    max_recur_limit: int = Field(100, description="Maximum recursion limit", ge=1)


class ToolConfig(BaseConfigSettings):
    """Configuration for tool settings."""

    online_tools: bool = Field(True, description="Use live APIs instead of cached data")
    data_cache_dir: str = Field(f"{PROJECT_ROOT}/data/data_cache", description="Directory for caching online data")

    @field_validator("data_cache_dir")
    @classmethod
    def create_cache_dir(cls, v: str) -> str:
        """Ensure the cache directory exists."""
        cache_dir = Path(v)
        os.makedirs(cache_dir, exist_ok=True)
        return v

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TradingConfig":
        """Create a TradingConfig instance from a flat dictionary.

        This method transforms a flat dictionary with keys like 'llm_provider'
        into a nested structure needed for the TradingConfig class.

        Args:
            config_dict: A dictionary with flat key-value pairs for configuration

        Returns:
            A configured TradingConfig instance
        """
        # Transform flat dictionary into nested structure
        nested_config = {
            "results_dir": config_dict.get("results_dir", f"{PROJECT_ROOT}/data/results"),
            "llm": {
                "llm_provider": config_dict.get("llm_provider", "openai"),
                "deep_think_llm": config_dict.get("deep_think_llm", "gpt-4o"),
                "quick_think_llm": config_dict.get("quick_think_llm", "gpt-4o-mini"),
                "backend_url": config_dict.get("backend_url", "https://api.openai.com/v1"),
            },
            "debate": {
                "max_debate_rounds": config_dict.get("max_debate_rounds", 2),
                "max_risk_discuss_rounds": config_dict.get("max_risk_discuss_rounds", 1),
                "max_recur_limit": config_dict.get("max_recur_limit", 100),
            },
            "tools": {
                "online_tools": config_dict.get("online_tools", True),
                "data_cache_dir": config_dict.get("data_cache_dir", f"{PROJECT_ROOT}/data/data_cache"),
            },
        }

        # Filter out None values to use class defaults
        nested_config["llm"] = {k: v for k, v in nested_config["llm"].items() if v is not None}
        nested_config["debate"] = {k: v for k, v in nested_config["debate"].items() if v is not None}
        nested_config["tools"] = {k: v for k, v in nested_config["tools"].items() if v is not None}

        return cls(**nested_config)


class TradingConfig(BaseConfigSettings):
    """Main trading configuration."""

    results_dir: str = Field(f"{PROJECT_ROOT}/data/results", description="Directory for storing results")
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration settings")
    debate: DebateConfig = Field(default_factory=DebateConfig, description="Debate configuration settings")
    tools: ToolConfig = Field(default_factory=ToolConfig, description="Tool configuration settings")

    @field_validator("results_dir")
    @classmethod
    def create_results_dir(cls, v: str) -> str:
        """Ensure the results directory exists."""
        results_dir = Path(v)
        os.makedirs(results_dir, exist_ok=True)
        return v


class Settings(BaseConfigSettings):
    app_version: str = "0.1.0"
    debug: bool = True
    environment: Literal["development", "staging", "production"] = "development"
    service_name: str = "agentic-stocks-trading"

    OPENAI_API_KEY: str = ""
    FINNHUB_API_KEY: str = ""
    TAVILY_API_KEY: str = ""
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_TRACING: str = ""
    LANGSMITH_PROJECT: str = ""

    log: LogConfig = Field(default_factory=LogConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)


def get_settings() -> Settings:
    return Settings()


if __name__ == "__main__":
    settings = get_settings()

    # Setup logging system
    setup_logging(settings.log)

    # Get a named logger for this module
    logger = get_logger()

    logger.info(f"Env file path: {ENV_FILE_PATH / '.env'}")
    logger.info(settings.model_dump())
