import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

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
    log_file: str = Field("app.log", description="Path for log output")
    intercept_modules: list[str] = Field(default_factory=lambda: ["uvicorn", "sqlalchemy"])


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


def get_settings() -> Settings:
    return Settings()


if __name__ == "__main__":
    settings = get_settings()
    print(f"Env file path: {ENV_FILE_PATH / '.env'}")
    print(f"API Key: {settings.OPENAI_API_KEY}")
    print(settings.model_dump())
