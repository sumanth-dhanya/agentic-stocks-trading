from src.agentic_stocks_trading.config import get_settings
from src.agentic_stocks_trading.infrastructure.monitoring.logger_factory import get_logger, setup_logging

_settings = get_settings()
setup_logging(_settings.log)
logger = get_logger()

# Expose logger factory for convenience
__all__ = [
    "logger",
    "setup_logging",
    "get_settings",
    "get_logger",
]
