from loguru import logger
from .intercept_handler import InterceptHandler
import sys, logging


def setup_logging(config=None):
    logger.remove()
    format_console = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<cyan>{module}</cyan>:<cyan>{function}</cyan> | "
        "<level>{level}</level> | <level>{message}</level>"
    )
    format_file = (
        "{{"  # Double braces for literal '{'
        "\"time\":\"{time:YYYY-MM-DDTHH:mm:ss}\","
        "\"level\":\"{level}\","
        "\"module\":\"{module}\","
        "\"message\":\"{message}\""
        "}}"  # Double braces for literal '}'
    )

    # Default values if config is not provided
    log_level = "DEBUG"
    log_to_console = True
    log_to_file = False
    log_file = "application.log"
    intercept_modules = ["uvicorn", "sqlalchemy"]

    # Override with provided config if available
    if config:
        log_level = config.log_level
        log_to_console = config.log_to_console
        log_to_file = config.log_to_file
        log_file = config.log_file
        intercept_modules = config.intercept_modules

    if log_to_console:
        logger.add(sys.stderr, level=log_level.upper(), format=format_console, enqueue=True)
    if log_to_file:
        logger.add(log_file, level=log_level.upper(), format=format_file,
                   enqueue=True, rotation="1 week", serialize=True)
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for mod in intercept_modules:
        logging.getLogger(mod).handlers = [InterceptHandler()]
        logging.getLogger(mod).propagate = False
    return logger


def get_logger(name=None):
    """Get a logger instance, optionally with a specific name."""
    if name:
        # Use the provided name
        return logger.bind(name=name)
    else:
        # Try to get the caller's module name
        import inspect
        frame = inspect.currentframe().f_back
        module_name = frame.f_globals['__name__']
        if module_name == '__main__':
            # Try to get the actual filename if running as main
            import os
            module_name = os.path.basename(frame.f_globals['__file__']).replace('.py', '')

        return logger.bind(module=module_name)

