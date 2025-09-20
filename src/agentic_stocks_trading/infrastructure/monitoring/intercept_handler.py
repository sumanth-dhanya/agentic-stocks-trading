import logging

from loguru import logger


class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        module_label = getattr(record, "name", None) or getattr(record, "module", None) or "unknown"

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_globals.get("__name__") == __name__:
            frame = frame.f_back
            depth += 1
        logger.bind(module=module_label).opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
