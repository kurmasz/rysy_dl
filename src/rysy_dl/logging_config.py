"""
logging_config.py

Custom logging setup shared across the project:
  - Adds a PROGRESS level (between INFO and WARNING).
  - Defines a Logger subclass with a `.progress()` method, registered via
    `logging.setLoggerClass` so type checkers (and IDEs) can see the method,
    instead of monkey-patching `logging.Logger` directly.

Any module that wants a logger with `.progress()` should do:

    from .logging_config import get_logger
    logger = get_logger(__name__)

Importing this module (directly, or transitively via `get_logger`) registers
the custom logger class before any logger is created, so ordering relative to
other imports doesn't matter.
"""

import logging
from typing import cast

PROGRESS = 25
logging.addLevelName(PROGRESS, 'PROGRESS')


class ProgressLogger(logging.Logger):
    def progress(self, message, *args, **kwargs):
        if self.isEnabledFor(PROGRESS):
            self._log(PROGRESS, message, args, **kwargs)


logging.setLoggerClass(ProgressLogger)


def get_logger(name: str) -> ProgressLogger:
    return cast(ProgressLogger, logging.getLogger(name))
