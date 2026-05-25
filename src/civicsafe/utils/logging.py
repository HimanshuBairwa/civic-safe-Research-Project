"""Structured logging with colored console output and JSON-lines file output."""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# ANSI color codes for console handler
# ---------------------------------------------------------------------------

_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: "\033[36m",  # cyan
    logging.INFO: "\033[32m",  # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[35m",  # magenta
}
_RESET: str = "\033[0m"


# ---------------------------------------------------------------------------
# Custom formatters
# ---------------------------------------------------------------------------


class _ColoredConsoleFormatter(logging.Formatter):
    """Formatter that prepends an ANSI color code based on log level."""

    def format(self, record: logging.LogRecord) -> str:
        color: str = _LEVEL_COLORS.get(record.levelno, "")
        timestamp: str = datetime.fromtimestamp(record.created, tz=UTC).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        return (
            f"{color}{timestamp} [{record.levelname:<8}] "
            f"{record.name}: {record.getMessage()}{_RESET}"
        )


class _JsonLinesFormatter(logging.Formatter):
    """Formatter that emits one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, str] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(log_entry, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_logger(
    name: str,
    log_dir: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a logger with colored console and optional JSON-lines file output.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        log_dir: Directory for the JSON-lines log file. If ``None``, only the
            console handler is attached. The directory is created if absent.
        level: Logging level. Default: ``logging.INFO``.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    console_handler: logging.StreamHandler = logging.StreamHandler(sys.stderr)  # type: ignore[type-arg]
    console_handler.setFormatter(_ColoredConsoleFormatter())
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path: Path = log_dir / f"{name}.jsonl"
        file_handler: logging.FileHandler = logging.FileHandler(
            log_path, encoding="utf-8"
        )
        file_handler.setFormatter(_JsonLinesFormatter())
        logger.addHandler(file_handler)

    return logger


def log_metrics(
    logger: logging.Logger,
    metrics: dict[str, float],
    step: int,
    prefix: str = "",
) -> None:
    """Log a dictionary of metrics at INFO level with consistent formatting.

    Each metric is logged as ``[prefix/]key=value`` on a single line.

    Args:
        logger: Target logger instance.
        metrics: Mapping of metric names to scalar values.
        step: Training step or epoch number for context.
        prefix: Optional prefix prepended to each metric name.
    """
    tag: str = f"{prefix}/" if prefix else ""
    parts: list[str] = [f"{tag}{key}={value:.6g}" for key, value in metrics.items()]
    logger.info("step=%d  %s", step, "  ".join(parts))
