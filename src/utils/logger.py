"""Structured logging for the UC4 aim assist pipeline."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str, config: Optional[dict] = None) -> logging.Logger:
    """Return a configured logger. Safe to call multiple times for the same name."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured in this process

    cfg = config or {}
    level_str = cfg.get("level", "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file handler
    if cfg.get("log_to_file", False):
        log_dir = Path(cfg.get("log_dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_dir / "uc4_aim_assist.log",
            maxBytes=cfg.get("max_log_size_mb", 50) * 1024 * 1024,
            backupCount=cfg.get("backup_count", 3),
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
