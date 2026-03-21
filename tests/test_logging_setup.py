# tests/test_logging_setup.py
import logging
import os
import pytest


def test_setup_logging_attaches_three_handlers(tmp_path):
    """Root logger must have exactly 3 handlers: console, info file, warn file."""
    import sys

    if "main" in sys.modules:
        del sys.modules["main"]

    root = logging.getLogger()
    original_handlers = root.handlers[:]
    root.handlers.clear()

    try:
        from main import setup_logging
        setup_logging(log_dir=str(tmp_path))

        handlers = root.handlers
        assert len(handlers) == 3, f"Expected 3 handlers, got {len(handlers)}"
    finally:
        root.handlers = original_handlers


def test_info_handler_level(tmp_path):
    """FileHandler for info.log must be set to INFO (level 20)."""
    import sys
    if "main" in sys.modules:
        del sys.modules["main"]

    root = logging.getLogger()
    original_handlers = root.handlers[:]
    root.handlers.clear()

    try:
        from main import setup_logging
        setup_logging(log_dir=str(tmp_path))

        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        info_handler = next(
            (h for h in file_handlers if "info.log" in h.baseFilename), None
        )
        assert info_handler is not None, "info.log handler not found"
        assert info_handler.level == logging.INFO
    finally:
        root.handlers = original_handlers


def test_warn_handler_level(tmp_path):
    """FileHandler for warn.log must be set to WARNING (level 30)."""
    import sys
    if "main" in sys.modules:
        del sys.modules["main"]

    root = logging.getLogger()
    original_handlers = root.handlers[:]
    root.handlers.clear()

    try:
        from main import setup_logging
        setup_logging(log_dir=str(tmp_path))

        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        warn_handler = next(
            (h for h in file_handlers if "warn.log" in h.baseFilename), None
        )
        assert warn_handler is not None, "warn.log handler not found"
        assert warn_handler.level == logging.WARNING
    finally:
        root.handlers = original_handlers


def test_log_files_are_created(tmp_path):
    """Both log files must exist after setup_logging() is called."""
    import sys
    if "main" in sys.modules:
        del sys.modules["main"]

    root = logging.getLogger()
    original_handlers = root.handlers[:]
    root.handlers.clear()

    try:
        from main import setup_logging
        setup_logging(log_dir=str(tmp_path))

        # Emit a message to ensure file creation is flushed
        logging.getLogger("test").info("ping")

        assert (tmp_path / "info.log").exists(), "info.log not created"
        assert (tmp_path / "warn.log").exists(), "warn.log not created"
    finally:
        # Close file handlers before cleanup
        for h in root.handlers:
            if isinstance(h, logging.FileHandler):
                h.close()
        root.handlers = original_handlers
