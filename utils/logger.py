"""日志工具模块。"""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(
    name: str = "vision_gui_agent",
    log_level: int = logging.INFO,
    log_file_path: Path | None = None,
) -> logging.Logger:
    """
    创建并返回标准 logger（同时输出到控制台和日志文件）。

    Args:
        name: logger 名称。
        log_level: 日志级别，默认 INFO。
        log_file_path: 日志文件路径。默认使用 data/logs/vision_gui_agent.log。

    Returns:
        logging.Logger: 配置完成的日志对象。
    """
    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 避免在多次初始化时重复添加 handler。
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    if log_file_path is None:
        logs_dir = Path("data/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = logs_dir / "vision_gui_agent.log"
    else:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
