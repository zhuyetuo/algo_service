"""
日志配置：loguru 双输出（stdout + 滚动文件）。
同时拦截 uvicorn / sqlalchemy / apscheduler 的标准 logging，统一路由到 loguru。
"""
import logging
import sys

from loguru import logger

from config import settings


class _InterceptHandler(logging.Handler):
    """将标准 logging 调用转发给 loguru。"""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging() -> None:
    """应用启动时调用一次，配置 loguru sinks。"""
    logger.remove()

    # stdout：彩色输出，级别由配置决定
    logger.add(
        sys.stdout,
        level=settings.log_level.upper(),
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # 文件：按大小滚动，保留最近 10 个文件，压缩归档
    logger.add(
        "logs/algo_service.log",
        level="DEBUG",
        rotation="50 MB",
        retention=10,
        compression="zip",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} - {message}",
    )

    # 拦截标准 logging（uvicorn、sqlalchemy、apscheduler 等）
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).handlers = [_InterceptHandler()]
        logging.getLogger(name).propagate = False

    # 屏蔽第三方库的 DEBUG 噪音
    for noisy in ("urllib3", "urllib3.connectionpool", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
