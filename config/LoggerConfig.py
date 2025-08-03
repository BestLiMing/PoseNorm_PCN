import os
import sys
import logging
import colorlog
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Union
import datetime
import inspect

'''
single GPU
from logger_config import setup_logger
logger = setup_logger(name="MyProject", log_dir="./project_logs")
logger.info("项目启动")

multi GPU
def train(rank):
    logger = setup_logger(
        name="DDP_Training",
        log_dir="./ddp_logs",
        ddp_rank=rank,
        console_level=logging.INFO if rank == 0 else logging.WARNING
    )
    logger.debug(f"Rank {rank} 初始化完成")
    
Automatically get caller
from logger_config import get_caller_logger
logger = get_caller_logger()
def func():
    logger.debug("调试信息")
'''


def setup_logger(
        name: str = "MAIN",
        log_dir: str = "./logs",
        console_level: Union[int, str] = logging.INFO,
        file_level: Union[int, str] = logging.DEBUG,
        when: str = 'midnight',
        backup_count: int = 7,
        max_bytes: int = 100 * 1024 * 1024,  # 100MB
        ddp_rank: Optional[int] = None,
        formatter_str: str = '%(asctime)s - %(levelname)s - PID:%(process)d - %(name)s - %(filename)s:%(lineno)d - %(message)s'
) -> logging.Logger:
    """
        Args:
            name: 日志名称（通常为模块名或项目名）
            log_dir: 日志目录路径
            console_level: 控制台日志级别
            file_level: 文件日志级别
            when: 时间轮转间隔 ('S', 'M', 'H', 'D', 'midnight'等)
            backup_count: 保留的历史日志文件数量
            max_bytes: 单个日志文件最大字节数（用于大小轮转）
            ddp_rank: 分布式训练的进程rank（自动隔离不同进程的日志）
            formatter_str: 日志格式字符串
    """
    if ddp_rank is not None:
        log_dir = os.path.join(log_dir, f"rank_{ddp_rank}")
        name = f"{name}_RANK{ddp_rank}"
    elif 'RANK' in os.environ:  # 自动检测DDP环境变量
        ddp_rank = int(os.environ.get('RANK', 0))
        return setup_logger(name, log_dir, console_level, file_level, when, backup_count, max_bytes, ddp_rank,
                            formatter_str)

    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, file_level))
    if logger.handlers:
        return logger
    formatter = colorlog.ColoredFormatter(
        formatter_str,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    log_filename = os.path.join(log_dir, f"{name}_{datetime.date.today()}.log")
    if when:  # 时间轮转
        file_handler = TimedRotatingFileHandler(
            filename=log_filename,
            when=when,
            backupCount=backup_count,
            encoding='utf-8'
        )
    else:  # 大小轮转
        file_handler = RotatingFileHandler(
            filename=log_filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    return logger


def get_caller_logger() -> logging.Logger:
    """自动获取调用者模块的Logger"""
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    return logging.getLogger(module.__name__ if module else "__main__")


