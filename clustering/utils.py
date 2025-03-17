# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging
import os
import numpy as np
import random


def seed_everything(seed: int = 42):
    """
    Function to set seed for random number generators for reproducibility.

    Args:
        seed: The seed value to use for random number generators. Default is 42.

    Returns:
        None
        该函数用于设置随机数生成器的种子值，从而确保代码在多次运行时产生的随机数序列是一致的，也就是实现可重复性。
    """
    # Set seed values for various random number generators
    # 为各种随机数生成器设置种子值
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)#确定随机性的种子，以确保在多次运行时产生的随机数序列是一致的

    # Ensure deterministic behavior for CUDA algorithms
    # 为CUDA算法确保确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def get_logger(
#     file_name="logger.log", level=logging.INFO, stdout=False
# ) -> logging.Logger:
#     """
#     When the level is set to "logging.INFO", the debugging logs will not be saved (lower level).
#     """
#     # See https://www.loggly.com/ultimate-guide/python-logging-basics/ for more information about pyhton logging module
#     logger = logging.getLogger()  # uses the module name
#     # set log level
#     logger.setLevel(level)
#     logger.handlers = []
#     # define file handler and set formatter
#     file_handler = logging.FileHandler(
#         file_name
#     )  # or use logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", file_name))
#     # define formatter
#     formatter = logging.Formatter(
#         "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
#     )  # or use logging.BASIC_FORMAT
#     file_handler.setFormatter(formatter)

#     stdout_handler = (
#         logging.StreamHandler()
#     )  # .setLevel(logging.DEBUG) #.setFormatter(CustomFormatter(fmt))

#     # add handler to logger
#     # if not logger.hasHandlers():
#     logger.addHandler(file_handler)
#     if stdout:
#         logger.addHandler(stdout_handler)

#     return logger


#     import logging


def get_logger(
    file_name: str = "logger.log", level: int = logging.INFO, stdout: bool = False
) -> logging.Logger:
    """
    Initialize and configure the logger object to save log entries to a file and optionally print to stdout.

    :param file_name: The name of the log file.
    :param level: The logging level to use (default: INFO).
    :param stdout: Whether to enable printing log entries to stdout (default: False).
    :return: A configured logging.Logger instance.
    初始化并配置Logger对象以将日志条目保存到文件中,并可选地打印到STDOUT。

    :param file_name:日志文件的名称。
    :param级别:要使用的记录级别(默认:info)。
    :param stdout:是否将打印日志条目启用到stdout(默认:false)。
    :返回:配置的logging.logger实例。
    """
    logger = logging.getLogger(__name__)

    # Set the logging level
    # 设置日志级别
    logger.setLevel(level)

    # Remove any existing handlers from the logger
    # 从记录器中删除任何现有处理程序
    logger.handlers = []

    # Create a file handler for the logger
    # 为记录器创建一个文件处理程序
    file_handler = logging.FileHandler(file_name)

    # Define the formatter for the log entries
    # 定义日志条目的格式化程序
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )

    # Set the formatter for the file handler
    # 为文件处理
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    # 将文件处理程序添加到记录器
    logger.addHandler(file_handler)

    # Optionally add a stdout handler to the logger
    # 可选地将stdout处理程序添加到记录器
    if stdout:
        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    # Return the configured logger instance
    # 返回配置的记录器实例
    return logger
