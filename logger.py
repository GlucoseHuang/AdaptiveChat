import logging
import sys


# --- 日志配置开始 ---
class ColoredFormatter(logging.Formatter):
    """自定义日志格式，添加颜色高亮"""
    # 颜色代码
    grey = "\x1b[38;20m"
    cyan = "\x1b[36;20m"  # DEBUG
    green = "\x1b[32;20m"  # INFO
    yellow = "\x1b[33;20m"  # WARNING
    red = "\x1b[31;20m"  # ERROR
    reset = "\x1b[0m"

    # 日志格式: [时间] [级别] [文件名:行号] 信息
    fmt = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"
    datefmt = "%H:%M:%S"  # 只需要时分秒，年月日对于实时调试太长了

    FORMATS = {
        logging.DEBUG: cyan + fmt + reset,
        logging.INFO: green + fmt + reset,
        logging.WARNING: yellow + fmt + reset,
        logging.ERROR: red + fmt + reset,
        logging.CRITICAL: red + fmt + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, self.datefmt)
        return formatter.format(record)


def setup_logger():
    # 创建 logger
    logger = logging.getLogger("Thesis_System")
    logger.setLevel(logging.INFO)  # 如果想看超详细信息，改为 logging.DEBUG

    # 避免重复添加 handler (Streamlit 可能会多次运行脚本)
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

    return logger


logger = setup_logger()
# --- 日志配置结束 ---