# coding: utf-8
import logging
from logging import handlers


class Logger(object):
    level_relations = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "crit": logging.CRITICAL,
    }  # 日志级别关系映射

    def __init__(
        self,
        filename,
        level="info",
        when="D",
        maxBytes=1024 * 1024 * 10,  # 10M
        backCount=3,
        # fmt="%(asctime)s - %(levelname)s: %(message)s",
        fmt="%(asctime)s - Id: %(message)s",
        DATE_FORMAT="%Y-%m-%d %H:%M:%S",
    ):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt, datefmt=DATE_FORMAT)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        # 往文件里写入#指定间隔时间自动生成文件的处理器
        # th = handlers.TimedRotatingFileHandler(
        #     filename=filename, when=when, backupCount=backCount, encoding="utf-8"
        # )
        # 按文件大小切割
        th = handlers.RotatingFileHandler(filename, maxBytes=maxBytes, backupCount=backCount, encoding="utf-8")

        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)


if __name__ == "__main__":
    log = Logger("t.log")
    import time

    for _ in range(10):
        log.logger.info("xxxx")
        time.sleep(3)
    # log.logger.warning('警告')
    # log.logger.error('报错')
    # log.logger.critical('严重')
    # Logger("error.log", level="error").logger.error("error")
