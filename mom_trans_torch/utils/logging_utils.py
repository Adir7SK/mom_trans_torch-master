import logging
import sqlite3
from datetime import datetime
from colorama import init, Fore, Style
import pandas as pd


def get_logger(name: str = None):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(name)-1s| %(levelname)-1s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


def print_bright(s):
    init()
    print(Style.BRIGHT + s + Style.RESET_ALL)


def print_green(info, value=""):
    print(Fore.GREEN + "[%s] " % info + Style.RESET_ALL + str(value))


def print_red(info, value=""):
    print(Fore.RED + "[%s] " % info + Style.RESET_ALL + str(value))


def str_to_bluestr(string):
    return Fore.BLUE + "%s" % string + Style.RESET_ALL


def str_to_yellowstr(string):
    return Fore.YELLOW + "%s" % string + Style.RESET_ALL