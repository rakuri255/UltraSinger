"""Docstring"""

import os
import shutil

from modules.console_colors import ULTRASINGER_HEAD


def create_folder(folder_name):
    """Docstring"""
    print(f"{ULTRASINGER_HEAD} Creating output folder. -> {folder_name}")
    is_exist = os.path.exists(folder_name)
    if not is_exist:
        os.makedirs(folder_name)


def move(original, target):
    """Docstring"""
    shutil.move(original, target)


def copy(original, target):
    """Docstring"""
    shutil.copy(original, target)


def current_executor_path():
    """Docstring"""
    return os.getcwd()


def path_join(path1, path2):
    """Docstring"""
    return os.path.join(path1, path2)


def check_file_exists(file_path):
    """Docstring"""
    return os.path.isfile(file_path)


def check_if_folder_exists(song_output):
    """Docstring"""
    return os.path.isdir(song_output)
