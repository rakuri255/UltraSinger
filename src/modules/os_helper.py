"""OS helper functions."""

import os
import shutil

from modules.console_colors import ULTRASINGER_HEAD


def create_folder(folder_name: str) -> None:
    """Creates a folder if it doesn't exist."""
    print(f"{ULTRASINGER_HEAD} Creating output folder. -> {folder_name}")
    is_exist = os.path.exists(folder_name)
    if not is_exist:
        os.makedirs(folder_name)


def move(original: str, target: str) -> None:
    """Moves a file from one location to another."""
    shutil.move(original, target)


def copy(original: str, target: str) -> None:
    """Copies a file from one location to another."""
    shutil.copy(original, target)


def rename(original: str, target: str) -> None:
    """Renames a file."""
    os.rename(original, target)


def current_executor_path() -> str:
    """Current executor path"""
    return os.getcwd()


def path_join(path1: str, path2: str) -> str:
    """Joins two paths together"""
    return os.path.join(path1, path2)


def check_file_exists(file_path: str) -> bool:
    """Checks if a file exists."""
    return os.path.isfile(file_path)


def check_if_folder_exists(song_output: str) -> bool:
    """Checks if a folder exists."""
    return os.path.isdir(song_output)
