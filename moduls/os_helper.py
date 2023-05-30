import os
import shutil
from moduls.Log import PRINT_ULTRASTAR


def create_folder(folder_name):
    print(f"{PRINT_ULTRASTAR} Creating output folder. -> {folder_name}")
    is_exist = os.path.exists(folder_name)
    if not is_exist:
        os.makedirs(folder_name)


def move(original, target):
    shutil.move(original, target)


def copy(original, target):
    shutil.copy(original, target)


def current_executor_path():
    return os.getcwd()


def path_join(path1, path2):
    return os.path.join(path1, path2)


def check_file_exists(file_path):
    return os.path.isfile(file_path)
