import os
import shutil


def create_folder(folder_name):
    print("Creating output folder. -> " + folder_name)
    is_exist = os.path.exists(folder_name)
    if not is_exist:
        os.makedirs(folder_name)


def move(original, target):
    shutil.move(original, target)


def current_executor_path():
    return os.getcwd()
