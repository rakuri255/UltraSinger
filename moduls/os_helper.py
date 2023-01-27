import os


def create_folder(folder_name):
    is_exist = os.path.exists(folder_name)
    if not is_exist:
        os.makedirs(folder_name)
