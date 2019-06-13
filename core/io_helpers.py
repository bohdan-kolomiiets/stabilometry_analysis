import os


def get_file_pathes_in_dir(path: str, extension: str):
    file_names = []
    for file_name in os.listdir(path):
        if file_name.endswith(f".{extension}"):
            file_names.append(f"{path}/{file_name}")
    return file_names


def extract_name_from_path(path):
    import re
    path_parts = re.split(r'[\\|\/]', path)
    file_name_with_extension = path_parts[-1]
    return file_name_with_extension.split('.')[0]

def create_folder_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
