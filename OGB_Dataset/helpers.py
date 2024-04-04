import os
import shutil


def createFolder(folder_path: str):
    try:
        os.makedirs(folder_path)
    except Exception:
        pass


def copyFile(folder_path: str, my_file: str):
    file_name = os.path.basename(my_file)

    destination = os.path.join(folder_path, file_name)

    try:
        shutil.copy2(my_file, destination)
    except FileNotFoundError:
        print("Could not find file")
    except Exception:
        pass

    return destination