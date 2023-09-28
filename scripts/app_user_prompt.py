from .app_utils import display_directories


def prompt():

    directories = display_directories()
    selected_directory_list = [directories[i] for i in range(len(directories))]

    return selected_directory_list
