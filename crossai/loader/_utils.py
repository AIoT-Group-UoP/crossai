import os
import platform


def get_sub_dirs(path):
    """Gets the subdirectories of a directory and returns the subdirectories
    and the names of the subdirectories.

    Args:
        path (str): path to the directory

    Returns:
        list: subdirectories
        list: names of the subdirectories
    """

    subdirs = [x[0] for x in os.walk(path)]
    subdirnames = []

    # get the name of the subdirectories
    # in windows the path is split with \ instead of /
    if platform.system() == 'Windows':
        subdirs = [x.split('\\')[-1] for x in subdirs]
    else:
        subdirs = [x.split('/')[-1] for x in subdirs]
    if len(subdirs) > 1:
        subdirs.pop(0)
        # copy subdirs
        subdirnames = subdirs.copy()
    else:
        if platform.system() == 'Windows':
            subdirnames = [path.split('\\')[-2]]
        else:
            subdirnames = [path.split('/')[-2]]

    return subdirs, subdirnames
