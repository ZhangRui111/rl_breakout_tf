import numpy as np
import errno
import os
import csv
import shutil


def binary_array_to_int(array, size):
    out_list = []
    for i in range(size):
        input = array[i]
        out = np.argmax(input)
        out_list.append(out)
    out_arr = np.array(out_list)
    return out_arr


def simple_binary_array_to_int(array):
    input = array
    out = 0
    for bit in input:
        out = (out << 1) | bit
    return out


def my_print(content, signal):
    print(signal*10, content, signal*10)


def write_file(path, content, overwrite=False):
    """ write data to file.
    :param path:
    :param content:
    :param overwrite: open file by 'w' (True) or 'a' (False)
    :return:
    """
    # Check the file path.
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    # Write data.
    if overwrite is True:
        with open(path, 'w') as fo:
            fo.write(content)
            fo.close()
    else:
        with open(path, 'a') as fo:
            fo.write(content + '\n')
            fo.close()


def read_file(path):
    # Check the file path.
    if os.path.exists(os.path.dirname(path)):
        with open(path, 'r') as fo:
            data = fo.read()
            fo.close()
    else:
        data = 'NONE'
    return data


def write_ndarray(path, data):
    """ write ndarray to csv file.
    """
    # Write data.
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
            np.savetxt(path, data)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    else:
        old_array = np.loadtxt(path)
        new_array = np.concatenate((old_array, data))
        np.savetxt(path, new_array)


def read_ndarray(path):
    """ read ndarray from file.
    """
    return np.loadtxt(path)


def copy_rename_folder(oldpath, newpath, new_name):
    """ copy a folder and rename it.
    :param oldpath: string
    :param newpath: string
    :param new_name: string
    :return:
    """
    # Check the old path.
    if os.path.exists(os.path.dirname(oldpath)):
        try:
            shutil.copytree(oldpath, newpath)
            os.rename(newpath, new_name)
        except OSError as exc:  # Guard against race condition
            print('Warning: old path is not valid!')
            if exc.errno != errno.EEXIST:
                raise