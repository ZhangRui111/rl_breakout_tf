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


def write_csv_file(path, data, overwrite=False):
    """ write data to csv file.
    :param path:
    :param data: list
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
        with open(path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(data)
    else:
        with open(path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(data)


def read_csv_file_to_int(path):
    """ read data from csv file.
    :param path:
    :return: One element is one row.
    """
    list_rows = []
    with open(path, encoding='utf-8') as fo:
        csv_reader = csv.reader(fo)
        for row in csv_reader:
            list_rows.append(row)
    rows = len(list_rows)
    all_data = list_rows[0]
    for i in range(rows):
        all_data += list_rows[i]
    for j in range((len(all_data))):
        all_data[j] = int(all_data[j])
    return all_data


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