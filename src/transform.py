import numpy as np
import paths as p
import pathlib as pl
import datetime as dt

"""Loads and saves the raw feature and label files into numpy arrays"""


def parse_timestamp(timestamp: str) -> float:
    """Parses the given timestamp string into a Unix timestamp
    :param timestamp: string timestamp of listed format
    :return: Unix timestamp of given string timestamp"""
    return dt.datetime.strptime(timestamp, '"%d/%m/%Y %H:%M:%S"\n').timestamp()


def load_labels(path: pl.Path) -> None:
    """Loads and saves the given path to a text file of labels and timestamps into a two numpy arrays of labels and
    timestamps
    :param path: path to the text file of labels and timestamps"""
    labels = []
    timestamps = []
    with open(path, 'r') as f:
        for line in f.readlines():
            label, timestamp = line.split(' ', 1)
            labels.append(0 if int(label) == -1 else 1)
            timestamps.append(parse_timestamp(timestamp))

    ts_array = np.array(timestamps)
    np.save(file=p.np_label_ts_path, arr=ts_array)

    labels_array = np.array(labels)
    np.save(file=p.np_labels_path, arr=labels_array)


def load_features(path: pl.Path) -> None:
    """Loads and saves the text file of features at the given path into a numpy array
    :param path: path to the text file of features """
    features = np.loadtxt(fname=path, dtype=float, delimiter=' ')
    np.save(file=p.np_features_path, arr=features)


if __name__ == '__main__':
    load_features(p.raw_features_path)
    load_labels(p.raw_labels_path)
