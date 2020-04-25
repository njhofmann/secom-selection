import sklearn.impute as ski
import paths as p
import numpy as np


def fill_in_nans(features: np.array) -> np.array:
    """Fills in the NaN values in the given numpy matrix of features, on a row by row basis
    :param features: numpy matrix of features
    :return: given features with NaN values filled in"""
    imputer = ski.SimpleImputer(missing_values=np.nan, strategy='mean')
    return imputer.fit_transform(features)


def remove_constant_columns(features: np.array) -> np.array:
    """Removes columns of constant value from given numpy feature matrix
    :param features: numpy matrix of features
    :return: given feature matrix with constant column values returned"""
    constant_indicies = [idx for idx, constant in enumerate(np.std(features, axis=0) == 0) if constant]
    return np.delete(features, constant_indicies, axis=1)


if __name__ == '__main__':
    features = np.load(p.np_features_path)
    labels = np.load(p.np_labels_path)

    features = fill_in_nans(features)
    features = remove_constant_columns(features)
    np.save(file=p.clean_features_path, arr=features)
