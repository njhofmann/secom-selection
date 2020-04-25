import pathlib as pl

data_dir = pl.Path(__file__).parent.joinpath('data')
raw_features_path = data_dir.joinpath('features')
raw_labels_path = data_dir.joinpath('labels')
np_features_path = data_dir.joinpath('features.npy')
np_labels_path = data_dir.joinpath('labels.npy')
np_label_ts_path = data_dir.joinpath('timestamps.npy')
clean_features_path = data_dir.joinpath('clean_features.npy')
