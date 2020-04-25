import numpy as np
import sys
import functools as f
import paths as p
import itertools as i
from typing import Tuple, Generator, List, Union, Dict, Optional
import dataclasses as dc
import collections as c
import statistics as stat

import sklearn.base as skb
import sklearn.model_selection as skms
import sklearn.linear_model as sklm
import sklearn.naive_bayes as sknb
import sklearn.neural_network as sknn
import sklearn.svm as skvm
import sklearn.preprocessing as skp
import sklearn.metrics as skm
import sklearn.feature_selection as skfs

ModelParam = Dict[str, List]
ModelParams = Union[ModelParam, List[ModelParam]]

INNER_CV_K = 5
RANDOM_SEED = 69


@dc.dataclass(repr=True)
class Scores:
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def __init__(self, y_true: np.array, y_pred: np.array) -> None:
        self.balanced_accuracy = skm.balanced_accuracy_score(y_true, y_pred)
        self.f1 = skm.f1_score(y_true, y_pred)
        self.precision = skm.precision_score(y_true, y_pred)
        self.recall = skm.recall_score(y_true, y_pred)
        self.roc_auc = skm.roc_auc_score(y_true, y_pred)


@dc.dataclass()
class TrainSessionInfo:
    selected_features: np.array
    hyperparams: Optional[Dict]
    best_inner_cv_score: Optional[float]  # mean scores for inner cv
    scores: Scores  # scores on outer test set

    def __repr__(self) -> str:
        return f'selected features: {self.selected_features}\n' \
               f'hyperparameters: {self.hyperparams}\n' \
               f'scores: {repr(self.scores)}\n'


def stratified_split(features: np.array, labels: np.array) -> List[Tuple[np.array, np.array]]:
    return skms.StratifiedKFold(n_splits=10, shuffle=True).split(X=features, y=labels)


def get_feature_selectors() -> List[skfs.SelectPercentile]:
    return [skfs.SelectPercentile(score_func=method, percentile=percent) for percent in (1, 5, 10, 20, 100)
            for method in (skfs.f_classif, skfs.mutual_info_classif)]


def normalize(train_features: np.array, test_features: np.array) -> Tuple[np.array, np.array]:
    normer = skp.StandardScaler(with_mean=True, with_std=True)
    normed_train = normer.fit_transform(X=train_features)
    return normed_train, normer.transform(X=test_features)


def train(features: np.array, labels: np.array, model: skb.ClassifierMixin, params: ModelParams,
          selector: skfs.SelectPercentile) -> List[TrainSessionInfo]:
    infos = []
    for train_set, test_set in stratified_split(features, labels):
        cv_model = skb.clone(model)
        train_features = features[train_set]
        train_labels = labels[train_set]
        test_features = features[test_set]
        test_labels = labels[test_set]

        # fit and transform on train data, only transform test data
        pruned_train_features = selector.fit_transform(X=train_features, y=train_labels)
        pruned_test_features = selector.transform(X=test_features)
        selected_features = selector.get_support(indices=True)

        # normalize by train features
        pruned_train_features, pruned_test_features = normalize(pruned_train_features, pruned_test_features)

        if params is None:
            cv_model.fit(X=pruned_train_features, y=train_labels)
            pred_labels = cv_model.predict(X=pruned_test_features)
            infos.append(TrainSessionInfo(selected_features,
                                          None,
                                          None,
                                          Scores(test_labels, pred_labels)))

        # get best hyperparameters, predict on test data
        grid = skms.GridSearchCV(estimator=cv_model, param_grid=params, scoring='f1', cv=INNER_CV_K, refit=True)
        grid.fit(X=pruned_train_features, y=train_labels)  # TODO does this train over whole training set
        pred_labels = grid.predict(X=pruned_test_features)

        infos.append(TrainSessionInfo(selected_features,
                                      grid.best_params_,
                                      grid.best_score_,
                                      Scores(test_labels, pred_labels)))

    return infos


def print_average_scores(scores: List[Scores]) -> None:
    print(f'score averages:')
    print(f'average balanced accuracy: {stat.mean([score.balanced_accuracy for score in scores])}')
    print(f'average precision: {stat.mean([score.precision for score in scores])}')
    print(f'average recall: {stat.mean([score.recall for score in scores])}')
    print(f'average f1 score: {stat.mean([score.f1 for score in scores])}')
    print(f'average roc auc score: {stat.mean([score.roc_auc for score in scores])}')


def get_feature_counts(feature_groups: List[np.array]) -> Dict[int, int]:
    return f.reduce(lambda x, y: x + c.Counter(y), feature_groups, c.Counter())


def get_feature_stability_score(feature_counts: Dict[int, int], iterations: int, num_of_features: int) -> float:
    return (sum(feature_counts.values()) - len(feature_counts.keys())) / ((iterations - 1) * num_of_features)


def print_training_infos(infos: List[TrainSessionInfo]) -> None:
    for idx, info in enumerate(infos):
        print(f'info: {idx}')
        print(info)

    print_average_scores([info.scores for info in infos])
    feature_counts = get_feature_counts([info.selected_features for info in infos])
    print(f'feature counts: {feature_counts}')
    stability_score = get_feature_stability_score(feature_counts, len(infos), len(infos[0].selected_features))
    print(f'feature stability score: {stability_score}')


def get_log_reg_items(labels: np.array) -> Tuple[skb.ClassifierMixin, ModelParams]:
    model = sklm.LogisticRegression()
    weight = lambda x: sum(labels == x) / len(labels)
    params = {'class_weight': [{0: weight(1), 1: weight(0)}],
              'C': [.01, .1, 1, 10],
              'max_iter': [10000],
              'solver': ['liblinear']}
    return model, params


def get_svc_items() -> Tuple[skb.ClassifierMixin, ModelParams]:
    model = skvm.SVC()
    params = {'kernel': ['rbf'],
              'C': [.01, .1, 1, 10],
              'gamma': [.001, .01, .1, 1, 10],
              'class_weight': ['balanced'],
              'max_iter': [10000000]}
    return model, params


def get_nn_hidden_layer_sizes(input_feature_count: int, num_of_samples: int) -> List[Tuple[int]]:
    sizes = [round(num_of_samples / (alpha * (input_feature_count + 1))) for alpha in (2,)]
    return list([item for item in i.chain(*[((size,), (size, size,)) for size in sizes])])


def get_neural_net_items() -> Tuple[skb.ClassifierMixin, ModelParams]:
    model = sknn.MLPClassifier()
    # params = {'activation': ['logistic', 'relu', 'tanh'],
    #           'batch_size': [10, 50, 100],
    #           'max_iter': [2500],
    #           'hidden_layer_sizes': []}
    params = {'activation': ['relu'],
              'batch_size': [100],
              'max_iter': [2500],
              'hidden_layer_sizes': []}
    return model, params


def run_classifier(model_type: str) -> None:
    features = np.load(p.clean_features_path)
    labels = np.load(p.np_labels_path)

    if model_type == 'log_reg':
        model, params = get_log_reg_items(labels)
    elif model_type == 'nb':

        model = sknb.GaussianNB()
        params = {'priors': [(sum(labels == 0) / len(labels), sum(labels == 1) / len(labels)), None]}
    elif model_type == 'svc':
        model, params = get_svc_items()
    elif model_type == 'nn':
        model, params = get_neural_net_items()
    else:
        raise ValueError(f'unsupported classifier type {model_type}')

    for selector in get_feature_selectors():

        if model_type == 'nn':
            params['hidden_layer_sizes'] = get_nn_hidden_layer_sizes(selector.percentile * len(features[0]) / 100,
                                                                     len(features))

        training_infos = train(features, labels, model, params, selector)
        print(f'training results for model: {model_type}, '
              f'selector function: {selector.score_func.__name__}, '
              f'percentile: {selector.percentile}')
        print_training_infos(training_infos)
        print('\n')


if __name__ == '__main__':
    if len(sys.argv) > 2:
        raise ValueError(f'usage: {sys.argv[0]} model_type')
    run_classifier(sys.argv[1])
