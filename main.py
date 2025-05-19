from tqdm import tqdm
import pandas as pd
import numpy as np
import time
from mesa import Mesa
from arguments import parser
from utils import Rater, load_dataset
from sklearn.tree import DecisionTreeClassifier
# from data_util import *
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.model_selection import StratifiedKFold


def get_statlog_vehicle_silhouettes4():
    dataset = fetch_ucirepo(id=149)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("statlog_vehicle_silhouettes4 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 'van' else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("statlog_vehicle_silhouettes4 处理后类别分布:", Counter(y))
    return X, y, "statlog_vehicle_silhouettes4"

if __name__ == '__main__':
    
    # load dataset & prepare environment
    args = parser.parse_args()
    rater = Rater(args.metric)
    # X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(args.dataset)

    X, y, dataset_name = get_statlog_vehicle_silhouettes4()
    base_estimator = DecisionTreeClassifier(max_depth=None)

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    runs = 5  # 5 folds
    scores_list, time_list = [], []

    print('\nStart 5-Fold Stratified Cross-Validation of MESA ... ...\n')

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f'Running Fold {fold_idx + 1}/5 ...')
        X_train_val, X_test = X[train_index], X[test_index]
        y_train_val, y_test = y[train_index], y[test_index]

        # Split training into train and valid
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
        )

        mesa = Mesa(
            args=args,
            base_estimator=base_estimator,
            n_estimators=args.max_estimators
        )

        # meta training
        mesa.meta_fit(X_train, y_train, X_valid, y_valid, X_test, y_test)

        # ensemble test
        start_time = time.perf_counter()
        mesa.fit(X_train, y_train, X_valid, y_valid, verbose=False)
        end_time = time.perf_counter()
        time_list.append(end_time - start_time)

        score_train = rater.score(y_train, mesa.predict_proba(X_train)[:, 1])
        score_valid = rater.score(y_valid, mesa.predict_proba(X_valid)[:, 1])
        score_test = rater.score(y_test, mesa.predict_proba(X_test)[:, 1])
        scores_list.append([score_train, score_valid, score_test])

    # print results
    df_scores = pd.DataFrame(scores_list, columns=['train', 'valid', 'test'])
    info = f'Dataset: {dataset_name}\nMESA {args.metric} |'
    for column in df_scores.columns:
        info += ' {} {:.3f}-{:.3f} |'.format(column, df_scores.mean()[column], df_scores.std()[column])
    info += ' 5-Fold CV (mean-std) |'
    info += ' ave run time: {:.2f}s'.format(np.mean(time_list))
    print(info)


    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
    #
    # base_estimator = DecisionTreeClassifier(max_depth=None)
    #
    # # meta-training
    # print ('\nStart meta-training of MESA ... ...\n')
    # mesa = Mesa(
    #     args=args,
    #     base_estimator=base_estimator,
    #     n_estimators=args.max_estimators)
    # mesa.meta_fit(X_train, y_train, X_valid, y_valid, X_test, y_test)
    #
    # # test
    # print ('\nStart ensemble training of MESA ... ...\n')
    # runs = 100
    # scores_list, time_list = [], []
    # for i_run in tqdm(range(runs)):
    #     start_time = time.clock()
    #     mesa.fit(X_train, y_train, X_valid, y_valid, verbose=False)
    #     end_time = time.clock()
    #     time_list.append(end_time - start_time)
    #     score_train = rater.score(y_train, mesa.predict_proba(X_train)[:,1])
    #     score_valid = rater.score(y_valid, mesa.predict_proba(X_valid)[:,1])
    #     score_test = rater.score(y_test, mesa.predict_proba(X_test)[:,1])
    #     scores_list.append([score_train, score_valid, score_test])
    #
    # # print results to stdout
    # df_scores = pd.DataFrame(scores_list, columns=['train', 'valid', 'test'])
    # info = f'Dataset: {args.dataset}\nMESA {args.metric}|'
    # for column in df_scores.columns:
    #     info += ' {} {:.3f}-{:.3f} |'.format(column, df_scores.mean()[column], df_scores.std()[column])
    # info += ' {} runs (mean-std) |'.format(runs)
    # info += ' ave run time: {:.2f}s'.format(np.mean(time_list))
    # print (info)