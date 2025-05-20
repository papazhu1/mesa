from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from BaseForest import BaseForest
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from logger import get_logger
from imblearn.under_sampling import RandomUnderSampler
'''
此处导入新的包
'''

LOGGER_2 = get_logger("KFoldWrapper")


class KFoldWrapper(object):
    def __init__(self, layer_id, index, config, random_state):
        self.config = config
        self.name = "layer_{}, estimator_{}, {}".format(layer_id, index, self.config["type"])
        if random_state is not None:
            self.random_state = (random_state + hash(self.name)) % 1000000007
        else:
            self.random_state = None
        # print(self.random_state)
        self.n_fold = self.config["n_fold"]

        # estimators用于存储每折交叉验证的分类器
        self.estimators = [None for i in range(self.config["n_fold"])]
        self.config.pop("n_fold")
        self.estimator_class = globals()[self.config["type"]]
        self.config.pop("type")

    def _init_estimator(self):

        estimator_args = self.config
        est_args = estimator_args.copy()
        # est_args["random_state"]=self.random_state
        return self.estimator_class(**est_args)

    def fit(self, x, y, X_valid, y_valid, X_test, y_test, gc):

        skf = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=self.random_state)
        cv = [(t, v) for (t, v) in skf.split(x, y)]

        n_label = len(np.unique(y))

        # 这个y_probas存储了所有训练样本在每一折上当验证集时的预测类概率向量
        y_probas = np.zeros((x.shape[0], n_label))

        for k in range(self.n_fold):
            est = self._init_estimator()
            train_id, val_id = cv[k]
            # print(x[train_id])
            x_train = x[train_id]
            y_train = y[train_id]

            # x_train, y_train = RandomUnderSampler().fit_resample(x_train, y_train)  # 固定随机种子以保证结果可重复
            est.fit(x_train, y_train, X_valid, y_valid, X_test, y_test, train_id, gc)
            y_proba = est.predict_proba(x[val_id])
            y_pred = est.predict(x[val_id])
            LOGGER_2.info(
                "{}, n_fold_{},Accuracy={:.4f}, f1_score={:.4f}".format(self.name, k, accuracy_score(y[val_id], y_pred),
                                                                        f1_score(y[val_id], y_pred, average="macro")))
            y_probas[val_id] += y_proba
            self.estimators[k] = est
        LOGGER_2.info("{}, {},Accuracy={:.4f}, f1_score={:.4f}".format(self.name, "wrapper",
                                                                       accuracy_score(y, np.argmax(y_probas, axis=1)),
                                                                       f1_score(y, np.argmax(y_probas, axis=1),
                                                                                average="macro")))
        LOGGER_2.info("----------")
        return y_probas

    def predict_proba(self, x_test):
        proba = None
        for est in self.estimators:
            if proba is None:
                proba = est.predict_proba(x_test)
            else:
                proba += est.predict_proba(x_test)
        proba /= self.n_fold
        # print(proba)
        return proba
