import numpy as np
from sklearn import ensemble
from layer import Layer
from logger import get_logger
from k_fold_wrapper import KFoldWrapper

import pickle

LOGGER = get_logger("gcForest")


class gcForest(object):

    def __init__(self, config):
        self.random_state = config["random_state"]
        self.max_layers = config["max_layers"]
        self.early_stop_rounds = config["early_stop_rounds"]
        self.if_stacking = config["if_stacking"]
        self.if_save_model = config["if_save_model"]

        # self.train_evaluation是一个函数
        self.train_evaluation = config["train_evaluation"]
        self.estimator_configs = config["estimator_configs"]
        self.layers = []
        self.cascade_pred_proba = []
        self.cascade_pred_proba_sum = []

    def fit(self, x_train, y_train, x_valid, y_valid):

        x_train, n_feature, n_label = self.preprocess(x_train, y_train)

        evaluate = self.train_evaluation
        best_layer_id = 0
        depth = 0
        best_layer_evaluation = 0.0

        # 用于记录所有层的共同预测结果
        self.cascade_pred_proba_sum = np.zeros((x_train.shape[0], n_label))
        self.cascade_pred_proba = np.zeros((x_train.shape[0], n_label))

        # max_layers应该是表示深度森林的最大深度
        while depth < self.max_layers:

            # 记录了当前层的所有森林对所有训练样本的预测类概率向量
            y_train_probas = np.zeros((x_train.shape[0], n_label * len(self.estimator_configs)))

            current_layer = Layer(depth)
            LOGGER.info(
                "-----------------------------------------layer-{}--------------------------------------------".format(
                    current_layer.layer_id))
            LOGGER.info("The shape of x_train is {}".format(x_train.shape))

            # 记录了当前层的所有森林对所有训练样本的预测类概率向量的平均值
            y_train_probas_avg = np.zeros((x_train.shape[0], n_label))

            # 在这一层中生成若干个森林
            for index in range(len(self.estimator_configs)):

                # 复制当前森林的配置
                config = self.estimator_configs[index].copy()
                k_fold_est = KFoldWrapper(current_layer.layer_id, index, config, random_state=self.random_state)

                y_proba = k_fold_est.fit(x_train, y_train)

                # 将第index个五折交叉森林加入到当前层中
                current_layer.add_est(k_fold_est)
                y_train_probas[:, index * n_label:index * n_label + n_label] += y_proba
                y_train_probas_avg += y_proba

            y_train_probas_avg /= len(self.estimator_configs)
            self.cascade_pred_proba_sum += y_train_probas_avg
            self.cascade_pred_proba = self.cascade_pred_proba_sum / len(self.estimator_configs)
            label_tmp = self.category[np.argmax(y_train_probas_avg, axis=1)]
            current_evaluation = evaluate(y_train, label_tmp)

            # 如果堆叠的话，将所有层的4个森林的类概率向量都拼接在一起，否则只拼接当前层的4个森林的类概率向量
            if self.if_stacking:
                x_train = np.hstack((x_train, y_train_probas))
            else:
                x_train = np.hstack((x_train[:, 0:n_feature], y_train_probas))

            if current_evaluation > best_layer_evaluation:
                best_layer_id = current_layer.layer_id
                best_layer_evaluation = current_evaluation
            LOGGER.info(
                "The evaluation[{}] of layer_{} is {:.4f}".format(evaluate.__name__, depth, current_evaluation))

            self.layers.append(current_layer)

            if current_layer.layer_id - best_layer_id >= self.early_stop_rounds:
                self.layers = self.layers[0:best_layer_id + 1]
                LOGGER.info("training finish...")
                LOGGER.info(
                    "best_layer: {}, current_layer:{}, save layers: {}".format(best_layer_id, current_layer.layer_id,
                                                                               len(self.layers)))
                break

            depth += 1

        # if self.if_save_model:
        #     pickle.dump(self,open("gc.pkl","wb"))

    def predict(self, x):
        prob = self.predict_proba(x)
        label = self.category[np.argmax(prob, axis=1)]
        return label

    def predict_proba(self, x):
        x_test = x.copy()
        x_test = x_test.reshape((x.shape[0], -1))
        n_feature = x_test.shape[1]
        # print(x_test.shape)
        x_test_proba = None
        for index in range(len(self.layers)):

            # 前几层的森林返回堆叠后的一层中的4个类概率向量，最后一层的森林返回的是类概率向量
            if index == len(self.layers) - 1:
                # print(index)
                x_test_proba = self.layers[index]._predict_proba(x_test)
            else:
                x_test_proba = self.layers[index].predict_proba(x_test)
                if not self.if_stacking:
                    x_test = x_test[:, 0:n_feature]
                x_test = np.hstack((x_test, x_test_proba))
        return x_test_proba

    # 这个代码返回的是训练样本、特征数、标签数
    def preprocess(self, x_train, y_train):
        x_train = x_train.reshape((x_train.shape[0], -1))
        category = np.unique(y_train)
        self.category = category
        # print(len(self.category))
        n_feature = x_train.shape[1]
        n_label = len(np.unique(y_train))
        LOGGER.info("Begin to train....")
        LOGGER.info("the shape of training samples: {}".format(x_train.shape))
        LOGGER.info("use {} as training evaluation".format(self.train_evaluation.__name__))
        LOGGER.info("stacking: {}, save model: {}".format(self.if_stacking, self.if_save_model))
        return x_train, n_feature, n_label
