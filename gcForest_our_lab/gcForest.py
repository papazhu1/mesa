import numpy as np
from sklearn import ensemble
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from evaluation import f1_macro
from layer import Layer
from logger import get_logger
from k_fold_wrapper import KFoldWrapper
from arguments import parser
import pickle
from gym import spaces
from sac_src.sac import SAC
from sac_src.replay_memory import ReplayMemory
from myEnvironment import MyEnsembleTrainingEnv
from utils import *
LOGGER = get_logger("gcForest")


class gcForest(object):

    def __init__(self, config):

        self.args = parser.parse_args()
        # super(gcForest, self).__init__(self.args)
        self.random_state = config["random_state"]
        self.max_layers = config["max_layers"]
        self.early_stop_rounds = config["early_stop_rounds"]
        self.if_stacking = config["if_stacking"]
        self.if_save_model = config["if_save_model"]
        self.base_estimator_ = config["base_estimator_"]
        # self.train_evaluation是一个函数
        self.train_evaluation = config["train_evaluation"]
        self.estimator_configs = config["estimator_configs"]
        self.layers = []
        self.cascade_train_pred_proba = None
        self.cascade_train_pred_proba_sum = None
        self.cascade_valid_pred_proba = None
        self.cascade_valid_pred_proba_sum = None
        self.cascade_test_pred_proba = None
        self.cascade_test_pred_proba_sum = None

        state_size = int(self.args.num_bins * 2)
        action_space = spaces.Box(low=0.0, high=1.0, shape=[1], dtype=np.float32)
        self.meta_sampler = SAC(state_size, action_space, self.args)
        self.env = MyEnsembleTrainingEnv(self.args, DecisionTreeClassifier())
        self.rater = Rater(metric=self.args.metric)
        self.memory = ReplayMemory(self.args.replay_size)



    def score(self, X, y):
        """Return area under precision recall curve (AUCPRC) scores for X, y.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input data instances.

        y : array-like of shape = [n_samples]
            Labels for X.

        Yields
        ----------
        z : float
        """
        yield sklearn.metrics.average_precision_score(
            y, self.predict_proba(X)[:, 1])

    def load_data(self, X_train, y_train, X_valid, y_valid, X_test=None, y_test=None, train_ratio=1):
        """Load and preprocess the train/valid/test data into the environment."""
        self.flag_use_test_set = False if X_test is None or y_test is None else True
        if train_ratio < 1:
            print ('Using {:.2%} random subset for meta-training.'.format(train_ratio))
            _, X_train, _, y_train = imbalance_train_test_split(X_train, y_train, test_size=train_ratio)
        self.X_train, self.y_train = pd.DataFrame(X_train), pd.Series(y_train)
        self.X_valid, self.y_valid = pd.DataFrame(X_valid), pd.Series(y_valid)
        self.X_test,  self.y_test  = pd.DataFrame(X_test),  pd.Series(y_test)
        self.mask_maj_train, self.mask_min_train = (y_train==0), (y_train==1)
        self.mask_maj_valid, self.mask_min_valid = (y_valid==0), (y_valid==1)
        self.n_min_samples = self.mask_min_train.sum()
        n_samples = int(self.n_min_samples*self.args.train_ir)
        if n_samples > self.mask_maj_train.sum():
            raise ValueError(f"\
                Argument 'train_ir' should be smaller than imbalance ratio,\n \
                Please set this parameter to < {self.mask_maj_train.sum()/self.mask_min_train.sum()}.\
                ")
        self.n_samples = n_samples

    def fit_step(self, X, y):
        """Bulid a new base classifier from the training set (X, y).

        Parameters
        ----------
        y : array-like of shape = [n_samples]
            The training labels.

        X : array-like of shape = [n_samples, n_features]
            The training instances.

        Returns
        ----------
        self : object (Ensemble)
        """
        self.estimators_.append(
            sklearn.base.clone(self.base_estimator_).fit(X, y)
            )
        return self

    # 对训练集、验证集和测试集的预测结果进行评估
    def record_scores(self):
        """Record the training/validation/test performance scores."""
        train_score = self.env.rater.score(self.env.y_train, self.env.y_pred_train_buffer)
        valid_score = self.env.rater.score(self.env.y_valid, self.env.y_pred_valid_buffer)
        test_score = self.env.rater.score(self.env.y_test,
                                          self.env.y_pred_test_buffer) if self.env.flag_use_test_set else 'NULL'

        acc_train = self.env.rater.score(self.env.y_train, self.env.y_pred_train_buffer, method="acc")
        acc_valid = self.env.rater.score(self.env.y_valid, self.env.y_pred_valid_buffer, method="acc")
        acc_test = self.env.rater.score(self.env.y_test, self.env.y_pred_test_buffer,
                                        method="acc") if self.env.flag_use_test_set else 'NULL'

        print('acc_train: {:.3f} | acc_valid: {:.3f} | acc_test: {:.3f}'.format(acc_train, acc_valid, acc_test))

        sen_train = self.env.rater.score(self.env.y_train, self.env.y_pred_train_buffer, method="sen")
        sen_valid = self.env.rater.score(self.env.y_valid, self.env.y_pred_valid_buffer, method="sen")
        sen_test = self.env.rater.score(self.env.y_test, self.env.y_pred_test_buffer,
                                        method="sen") if self.env.flag_use_test_set else 'NULL'

        print('sen_train: {:.3f} | sen_valid: {:.3f} | sen_test: {:.3f}'.format(sen_train, sen_valid, sen_test))

        spe_train = self.env.rater.score(self.env.y_train, self.env.y_pred_train_buffer, method="spe")
        spe_valid = self.env.rater.score(self.env.y_valid, self.env.y_pred_valid_buffer, method="spe")
        spe_test = self.env.rater.score(self.env.y_test, self.env.y_pred_test_buffer,
                                        method="spe") if self.env.flag_use_test_set else 'NULL'

        print('spe_train: {:.3f} | spe_valid: {:.3f} | spe_test: {:.3f}'.format(spe_train, spe_valid, spe_test))

        gmean_train = self.env.rater.score(self.env.y_train, self.env.y_pred_train_buffer, method="gmean")
        gmean_valid = self.env.rater.score(self.env.y_valid, self.env.y_pred_valid_buffer, method="gmean")
        gmean_test = self.env.rater.score(self.env.y_test, self.env.y_pred_test_buffer,
                                          method="gmean") if self.env.flag_use_test_set else 'NULL'

        print('gmean_train: {:.3f} | gmean_valid: {:.3f} | gmean_test: {:.3f}'.format(gmean_train, gmean_valid,
                                                                                      gmean_test))

        f1macro_train = self.env.rater.score(self.env.y_train, self.env.y_pred_train_buffer, method="f1macro")
        f1macro_valid = self.env.rater.score(self.env.y_valid, self.env.y_pred_valid_buffer, method="f1macro")
        f1macro_test = self.env.rater.score(self.env.y_test, self.env.y_pred_test_buffer,
                                            method="f1macro") if self.env.flag_use_test_set else 'NULL'

        print('f1macro_train: {:.3f} | f1macro_valid: {:.3f} | f1macro_test: {:.3f}'.format(f1macro_train,
                                                                                            f1macro_valid,
                                                                                            f1macro_test))

        self.scores.append(
            [train_score, valid_score, test_score] if self.env.flag_use_test_set else [train_score, valid_score])
        return

    # 元训练初始化
    def meta_fit_init(self):
        # buffer the predict probabilities for better efficiency
        #
        self.estimators_ = []
        self.y_pred_train_buffer = self.cascade_train_pred_proba
        self.y_pred_valid_buffer = self.cascade_valid_pred_proba
        if self.flag_use_test_set:
            self.y_pred_test_buffer = self.cascade_test_pred_proba
        self._warm_up()

    # 这个函数只在第一次的时候用，所以是随机采样
    def _warm_up(self):
        """Train the first base classifier with random under-sampling."""
        X_maj = self.X_train[self.mask_maj_train]
        X_min = self.X_train[self.mask_min_train]
        X_maj_rus = X_maj.sample(n=self.n_samples, random_state=self.args.random_state)
        # X_maj_rus = X_maj
        X_train_rus = pd.concat([X_maj_rus, X_min]).values
        y_train_rus = np.concatenate([np.zeros(X_maj_rus.shape[0]), np.ones(X_min.shape[0])])
        self.fit_step(X_train_rus, y_train_rus)
        self.update_all_pred_buffer()
        return

    def update_all_pred_buffer(self):
        """Update all buffered predict probabilities."""
        n_clf = len(self.estimators_)
        print("n_clf: ", n_clf)
        print("len(self.layers) * self.estimator_configs[0][\"n_estimators\"]: ", len(self.layers) * self.estimator_configs[0]["n_estimators"])
        sum_num_ests = n_clf + len(self.layers) * self.estimator_configs[0]["n_estimators"]

        self.y_pred_train_buffer = self._update_pred_buffer(sum_num_ests, self.X_train, self.y_pred_train_buffer)
        self.y_pred_valid_buffer = self._update_pred_buffer(sum_num_ests, self.X_valid, self.y_pred_valid_buffer)
        if self.flag_use_test_set:
            self.y_pred_test_buffer = self._update_pred_buffer(sum_num_ests, self.X_test,  self.y_pred_test_buffer)
        return

    # 用最新训练的基分类器预测，然后更新平均预测值
    def _update_pred_buffer(self, sum_num_ests, X, y_pred_buffer):
        """Update buffered predict probabilities.

        Parameters
        ----------
        sum_num_ests : int
            Current ensemble size.

        X : array-like of shape = [n_samples, n_features]
            The input data instances.

        y_pred_buffer : array-like of shape [n_samples]
            The buffered predict probabilities of X.

        Returns
        ----------
        y_pred_updated : array-like of shape [n_samples]
        """
        y_pred_last_clf = self.estimators_[-1].predict_proba(X)[:, 1]
        y_pred_buffer_updated = (y_pred_buffer[:, 1] * (sum_num_ests - 1) + y_pred_last_clf) / sum_num_ests
        return y_pred_buffer_updated

    # 获取当前的环境状态，基于训练和验证数据的错误分布直方图来表示状态。
    def get_state(self):
        """Fetch the current state of the environment."""
        hist_train = histogram_error_distribution(
            self.y_train[self.mask_maj_train],
            self.y_pred_train_buffer[self.mask_maj_train],
            self.args.num_bins)
        hist_valid = histogram_error_distribution(
            self.y_valid[self.mask_maj_valid],
            self.y_pred_valid_buffer[self.mask_maj_valid],
            self.args.num_bins)
        hist_train = hist_train / hist_train.sum() * self.args.num_bins
        hist_valid = hist_valid / hist_valid.sum() * self.args.num_bins
        state = np.concatenate([hist_train, hist_valid])
        return state

    # action是一个浮点数，表示采样比例mu
    # step是用来采样的, 然后训练一个基分类器
    def step(self, action, verbose=False):
        """Perform an environment step.

        Parameters
        ----------
        action: float, in [0, 1]
            The action (mu) to execute in the environment.

        verbose: bool, optional (default=False)
            Whether to compute and return the information about the current ensemble.

        Returns
        ----------
        next_state : array-like of shape [state_size]
            The state of the environment after executing the action.

        reward : float
            The reward of taking the action.

        done : bool
            Indicates the end of an episode.
            True if the ensemble reaches the maximum number of base estimators.

        info : string
            Information about the current ensemble.
            Empty string if verbose == False.
        """
        # check action value
        if action < 0 or action > 1:
            raise ValueError("Action must be a float in [0, 1].")

        # perform meta-sampling
        X_maj_subset, X_idx = meta_sampling(
            y_pred=self.y_pred_train_buffer[self.mask_maj_train],
            y_true=self.y_train[self.mask_maj_train],
            n_under_samples=self.n_samples,
            X=self.X_train[self.mask_maj_train],
            mu=action,
            sigma=self.args.sigma,
            random_state=self.args.random_state, )

        # build training subset (X_train_iter, y_train_iter)
        X_train_iter = pd.concat([X_maj_subset, self.X_train[self.mask_min_train]]).values
        y_train_iter = np.concatenate([np.zeros(X_maj_subset.shape[0]), np.ones(self.n_min_samples)])

        score_valid_before = self.rater.score(self.y_valid, self.y_pred_valid_buffer)

        # build a new base classifier from (X_train_iter, y_train_iter)
        self.fit_step(X_train_iter, y_train_iter)
        self.update_all_pred_buffer()

        score_valid = self.rater.score(self.y_valid, self.y_pred_valid_buffer)

        # obtain return values
        next_state = self.get_state()
        reward = score_valid - score_valid_before
        done = True if len(self.estimators_) >= self.args.max_estimators else False
        info = ''

        # fetch environment information if verbose==True
        if self.args.meta_verbose is 'full' or verbose:
            score_train = self.rater.score(self.y_train, self.y_pred_train_buffer)
            score_test = self.rater.score(self.y_test,
                                          self.y_pred_test_buffer) if self.flag_use_test_set else 'NULL'
            info = 'k={:<3d}|{}| train {:.3f} | valid {:.3f} | '.format(
                len(self.estimators_) - 1, self.args.metric, score_train, score_valid)
            info += 'test {:.3f}'.format(score_test) if self.flag_use_test_set else 'test NULL'

        return next_state, reward, done, info


    # 每次元训练在下一层开始时训练，state为当前深度森林性能，next_state为加上当前的分类器的深度森林性能
    def meta_fit(self, X_train, y_train, X_valid, y_valid, X_test=None, y_test=None):
        """Meta-training process of MESA.

        Parameters
        ----------
        X_train : array-like of shape = [n_training_samples, n_features]
            The training data instances.

        y_train : array-like of shape = [n_training_samples]
            Labels for X_train.

        X_valid : array-like of shape = [n_validation_samples, n_features]
            The validation data instances.

        y_valid : array-like of shape = [n_validation_samples]
            Labels for X_valid.

        X_test : array-like of shape = [n_training_samples, n_features], optional (default=None)
            The test data instances.

        y_train : array-like of shape = [n_training_samples], optional (default=None)
            Labels for X_test.

        Returns
        ----------
        self : object (Mesa)
        """
        # initialize replay memory and environment

        print("meta_fit begin---------------------------------")
        self.env.load_data(X_train, y_train, X_valid, y_valid, X_test, y_test, train_ratio=self.args.train_ratio)
        self.memory = memory_init_fulfill(self.args, ReplayMemory(self.args.replay_size))

        self.scores = []

        # total_steps：总的训练步数，由 update_steps 和 start_steps 决定。
        # start_steps 是模型开始元学习之前的随机探索阶段，update_steps 是元学习阶段。
        total_steps = self.args.update_steps + self.args.start_steps
        num_steps, num_updates, num_episodes = 0, 0, 0

        # start meta-training

        print("total_steps:", total_steps)
        print("start_steps:", self.args.start_steps)
        print("update_steps:", self.args.update_steps)

        while num_steps < total_steps:
            print("num_steps:", num_steps)
            self.env.init()
            state = self.env.get_state()
            done = False

            # for each episode
            while not done:
                num_steps += 1

                # take an action
                # 在 start_steps 之前，采取随机动作；之后则通过元采样器（meta_sampler）选择最优动作。
                if num_steps >= self.args.start_steps:
                    action, by = self.meta_sampler.select_action(state), 'mesa'
                else:
                    action, by = self.meta_sampler.action_space.sample(), 'rand'

                # store transition
                # print("action:")
                # print(action)
                next_state, reward, done, info = self.env.step(action[0])
                reward = reward * self.args.reward_coefficient
                self.memory.push(state, action, reward, next_state, float(done))

                # update meta-sampler parameters
                if num_steps > self.args.start_steps:
                    for i in range(self.args.updates_per_step):
                        _, _, _, _, _ = self.meta_sampler.update_parameters(
                            self.memory, self.args.batch_size, num_updates)
                        num_updates += self.args.updates_per_step

                # print log to stdout
                if self.args.meta_verbose is 'full':
                    print('Epi.{:<4d} updates{:<4d}| {} | {} by {}'.format(num_episodes, num_updates, info, action[0],
                                                                           by))

                if done:
                    num_episodes += 1
                    self.record_scores()
                    # record print mean score of latest args.meta_verbose_mean_episodes to stdout
                    self.verbose_mean_scores(num_episodes, num_updates, by)

        return self


    def fit(self, X_train, y_train, X_valid, y_valid, X_test, y_test, gc):

        X_train, n_feature, n_label = self.preprocess(X_train, y_train)
        self.n_label = n_label

        self.meta_fit(X_train, y_train, X_valid, y_valid, X_test, y_test)

        evaluate = self.train_evaluation
        best_layer_id = 0
        depth = 0
        best_layer_evaluation = 0.0

        self.cascade_train_pred_proba = np.zeros((X_train.shape[0], n_label))
        self.cascade_valid_pred_proba = np.zeros((X_valid.shape[0], n_label))
        self.cascade_test_pred_proba = np.zeros((X_test.shape[0], n_label))

        self.cascade_train_pred_proba_sum = np.zeros((X_train.shape[0], n_label))
        self.cascade_valid_pred_proba_sum = np.zeros((X_valid.shape[0], n_label))
        self.cascade_test_pred_proba_sum = np.zeros((X_test.shape[0], n_label))



        # max_layers应该是表示深度森林的最大深度
        while depth < self.max_layers:
            print(depth)
            self.load_data(X_train, y_train, X_valid, y_valid, X_test, y_test)
            self.meta_fit_init()
            self.actions_record = []

            # 记录了当前层的所有森林对所有训练样本的预测类概率向量
            y_train_probas = np.zeros((X_train.shape[0], n_label * len(self.estimator_configs)))
            y_valid_probas = np.zeros((X_valid.shape[0], n_label * len(self.estimator_configs)))
            y_test_probas = np.zeros((X_test.shape[0], n_label * len(self.estimator_configs)))


            current_layer = Layer(depth)
            LOGGER.info(
                "-----------------------------------------layer-{}--------------------------------------------".format(
                    current_layer.layer_id))
            LOGGER.info("The shape of X_train is {}".format(X_train.shape))

            # 记录了当前层的所有森林对所有训练样本的预测类概率向量的平均值
            y_train_probas_avg = np.zeros((X_train.shape[0], n_label))
            y_valid_probas_avg = np.zeros((X_valid.shape[0], n_label))
            y_test_probas_avg = np.zeros((X_test.shape[0], n_label))

            # 在这一层中生成若干个森林
            for index in range(len(self.estimator_configs)):

                # 复制当前森林的配置
                config = self.estimator_configs[index].copy()
                k_fold_est = KFoldWrapper(current_layer.layer_id, index, config, random_state=self.random_state)

                y_proba = k_fold_est.fit(X_train, y_train, X_valid, y_valid, X_test, y_test, self)
                y_proba_valid = k_fold_est.predict_proba(X_valid)
                y_proba_test = k_fold_est.predict_proba(X_test)

                # 将第index个五折交叉森林加入到当前层中
                current_layer.add_est(k_fold_est)
                y_train_probas[:, index * n_label:index * n_label + n_label] += y_proba
                y_train_probas_avg += y_proba

                y_valid_probas[:, index * n_label:index * n_label + n_label] += y_proba_valid
                y_valid_probas_avg += y_proba_valid

                y_test_probas[:, index * n_label:index * n_label + n_label] += y_proba_test
                y_test_probas_avg += y_proba_test

            y_train_probas_avg /= len(self.estimator_configs)
            self.cascade_train_pred_proba_sum += y_train_probas_avg
            self.cascade_train_pred_proba = self.cascade_train_pred_proba_sum / len(self.estimator_configs)

            y_valid_probas_avg /= len(self.estimator_configs)
            self.cascade_valid_pred_proba_sum += y_valid_probas_avg
            self.cascade_valid_pred_proba = self.cascade_valid_pred_proba_sum / len(self.estimator_configs)

            y_test_probas_avg /= len(self.estimator_configs)
            self.cascade_test_pred_proba_sum += y_test_probas_avg
            self.cascade_test_pred_proba = self.cascade_test_pred_proba_sum / len(self.estimator_configs)

            label_tmp = self.category[np.argmax(y_train_probas_avg, axis=1)]
            current_evaluation = evaluate(y_train, label_tmp)

            # 如果堆叠的话，将所有层的4个森林的类概率向量都拼接在一起，否则只拼接当前层的4个森林的类概率向量
            if self.if_stacking:
                X_train = np.hstack((X_train, y_train_probas))
                X_valid = np.hstack((X_valid, y_valid_probas))
                X_test = np.hstack((X_test, y_test_probas))
            else:
                X_train = np.hstack((X_train[:, 0:n_feature], y_train_probas))
                X_valid = np.hstack((X_valid[:, 0:n_feature], y_valid_probas))
                X_test = np.hstack((X_test[:, 0:n_feature], y_test_probas))

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
    def preprocess(self, X_train, y_train):
        X_train = X_train.reshape((X_train.shape[0], -1))
        category = np.unique(y_train)
        self.category = category
        # print(len(self.category))
        n_feature = X_train.shape[1]
        n_label = len(np.unique(y_train))
        LOGGER.info("Begin to train....")
        LOGGER.info("the shape of training samples: {}".format(X_train.shape))
        LOGGER.info("use {} as training evaluation".format(self.train_evaluation))
        LOGGER.info("stacking: {}, save model: {}".format(self.if_stacking, self.if_save_model))
        return X_train, n_feature, n_label

    def verbose_mean_scores(self, num_episodes, num_updates, by):
        """Print mean score of latest n episodes to stdout.

        n = args.meta_verbose_mean_episodes

        Parameters
        ----------
        num_episodes : int
            The number of finished meta-training episodes.

        num_updates : int
            The number of finished meta-sampler updates.

        by : {'rand', 'mesa'}, string
            The way of selecting actions in the current episode.
        """
        if self.args.meta_verbose is 'full' or (
                self.args.meta_verbose != 0 and num_episodes % self.args.meta_verbose == 0):
            view_bound = max(-self.args.meta_verbose_mean_episodes, -len(self.scores))
            recent_scores_mean = np.array(self.scores)[view_bound:].mean(axis=0)
            print(
                'Epi.{:<4d} updates {:<4d} |last-{}-mean-{}| train {:.3f} | valid {:.3f} | test {:.3f} | by {}'.format(
                    num_episodes, num_updates, self.args.meta_verbose_mean_episodes, self.args.metric,
                    recent_scores_mean[0], recent_scores_mean[1], recent_scores_mean[2], by))
        return