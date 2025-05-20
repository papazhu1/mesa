from sklearn.ensemble import RandomForestClassifier
import gcForest
import pandas as pd
import numpy as np
from utils import *
import sklearn


class BaseForest(RandomForestClassifier):
    def __init__(self,
                 estimator=None,
                 *,
                 n_estimators=100,
                 estimator_params=tuple(),
                 n_jobs=None,
                 random_state=None,
                 max_depth=None,
                 verbose=0,
                 ):
        super().__init__(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
            max_depth=max_depth,
            verbose=verbose,
        )

        self.name = "BaseForest"
        # self.sampler_ = BalancingSampler()
        self.y_pred_proba = None
        self.y_pred_proba_list = []

    def load_data(self, X_train, y_train, X_valid, y_valid, X_test, y_test, gc, train_ratio=1):
        """Load and preprocess the train/valid/test data into the environment."""
        self.flag_use_test_set = False if X_test is None or y_test is None else True
        if train_ratio < 1:
            print('Using {:.2%} random subset for meta-training.'.format(train_ratio))
            _, X_train, _, y_train = imbalance_train_test_split(X_train, y_train, test_size=train_ratio)
        self.X_train, self.y_train = pd.DataFrame(X_train), pd.Series(y_train)
        self.X_valid, self.y_valid = pd.DataFrame(X_valid), pd.Series(y_valid)
        self.X_test, self.y_test = pd.DataFrame(X_test), pd.Series(y_test)
        self.all_X_train, self.all_y_train = gc.X_train, gc.y_train
        self.mask_maj_train, self.mask_min_train = (y_train == 0), (y_train == 1)
        self.mask_maj_valid, self.mask_min_valid = (y_valid == 0), (y_valid == 1)

        self.mask_maj_all_train, self.mask_min_all_train = (gc.y_train == 0), (gc.y_train == 1)


        self.n_min_samples = self.mask_min_train.sum()
        n_samples = int(self.n_min_samples * gc.args.train_ir)
        if n_samples > self.mask_maj_train.sum():
            raise ValueError(f"\
                Argument 'train_ir' should be smaller than imbalance ratio,\n \
                Please set this parameter to < {self.mask_maj_train.sum() / self.mask_min_train.sum()}.\
                ")
        self.n_samples = n_samples

    # 元训练初始化
    def fit_init(self, gc):
        # buffer the predict probabilities for better efficiency
        # initialize
        self.y_pred_train_buffer = gc.cascade_train_pred_proba[:, 1]
        self.y_pred_valid_buffer = gc.cascade_valid_pred_proba[:, 1]
        if gc.flag_use_test_set:
            self.y_pred_test_buffer = gc.cascade_test_pred_proba[:, 1]
        self._warm_up(gc)

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

    # 这个函数只在第一次的时候用，所以是随机采样
    def _warm_up(self, gc):
        """Train the first base classifier with random under-sampling."""
        X_maj = self.X_train[self.mask_maj_train]
        X_min = self.X_train[self.mask_min_train]
        X_maj_rus = X_maj.sample(n=self.n_samples, random_state=gc.args.random_state)
        # X_maj_rus = X_maj
        X_train_rus = pd.concat([X_maj_rus, X_min]).values
        y_train_rus = np.concatenate([np.zeros(X_maj_rus.shape[0]), np.ones(X_min.shape[0])])
        self.fit_step(X_train_rus, y_train_rus)
        self.update_all_pred_buffer(gc)
        return

    def update_all_pred_buffer(self, gc):
        """Update all buffered predict probabilities."""
        n_clf = len(self.estimators_)
        # print("gc.estimator_configs[0][\"n_estimators\"]: ", gc.estimator_configs[0]["n_estimators"])
        sum_num_ests = n_clf + len(gc.layers) * gc.estimator_configs[0]["n_estimators"]

        self.y_pred_train_buffer = self._update_pred_buffer(sum_num_ests, self.all_X_train, self.y_pred_train_buffer)
        self.y_pred_valid_buffer = self._update_pred_buffer(sum_num_ests, gc.X_valid, self.y_pred_valid_buffer)
        if gc.flag_use_test_set:
            self.y_pred_test_buffer = self._update_pred_buffer(sum_num_ests, gc.X_test, self.y_pred_test_buffer)
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
        # y_pred_last_clf = self.estimators_[-1].predict_proba(X)[:, 1]
        # y_pred_buffer_updated = (y_pred_buffer[:, 1] * (sum_num_ests - 1) + y_pred_last_clf) / sum_num_ests

        # 这里传入的 X 就是 all_train
        y_pred_last_clf = self.estimators_[-1].predict_proba(X)[:, 1]
        y_pred_buffer_updated = (y_pred_buffer * (sum_num_ests - 1) + y_pred_last_clf) / sum_num_ests

        return y_pred_buffer_updated

    # 获取当前的环境状态，基于训练和验证数据的错误分布直方图来表示状态。
    def get_state(self, gc):
        # print(self.mask_maj_all_train)
        # print(self.mask_maj_train)
        """Fetch the current state of the environment."""
        hist_train = histogram_error_distribution(
            self.all_y_train[self.mask_maj_all_train],
            self.y_pred_train_buffer[self.mask_maj_all_train],
            gc.args.num_bins)
        hist_valid = histogram_error_distribution(
            self.y_valid[self.mask_maj_valid],
            self.y_pred_valid_buffer[self.mask_maj_valid],
            gc.args.num_bins)
        hist_train = hist_train / hist_train.sum() * gc.args.num_bins
        hist_valid = hist_valid / hist_valid.sum() * gc.args.num_bins
        state = np.concatenate([hist_train, hist_valid])
        return state

    # action是一个浮点数，表示采样比例mu
    # step是用来采样的, 然后训练一个基分类器
    def step(self, action, gc, verbose=False):
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
            y_pred=self.y_pred_train_buffer[self.mask_maj_all_train],
            y_true=self.all_y_train[self.mask_maj_all_train],
            n_under_samples=self.n_samples,
            X=self.all_X_train[self.mask_maj_all_train],
            mu=action,
            sigma=gc.args.sigma,
            random_state=gc.args.random_state, )

        X_idx_in_train = np.intersect1d(X_idx, self.train_idx, assume_unique=True)
        X_maj_subset = self.all_X_train.loc[X_idx_in_train]  # if DataFrame


        # build training subset (X_train_iter, y_train_iter)
        X_train_iter = pd.concat([X_maj_subset, self.X_train[self.mask_min_train]]).values
        y_train_iter = np.concatenate([np.zeros(X_maj_subset.shape[0]), np.ones(self.n_min_samples)])

        score_valid_before = gc.rater.score(self.y_valid, self.y_pred_valid_buffer)

        # build a new base classifier from (X_train_iter, y_train_iter)
        self.fit_step(X_train_iter, y_train_iter)
        self.update_all_pred_buffer(gc)

        score_valid = gc.rater.score(self.y_valid, self.y_pred_valid_buffer)

        # obtain return values
        next_state = self.get_state(gc)
        reward = score_valid - score_valid_before
        done = True if len(self.estimators_) >= gc.args.max_estimators else False
        info = ''

        # fetch environment information if verbose==True
        if gc.args.meta_verbose is 'full' or verbose:
            score_train = gc.rater.score(self.y_train, self.y_pred_train_buffer)
            score_test = gc.rater.score(self.y_test,
                                              self.y_pred_test_buffer) if gc.flag_use_test_set else 'NULL'
            info = 'k={:<3d}|{}| train {:.3f} | valid {:.3f} | '.format(
                len(self.estimators_) - 1, gc.args.metric, score_train, score_valid)
            info += 'test {:.3f}'.format(score_test) if gc.flag_use_test_set else 'test NULL'

        return next_state, reward, done, info

    def fit(self, X_train, y_train, X_valid, y_valid, X_test, y_test, train_idx, gc, verbose=0, **kwargs):
        super().fit(X_train, y_train, **kwargs)
        n_estimators = self.n_estimators
        self.train_idx = train_idx
        # print("len:X_train: ", len(X_train))
        # print("len:X_valid:", len(X_valid))
        self.load_data(X_train, y_train, X_valid, y_valid, X_test, y_test, gc)
        self.fit_init(gc)
        self.actions_record = []
        for i in range(n_estimators - 1):
            # 对于集成学习，每个state就是当前集成器的性能，作者将它们归结成直方图
            state = self.get_state(gc)
            action = gc.meta_sampler.select_action(state)
            self.actions_record.append(action[0])

            # 在step中完成了集成器的训练和性能评估
            _, _, _, info = self.step(action[0], gc, verbose)
            if verbose:
                print('{:<12s} | action: {} {}'.format('Mesa', action, info))
        return self
