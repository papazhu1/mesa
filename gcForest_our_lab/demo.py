from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.tree import DecisionTreeClassifier

from gcForest import gcForest
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from evaluation import accuracy, f1_binary, f1_macro, f1_micro
from imblearn.metrics import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *
from BaseForest import BaseForest
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_config():
    config = {}
    config["base_estimator_"] = DecisionTreeClassifier()
    config["random_state"] = 0
    config["max_layers"] = 100
    config["early_stop_rounds"] = 1
    config["if_stacking"] = False
    config["if_save_model"] = False
    config["train_evaluation"] = f1_macro  ##f1_binary,f1_macro,f1_micro
    config["estimator_configs"] = []
    for i in range(2):
        config["estimator_configs"].append(
            {"n_fold": 5, "type": "BaseForest", "n_estimators": 100, "max_depth": None, "n_jobs": -1})
    for i in range(2):
        config["estimator_configs"].append(
            {"n_fold": 5, "type": "BaseForest", "n_estimators": 100, "max_depth": None, "n_jobs": -1})
    return config


if __name__ == "__main__":
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    # dataset_name = "car_eval_4"
    dataset = np.load("../zenodo_datasets/zenodo/x2data.npz")
    print(dataset.files)
    X, y = dataset['data'], dataset['label']
    y = np.where(y == -1, 0, y)


    # X, y, name = get_yeast5()
    f1_macro_list = []
    auc_list = []
    aupr_list = []
    gmean_list = []

    for train_idx, test_idx in skf.split(X, y):
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_train_cv, y_test_cv = y[train_idx], y[test_idx]

        clf = EasyEnsembleClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_cv, y_train_cv)

        # 预测与评估
        clf_pred = clf.predict(X_test_cv)
        clf_pred_proba = clf.predict_proba(X_test_cv)

        print("Accuracy:", accuracy_score(y_test_cv, clf_pred))
        print("F1-score:", f1_score(y_test_cv, clf_pred, average='macro'))

        X_train_cv, X_valid_cv, y_train_cv, y_valid_cv = train_test_split(X_train_cv, y_train_cv, test_size=0.2)


        model = gcForest(get_config())
        model.fit(X_train_cv, y_train_cv, X_valid_cv, y_valid_cv, X_test_cv, y_test_cv, gcForest)

        # 预测
        y_pred = model.predict(X_test_cv)
        y_pred_proba = model.predict_proba(X_test_cv)[:, 1]  # 获取预测的概率值，用于计算 AUC 和 AUPR

        # 计算性能指标
        f1_macro_val = f1_score(y_test_cv, y_pred, average='macro')
        auc_val = roc_auc_score(y_test_cv, y_pred_proba)
        aupr_val = average_precision_score(y_test_cv, y_pred_proba)
        gmean_val = geometric_mean_score(y_test_cv, y_pred)

        # 保存每轮的性能指标
        f1_macro_list.append(f1_macro_val)
        auc_list.append(auc_val)
        aupr_list.append(aupr_val)
        gmean_list.append(gmean_val)

        # 输出当前训练集比例和各项指标
        print(f"F1-macro: {f1_macro_val:.4f}")
        print(f"AUC: {auc_val:.4f}")
        print(f"AUPR: {aupr_val:.4f}")
        print(f"Gmean: {gmean_val:.4f}")
        print("-" * 40)

    print(f1_macro_list)
    print(np.mean(f1_macro_list))
    print(auc_list)
    print(np.mean(auc_list))
    print(aupr_list)
    print(np.mean(aupr_list))
    print(gmean_list)
    print(np.mean(gmean_list))




