import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from imblearn.metrics import geometric_mean_score
from imbens.ensemble import (
    SelfPacedEnsembleClassifier,
    BalanceCascadeClassifier,
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier,
    UnderBaggingClassifier,
    OverBoostClassifier,
    SMOTEBoostClassifier,
    KmeansSMOTEBoostClassifier,
    OverBaggingClassifier,
    SMOTEBaggingClassifier,
)

# 模型列表
model_classes = {
    "SelfPacedEnsembleClassifier": SelfPacedEnsembleClassifier,
    "BalanceCascadeClassifier": BalanceCascadeClassifier,
    "EasyEnsembleClassifier": EasyEnsembleClassifier,
    "RUSBoostClassifier": RUSBoostClassifier,
    "UnderBaggingClassifier": UnderBaggingClassifier,
    "OverBoostClassifier": OverBoostClassifier,
    "SMOTEBoostClassifier": SMOTEBoostClassifier,
    "OverBaggingClassifier": OverBaggingClassifier,
    "SMOTEBaggingClassifier": SMOTEBaggingClassifier,
}

# 数据集路径与输出路径
dataset_dir = "../zenodo_datasets/zenodo/"
output_dir = "./model_comparison_results_filtered"
os.makedirs(output_dir, exist_ok=True)

# 主循环
for model_name, model_cls in model_classes.items():
    model_results = []

    for i in range(1, 28):  # 遍历 x1data.npz 到 x27data.npz
        dataset_path = os.path.join(dataset_dir, f"x{i}data.npz")
        if not os.path.exists(dataset_path):
            continue

        dataset = np.load(dataset_path)
        X, y = dataset["data"], dataset["label"]
        y = np.where(y == -1, 0, y)

        # 过滤掉样本量超过10000的数据集
        if len(X) > 10000:
            print(f"[Skipped] x{i}data.npz (samples={len(X)})")
            continue

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1_list, auc_list, aupr_list, gmean_list = [], [], [], []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = model_cls(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            f1_list.append(f1_score(y_test, y_pred, average="macro"))
            auc_list.append(roc_auc_score(y_test, y_proba))
            aupr_list.append(average_precision_score(y_test, y_proba))
            gmean_list.append(geometric_mean_score(y_test, y_pred))

        model_results.append({
            "Dataset": f"x{i}data.npz",
            "F1-macro": np.mean(f1_list),
            "AUC": np.mean(auc_list),
            "AUPR": np.mean(aupr_list),
            "Gmean": np.mean(gmean_list),
        })

    df = pd.DataFrame(model_results)
    df.to_csv(os.path.join(output_dir, f"{model_name}.csv"), index=False)
    print(f"[Saved] {model_name}.csv")
