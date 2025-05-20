from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.tree import DecisionTreeClassifier

from gcForest import gcForest
import pandas as pd
import os
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
    # Output files
    output_txt = "results.txt"
    output_csv = "results.csv"

    # Initialize CSV dataframe
    csv_results = []

    for i in range(1, 28):  # Demo with x1 to x2 (adjust to x1-x27 in real run)
        dataset_path = f"../zenodo_datasets/zenodo/x{i}data.npz"
        if not os.path.exists(dataset_path):
            raise ValueError()

        dataset = np.load(dataset_path)
        X, y = dataset['data'], dataset['label']
        y = np.where(y == -1, 0, y)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        f1_macro_list = []
        auc_list = []
        aupr_list = []
        gmean_list = []

        with open(output_txt, "a") as f_out:
            f_out.write(f"Dataset: x{i}data.npz\n")

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train_cv, X_test_cv = X[train_idx], X[test_idx]
                y_train_cv, y_test_cv = y[train_idx], y[test_idx]

                clf = EasyEnsembleClassifier(n_estimators=100, random_state=42)
                clf.fit(X_train_cv, y_train_cv)

                X_train_cv, X_valid_cv, y_train_cv, y_valid_cv = train_test_split(
                    X_train_cv, y_train_cv, test_size=0.2, random_state=42
                )

                model = gcForest(get_config())
                model.fit(X_train_cv, y_train_cv, X_valid_cv, y_valid_cv, X_test_cv, y_test_cv, gcForest)

                y_pred = model.predict(X_test_cv)
                y_pred_proba = model.predict_proba(X_test_cv)[:, 1]

                f1_macro_val = f1_score(y_test_cv, y_pred, average='macro')
                auc_val = roc_auc_score(y_test_cv, y_pred_proba)
                aupr_val = average_precision_score(y_test_cv, y_pred_proba)
                gmean_val = geometric_mean_score(y_test_cv, y_pred)

                f1_macro_list.append(f1_macro_val)
                auc_list.append(auc_val)
                aupr_list.append(aupr_val)
                gmean_list.append(gmean_val)

                f_out.write(
                    f"  Fold {fold_idx + 1}: "
                    f"F1-macro={f1_macro_val:.4f}, "
                    f"AUC={auc_val:.4f}, "
                    f"AUPR={aupr_val:.4f}, "
                    f"Gmean={gmean_val:.4f}\n"
                )

                csv_results.append({
                    "Dataset": f"x{i}data.npz",
                    "Fold": fold_idx + 1,
                    "F1-macro": f1_macro_val,
                    "AUC": auc_val,
                    "AUPR": aupr_val,
                    "Gmean": gmean_val
                })

            # Save mean scores
            f_out.write(
                f"  Mean: "
                f"F1-macro={np.mean(f1_macro_list):.4f}, "
                f"AUC={np.mean(auc_list):.4f}, "
                f"AUPR={np.mean(aupr_list):.4f}, "
                f"Gmean={np.mean(gmean_list):.4f}\n"
            )
            f_out.write("=" * 50 + "\n")

            csv_results.append({
                "Dataset": f"x{i}data.npz",
                "Fold": "Mean",
                "F1-macro": np.mean(f1_macro_list),
                "AUC": np.mean(auc_list),
                "AUPR": np.mean(aupr_list),
                "Gmean": np.mean(gmean_list)
            })

    # Save CSV
    df_csv = pd.DataFrame(csv_results)
    df_csv.to_csv(output_csv, index=False)
