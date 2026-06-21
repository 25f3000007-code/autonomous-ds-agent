import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import (HistGradientBoostingRegressor,
                               HistGradientBoostingClassifier,
                               ExtraTreesRegressor, ExtraTreesClassifier)
from sklearn.metrics import mean_squared_error, f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder
import warnings
from pandas.api.types import is_numeric_dtype

warnings.filterwarnings('ignore')

_HPO_PARAM_GRID_REG = {
    "max_iter":        [100, 200, 300],
    "max_leaf_nodes":  [31, 63, 127, 255],
    "learning_rate":   [0.03, 0.05, 0.1, 0.2],
    "l2_regularization": [0.0, 0.1, 0.5, 1.0],
    "min_samples_leaf": [10, 20, 30],
}

_HPO_PARAM_GRID_CLF = {
    "max_iter":        [100, 200, 300],
    "max_leaf_nodes":  [31, 63, 127],
    "learning_rate":   [0.03, 0.05, 0.1, 0.2],
    "l2_regularization": [0.0, 0.1, 1.0],
}


class ModelValidator:
    """
    Evaluates a dataset with one of three model strategies:
      - "HistGradientBoosting"  (default, fast sklearn boosting)
      - "ExtraTrees"            (pivot model — bagging, different inductive bias)
      - "HPO"                   (RandomizedSearchCV on HistGradientBoosting)
    """
    def __init__(self, target_column: str):
        self.target_column = target_column

    def evaluate(self, df: pd.DataFrame, model_name: str = "HistGradientBoosting") -> dict:
        if self.target_column not in df.columns:
            return {"error": f"Target column '{self.target_column}' missing."}

        clean_df = df.dropna(subset=[self.target_column]).copy()
        if len(clean_df) < 10:
            return {"error": "Not enough valid data rows to train a model."}

        X = clean_df.drop(columns=[self.target_column])
        y = clean_df[self.target_column]

        for col in X.columns:
            if is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna('Unknown')
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        is_classification = (not is_numeric_dtype(y)) or (y.nunique() < 10)
        if is_classification:
            le_y = LabelEncoder()
            y = le_y.fit_transform(y.astype(str))

        n_splits = min(5, len(X))
        if n_splits < 2:
            return {"error": "Not enough valid data rows to train a model."}

        if is_classification:
            class_counts = pd.Series(y).value_counts()
            if class_counts.min() < n_splits:
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            else:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = np.array(y)

        try:
            if model_name == "HPO":
                return self._evaluate_hpo(X_arr, y_arr, cv, is_classification)
            elif model_name == "ExtraTrees":
                return self._evaluate_cv(X_arr, y_arr, cv, is_classification, model_name)
            else:
                return self._evaluate_cv(X_arr, y_arr, cv, is_classification, "HistGradientBoosting")
        except Exception as e:
            return {"error": f"Model training failed: {e}"}

    def _build_model(self, model_name: str, is_classification: bool):
        if model_name == "ExtraTrees":
            if is_classification:
                return ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            else:
                return ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        else:
            if is_classification:
                return HistGradientBoostingClassifier(random_state=42, max_iter=50)
            else:
                return HistGradientBoostingRegressor(random_state=42, max_iter=50)

    def _evaluate_cv(self, X_arr, y_arr, cv, is_classification: bool, model_name: str) -> dict:
        scores = []
        for train_idx, test_idx in cv.split(X_arr, y_arr):
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]
            model = self._build_model(model_name, is_classification)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            if is_classification:
                scores.append(f1_score(y_test, preds, average='weighted'))
            else:
                scores.append(np.sqrt(mean_squared_error(y_test, preds)))
        score = np.mean(scores)
        metric_name = "F1-Score (Higher is better)" if is_classification else "RMSE (Lower is better)"
        return {
            "task_type": "Classification" if is_classification else "Regression",
            "metric": metric_name,
            "score": round(score, 4),
            "model": model_name,
        }

    def _evaluate_hpo(self, X_arr, y_arr, cv, is_classification: bool) -> dict:
        if is_classification:
            base = HistGradientBoostingClassifier(random_state=42)
            scorer = make_scorer(f1_score, average='weighted')
            param_grid = _HPO_PARAM_GRID_CLF
        else:
            base = HistGradientBoostingRegressor(random_state=42)
            scorer = make_scorer(
                lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)),
                greater_is_better=True
            )
            param_grid = _HPO_PARAM_GRID_REG

        search = RandomizedSearchCV(
            base, param_grid,
            n_iter=20, cv=cv, scoring=scorer,
            random_state=42, n_jobs=-1, refit=False
        )
        search.fit(X_arr, y_arr)

        if is_classification:
            best_score = search.best_score_
        else:
            best_score = -search.best_score_

        metric_name = "F1-Score (Higher is better)" if is_classification else "RMSE (Lower is better)"
        return {
            "task_type": "Classification" if is_classification else "Regression",
            "metric": metric_name,
            "score": round(best_score, 4),
            "model": "HPO (HistGradientBoosting)",
            "best_params": search.best_params_,
        }


if __name__ == "__main__":
    test_df = pd.DataFrame({
        "study_hours": [2, 3, 5, 1, 8, 7, 4, 9, 2, 6],
        "attendance":  [80, 85, 90, 60, 95, 92, 88, 98, 70, 91],
        "passed":      [0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    })
    validator = ModelValidator(target_column="passed")
    for m in ["HistGradientBoosting", "ExtraTrees", "HPO"]:
        print(f"\n{m}:", validator.evaluate(test_df, model_name=m))
