import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
from pandas.api.types import is_numeric_dtype

# Suppress minor scikit-learn warnings for cleaner terminal output
warnings.filterwarnings('ignore')

class ModelValidator:
    """
    Rapidly evaluates the predictive power of a dataset by training a 
    baseline HistGradientBoosting model using K-Fold cross-validation.
    """
    def __init__(self, target_column: str):
        self.target_column = target_column

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Processes features, encodes categoricals, and returns the stable CV performance score.
        """
        if self.target_column not in df.columns:
            return {"error": f"Target column '{self.target_column}' missing."}

        # Drop rows where the TARGET is missing (we can't train on those)
        clean_df = df.dropna(subset=[self.target_column]).copy()
        
        if len(clean_df) < 10:
            return {"error": "Not enough valid data rows to train a model."}

        X = clean_df.drop(columns=[self.target_column])
        y = clean_df[self.target_column]

        # 1. Quick Baseline Encoding: convert non-numeric columns to numbers
        for col in X.columns:
            if is_numeric_dtype(X[col]):
                # Fill remaining numeric NaNs with 0 for absolute safety
                X[col] = X[col].fillna(0)
            else:
                # Non-numeric column: encode it before model training
                X[col] = X[col].fillna('Unknown')
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        # 2. Determine Task Type (Classification vs Regression)
        is_classification = False
        # Check if target is non-numeric OR has few unique values (categorical indicator)
        if not is_numeric_dtype(y) or y.nunique() < 10:
            is_classification = True
            le_y = LabelEncoder()
            y = le_y.fit_transform(y.astype(str))

        # 3. Cross-Validation Configuration
        n_splits = 5
        # Ensure we have at least 2 samples per fold, or cap n_splits at the size of data
        n_splits = min(n_splits, len(X))
        if n_splits < 2:
            return {"error": "Not enough valid data rows to train a model."}

        # 4. Define Cross-Validation Strategy
        if is_classification:
            class_counts = pd.Series(y).value_counts()
            # If any class has fewer than n_splits instances, fallback to standard KFold
            if class_counts.min() < n_splits:
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            else:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # 5. Train and Evaluate with Cross-Validation
        try:
            scores = []
            
            # Use numpy arrays to prevent indexing issues during slicing
            X_arr = X.values if isinstance(X, pd.DataFrame) else X
            y_arr = np.array(y)
            
            for train_idx, test_idx in cv.split(X_arr, y_arr):
                X_train, X_test = X_arr[train_idx], X_arr[test_idx]
                y_train, y_test = y_arr[train_idx], y_arr[test_idx]
                
                if is_classification:
                    model = HistGradientBoostingClassifier(random_state=42, max_iter=50)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    scores.append(f1_score(y_test, preds, average='weighted'))
                else:
                    model = HistGradientBoostingRegressor(random_state=42, max_iter=50)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    scores.append(np.sqrt(mean_squared_error(y_test, preds)))
            
            score = np.mean(scores)
            metric_name = "F1-Score (Higher is better)" if is_classification else "RMSE (Lower is better)"

            return {
                "task_type": "Classification" if is_classification else "Regression",
                "metric": metric_name,
                "score": round(score, 4)
            }

        except Exception as e:
             return {"error": f"Model training failed: {e}"}

if __name__ == "__main__":
    print("--- Testing Validator Module ---")
    
    # Let's create a dataset simulating a classification problem (e.g., passing a course)
    test_df = pd.DataFrame({
        "study_hours": [2, 3, 5, 1, 8, 7, 4, 9, 2, 6],
        "attendance": [80, 85, 90, 60, 95, 92, 88, 98, 70, 91],
        "passed": [0, 0, 1, 0, 1, 1, 0, 1, 0, 1] # Target variable
    })
    
    print("\n📊 Validating Baseline Dataset...")
    validator = ModelValidator(target_column="passed")
    results = validator.evaluate(test_df)
    
    print(f"✅ Validation Complete:")
    for key, value in results.items():
        print(f"  - {key}: {value}")