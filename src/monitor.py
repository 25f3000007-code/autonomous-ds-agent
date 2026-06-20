import pandas as pd
import json
import numpy as np

class DataMonitor:
    """
    Analyzes raw tabular data and extracts a compressed state profile 
    without relying on external AI models.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    def load_data(self) -> bool:
        """Loads the CSV into a pandas DataFrame."""
        try:
            self.df = pd.read_csv(self.filepath)
            print(f"✅ [Monitor] Successfully loaded dataset: {self.filepath}")
            return True
        except FileNotFoundError:
            print(f"❌ [Monitor] Error: Could not find dataset at {self.filepath}")
            return False
        except Exception as e:
            print(f"❌ [Monitor] Unexpected error loading data: {e}")
            return False

    def generate_profile(self) -> str:
        """
        Computes structural metrics and compresses them into a JSON string.
        This string acts as the high-signal prompt context for the AI Brain.
        """
        if self.df is None:
            return '{"error":"No data loaded"}'

        # 1. Structural Basics
        shape = {"rows": self.df.shape[0], "cols": self.df.shape[1]}
        
        # 2. Data Types & Missing Values
        column_meta = {}
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            missing = int(self.df[col].isnull().sum())
            
            # 3. Simple Skewness Check for numericals
            meta = {
                "dtype": dtype,
                "missing": missing
            }
            
            if pd.api.types.is_numeric_dtype(self.df[col]):
                try:
                    skewness = self.df[col].skew()
                    if not pd.isna(skewness):
                        meta["skew"] = round(float(skewness), 2)
                except Exception:
                    pass

            column_meta[col] = meta

        profile_dict = {
            "shape": shape,
            "columns": column_meta
        }

        # Return a tight, token-efficient JSON string without extra whitespaces
        return json.dumps(profile_dict, separators=(',', ':'))

if __name__ == "__main__":
    # Local execution test
    print("--- Testing Monitor Module ---")
    
    # 1. We need a dummy CSV to test the monitor. 
    # Let's create a tiny one on the fly for testing purposes.
    test_csv_path = "../data/test_dataset.csv"
    dummy_data = pd.DataFrame({
        "age": [25, 30, np.nan, 45, 50, 120], # Includes a missing value and an outlier
        "income": [50000, 60000, 55000, np.nan, 80000, 90000],
        "category": ["A", "B", "A", "C", "A", "B"]
    })
    
    import os
    import numpy as np
    os.makedirs("../data", exist_ok=True)
    dummy_data.to_csv(test_csv_path, index=False)
    print("✅ Created temporary test dataset.")

    # 2. Run the Monitor
    monitor = DataMonitor(test_csv_path)
    if monitor.load_data():
        profile_json = monitor.generate_profile()
        print("\n📊 Generated Data Profile (To be sent to AI Brain later):")
        print(profile_json)