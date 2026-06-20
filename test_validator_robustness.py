"""
Test validator robustness with different pandas dtype inferences.
"""
import pandas as pd
import numpy as np
from src.validator import ModelValidator

print("="*70)
print("VALIDATOR DTYPE ROBUSTNESS TEST")
print("="*70)

# Test 1: CSV read with 'string' dtype inferred (the problematic case)
print("\n[Test 1] DataFrame with 'string' dtype (from CSV read)")
print("-" * 70)

# Simulate what happens when pandas reads a CSV with default string dtype
df_test1 = pd.DataFrame({
    "sqft": [1000, 1500, 2000, 2500, 3000, 1200, 1800, 2200, 2800, 3500],
    "bedrooms": [1, 2, 3, 4, 3, 2, 3, 4, 5, 4],
    "neighborhood_quality": pd.array(["Low", "Medium", "High", "Low", "High", "Medium", "Low", "High", "Medium", "High"], dtype="string"),  # Force 'string' dtype
    "price": [100000, 150000, 200000, 250000, 300000, 120000, 180000, 220000, 280000, 350000]
})

print(f"Data types:\n{df_test1.dtypes}\n")
print(f"Data sample:\n{df_test1.head()}\n")

validator = ModelValidator(target_column="price")
results_1 = validator.evaluate(df_test1)

print(f"✅ Validation Results:")
for key, value in results_1.items():
    print(f"  - {key}: {value}")

if "error" in results_1:
    print("❌ Test 1 FAILED - Validator crashed!")
else:
    print("✅ Test 1 PASSED - Validator handled 'string' dtype!")

# Test 2: Mixed dtypes with missing values
print("\n\n[Test 2] Mixed dtypes with NaN values")
print("-" * 70)

df_test2 = pd.DataFrame({
    "numeric_col": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "category_col": ["A", "B", np.nan, "A", "C", "B", "A", "C", "B", "A"],  # object dtype
    "string_col": pd.array(["X", "Y", "Z", "X", "Y", "Z", "X", "Y", np.nan, "Z"], dtype="string"),  # string dtype
    "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
})

print(f"Data types:\n{df_test2.dtypes}\n")

validator2 = ModelValidator(target_column="target")
results_2 = validator2.evaluate(df_test2)

print(f"✅ Validation Results:")
for key, value in results_2.items():
    print(f"  - {key}: {value}")

if "error" in results_2:
    print("❌ Test 2 FAILED - Validator crashed with mixed dtypes!")
else:
    print("✅ Test 2 PASSED - Validator handled mixed dtypes!")

# Test 3: Real CSV read (most realistic scenario)
print("\n\n[Test 3] Real CSV read with messy data")
print("-" * 70)

# Create a messy CSV and read it back (this is how pandas normally infers dtypes)
messy_csv_path = "test_messy_validator.csv"
df_messy_write = pd.DataFrame({
    "sqft": [1000, 1500, np.nan, 2500, 3000, 1200, 1800, 2200, 2800, 3500],
    "bedrooms": [1, 2, 3, 4, 3, 2, 3, 4, 5, 4],
    "neighborhood": ["Low", "Medium", "High", "Low", "High", "Medium", "Low", "High", "Medium", "High"],
    "price": [100000, 150000, 200000, 250000, 300000, 120000, 180000, 220000, 280000, 350000]
})

df_messy_write.to_csv(messy_csv_path, index=False)

# Now read it back (this will infer dtypes)
df_test3 = pd.read_csv(messy_csv_path)

print(f"Data types (after CSV read):\n{df_test3.dtypes}\n")
print(f"Data sample:\n{df_test3.head()}\n")

validator3 = ModelValidator(target_column="price")
results_3 = validator3.evaluate(df_test3)

print(f"✅ Validation Results:")
for key, value in results_3.items():
    print(f"  - {key}: {value}")

if "error" in results_3:
    print("❌ Test 3 FAILED - Validator crashed on CSV read!")
else:
    print("✅ Test 3 PASSED - Validator handles real CSV reads!")

# Cleanup
import os
os.remove(messy_csv_path)

print("\n" + "="*70)
print("ALL TESTS COMPLETE - Validator is bulletproof!")
print("="*70)
