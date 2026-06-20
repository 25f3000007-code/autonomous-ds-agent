"""
Test script to verify executor sandbox handles AI code safely and correctly.
Includes tests for success cases, error handling, and edge cases.
"""
import pandas as pd
import numpy as np
from src.executor import CodeExecutor
from src.brain import AIBrain
from src.monitor import DataMonitor

print("="*70)
print("EXECUTOR SANDBOX VERIFICATION TEST SUITE")
print("="*70)

# Test 1: Simple median imputation
print("\n[Test 1] Simple Median Imputation")
print("-" * 70)

df_test1 = pd.DataFrame({
    "age": [25, 30, np.nan, 45, 50],
    "income": [50000, 60000, 55000, 70000, 80000]
})

print("BEFORE:")
print(df_test1)
print(f"NaN count in 'age': {df_test1['age'].isnull().sum()}")

ai_code_1 = """
df['age'] = df['age'].fillna(df['age'].median())
""".strip()

executor = CodeExecutor()
df_result_1 = executor.apply_transformation(df_test1, ai_code_1)

print("\nAFTER:")
print(df_result_1)
print(f"NaN count in 'age': {df_result_1['age'].isnull().sum()}")

if df_result_1['age'].isnull().sum() == 0:
    print("✅ Test 1 PASSED: Median imputation successful!")
else:
    print("❌ Test 1 FAILED: NaN values remain!")

# Test 2: Multiple column transformations
print("\n\n[Test 2] Multiple Column Transformations")
print("-" * 70)

df_test2 = pd.DataFrame({
    "col_a": [1.0, 2.0, np.nan, 4.0, 5.0],
    "col_b": [10.0, np.nan, 30.0, 40.0, 50.0],
    "col_c": [100, 200, 300, 400, 500]
})

print("BEFORE:")
print(df_test2)
print(f"Total NaN count: {df_test2.isnull().sum().sum()}")

ai_code_2 = """
df['col_a'] = df['col_a'].fillna(df['col_a'].median())
df['col_b'] = df['col_b'].fillna(df['col_b'].mean())
""".strip()

df_result_2 = executor.apply_transformation(df_test2, ai_code_2)

print("\nAFTER:")
print(df_result_2)
print(f"Total NaN count: {df_result_2.isnull().sum().sum()}")

if df_result_2.isnull().sum().sum() == 0:
    print("✅ Test 2 PASSED: Multiple column transformations successful!")
else:
    print("❌ Test 2 FAILED: Some NaN values remain!")

# Test 3: Invalid code (should handle gracefully)
print("\n\n[Test 3] Error Handling - Invalid Code")
print("-" * 70)

df_test3 = pd.DataFrame({"x": [1, 2, 3]})
original_df_test3 = df_test3.copy()

print("BEFORE:")
print(df_test3)

invalid_code = """
df['y'] = df['x'] / 0  # Division by zero error
""".strip()

df_result_3 = executor.apply_transformation(df_test3, invalid_code)

print("\nAFTER:")
print(df_result_3)

if df_result_3.equals(original_df_test3):
    print("✅ Test 3 PASSED: Original DataFrame preserved on error!")
else:
    print("❌ Test 3 FAILED: DataFrame was modified despite error!")

# Test 4: Real-world scenario with monitor output
print("\n\n[Test 4] Real-World Scenario (Monitor → Brain → Executor Pipeline)")
print("-" * 70)

# Create test data
df_test4 = pd.DataFrame({
    "revenue": [1000, 2000, np.nan, 5000, 1500, np.nan, 8000],
    "user_count": [10, 20, 30, np.nan, 50, 60, 70],
    "category": ["A", "B", "A", "C", "A", "B", "C"]
})

print("ORIGINAL DATA:")
print(df_test4)
print(f"\nMissing values per column:")
print(df_test4.isnull().sum())

# Generate profile
monitor = DataMonitor.__new__(DataMonitor)
monitor.df = df_test4

profile_json = monitor.generate_profile()
print(f"\nData Profile:\n{profile_json}")

# Generate transformation code
brain = AIBrain()
print(f"\n[Brain] Generating transformation code...")
ai_code_4 = brain.generate_transformation_code(profile_json)

print(f"\n[Generated Code]:\n{ai_code_4}")

# Execute the transformation
print(f"\n[Executor] Running transformation...")
df_result_4 = executor.apply_transformation(df_test4, ai_code_4)

print("\nTRANSFORMED DATA:")
print(df_result_4)
print(f"\nMissing values per column after transformation:")
print(df_result_4.isnull().sum())

if df_result_4.isnull().sum().sum() < df_test4.isnull().sum().sum():
    print("✅ Test 4 PASSED: Missing values reduced by pipeline!")
else:
    print("⚠️ Test 4 WARNING: Check transformation results above")

print("\n" + "="*70)
print("TEST SUITE COMPLETE")
print("="*70)
