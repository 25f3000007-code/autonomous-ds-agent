"""
Validation script to ensure brain.py outputs only executable code,
no markdown formatting, backticks, or conversational text.
"""
import re
import ast
from src.brain import AIBrain
from src.monitor import DataMonitor

def validate_code_output(code_str: str, test_name: str) -> dict:
    """Validates code output for markdown/conversational filler."""
    issues = []
    
    # Check for markdown code block markers
    if "```" in code_str:
        issues.append("❌ Contains markdown code blocks (```)") 
    
    # NOTE: Python comments with # are valid and executable, so don't flag them.
    # We only flag markdown-style headers outside of code context.
    # Skip this check since # comments are legitimate Python.
    
    if re.search(r'\*\*.*\*\*', code_str):
        issues.append("❌ Contains markdown bold (**)")
    
    if re.search(r'^[-*]\s', code_str, re.MULTILINE):
        issues.append("❌ Contains markdown lists")
    
    # Check for conversational markers
    conversational_patterns = [
        r'(?i)(here\'s|here is|this is|the code|solution)',
        r'(?i)(^Here|^The|^This|^First|^Second)',
        r'(?i)(explanation|summary|note:|description:)',
    ]
    
    for pattern in conversational_patterns:
        if re.search(pattern, code_str, re.MULTILINE):
            issues.append(f"⚠️  Possible conversational text: {pattern}")
    
    # Try to parse as Python to verify it's syntactically valid
    try:
        ast.parse(code_str)
        is_valid_python = True
    except SyntaxError as e:
        is_valid_python = False
        issues.append(f"❌ Not valid Python: {e}")
    
    # Report
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")
    print(f"Output length: {len(code_str)} characters")
    print(f"Valid Python: {'✅ YES' if is_valid_python else '❌ NO'}")
    print(f"Markdown/Conversational Issues: {len(issues)}")
    
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  ✅ No issues detected - output is clean code!")
    
    print(f"\nActual Output:\n{code_str[:200]}{'...' if len(code_str) > 200 else ''}")
    
    return {
        "test_name": test_name,
        "is_valid_python": is_valid_python,
        "has_issues": len(issues) > 0,
        "issues": issues
    }

# Test 1: Simple numeric data
print("\n" + "="*60)
print("RUNNING CODE OUTPUT VALIDATION TESTS")
print("="*60)

brain = AIBrain()

profile_1 = """{
  "dataset_shape": {"rows": 100, "columns": 3},
  "column_metadata": {
    "age": {"dtype": "float64", "missing_count": 5, "skewness": 2.1},
    "salary": {"dtype": "float64", "missing_count": 2, "skewness": 0.8},
    "department": {"dtype": "str", "missing_count": 0, "skewness": null}
  }
}"""

code_1 = brain.generate_transformation_code(profile_1)
result_1 = validate_code_output(code_1, "Test 1: Numeric + Categorical Data")

# Test 2: Highly skewed data
profile_2 = """{
  "dataset_shape": {"rows": 500, "columns": 2},
  "column_metadata": {
    "revenue": {"dtype": "float64", "missing_count": 10, "skewness": 5.5},
    "user_id": {"dtype": "int64", "missing_count": 0, "skewness": null}
  }
}"""

code_2 = brain.generate_transformation_code(profile_2)
result_2 = validate_code_output(code_2, "Test 2: Highly Skewed Distribution")

# Test 3: Many missing values
profile_3 = """{
  "dataset_shape": {"rows": 50, "columns": 4},
  "column_metadata": {
    "col_a": {"dtype": "float64", "missing_count": 20, "skewness": 0.5},
    "col_b": {"dtype": "float64", "missing_count": 15, "skewness": 1.2},
    "col_c": {"dtype": "str", "missing_count": 8, "skewness": null},
    "col_d": {"dtype": "int64", "missing_count": 0, "skewness": 0.1}
  }
}"""

code_3 = brain.generate_transformation_code(profile_3)
result_3 = validate_code_output(code_3, "Test 3: Multiple Missing Values")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

results = [result_1, result_2, result_3]
all_valid = all(r["is_valid_python"] for r in results)
all_clean = not any(r["has_issues"] for r in results)

print(f"Total tests: {len(results)}")
print(f"Valid Python: {sum(r['is_valid_python'] for r in results)}/{len(results)} ✅" if all_valid else f"Valid Python: {sum(r['is_valid_python'] for r in results)}/{len(results)} ❌")
print(f"No markdown/conversational filler: {sum(not r['has_issues'] for r in results)}/{len(results)} ✅" if all_clean else f"No markdown/conversational filler: {sum(not r['has_issues'] for r in results)}/{len(results)} ⚠️")

if all_valid and all_clean:
    print("\n✅ CRITICAL CHECKPOINT PASSED: Output is pure, executable code!")
else:
    print("\n❌ CRITICAL CHECKPOINT FAILED: Output contains issues!")

print("\n" + "="*60)
