import pandas as pd
import numpy as np
import traceback
import warnings

class CodeExecutor:
    """
    Safely executes AI-generated Python code against a pandas DataFrame 
    within an isolated local namespace.
    """
    def __init__(self):
        pass

    def apply_transformation(self, df: pd.DataFrame, code_string: str) -> pd.DataFrame:
        """
        Takes the original DataFrame and the AI's code snippet.
        Returns the modified DataFrame if successful, or the original if it fails.
        """
        # Capture the original import function to use inside the restricted import wrapper
        original_import = __import__
        def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in ('pandas', 'numpy') or name.startswith(('pandas.', 'numpy.')):
                return original_import(name, globals, locals, fromlist, level)
            raise ImportError(f"Import of module '{name}' is unauthorized in the sandbox.")

        # Define whitelisted safe builtins to prevent arbitrary code execution, file I/O, etc.
        safe_builtins = {
            '__import__': restricted_import,
            'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
            'bytearray': bytearray, 'bytes': bytes, 'chr': chr, 'dict': dict,
            'dir': dir, 'divmod': divmod, 'enumerate': enumerate, 'filter': filter,
            'float': float, 'format': format, 'frozenset': frozenset,
            'getattr': getattr, 'hasattr': hasattr, 'hash': hash, 'hex': hex,
            'id': id, 'int': int, 'isinstance': isinstance, 'issubclass': issubclass,
            'iter': iter, 'len': len, 'list': list, 'map': map, 'max': max,
            'min': min, 'next': next, 'object': object, 'oct': oct, 'ord': ord,
            'pow': pow, 'print': print, 'range': range, 'repr': repr,
            'reversed': reversed, 'round': round, 'set': set, 'slice': slice,
            'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple,
            'type': type, 'zip': zip,
            # Whitelist standard exceptions so client code can catch/raise them
            'ArithmeticError': ArithmeticError, 'AssertionError': AssertionError,
            'AttributeError': AttributeError, 'BaseException': BaseException,
            'Exception': Exception, 'IndexError': IndexError,
            'KeyError': KeyError, 'KeyboardInterrupt': KeyboardInterrupt,
            'LookupError': LookupError, 'NameError': NameError,
            'TypeError': TypeError, 'ValueError': ValueError,
            'ZeroDivisionError': ZeroDivisionError,
            'FileNotFoundError': FileNotFoundError,
            'PermissionError': PermissionError,
            'OSError': OSError,
        }

        # Create a strict, isolated namespace for the exec() function.
        # We pass a copy of the dataframe to prevent partial mutations on failure.
        # We pass pd, np, df, and __builtins__ in globals so they are accessible inside
        # functions defined by the executed code.
        globals_dict = {
            '__builtins__': safe_builtins,
            'pd': pd,
            'np': np,
            'df': df.copy()
        }

        try:
            print("⚙️ [Executor] Compiling and running AI transformation script...")
            # Suppress pandas Copy-on-Write warnings (we use direct assignment, not inplace)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
                warnings.filterwarnings('ignore', message='.*ChainedAssignment.*')
                # Execute the raw string code within our isolated dictionary
                exec(code_string, globals_dict)
            
            print("✅ [Executor] Transformation applied successfully.")
            # Extract the mutated dataframe back out of the namespace
            return globals_dict['df']
            
        except Exception as e:
            print("❌ [Executor] Execution crashed. AI generated invalid code.")
            print("--- Error Traceback ---")
            print(traceback.format_exc())
            print("-----------------------")
            print("⚠️ [Executor] Reverting to original dataset state.")
            # Return the unmodified original dataframe
            return df

if __name__ == "__main__":
    print("--- Testing Executor Module ---")
    
    # 1. Create a raw dataframe with a known issue (missing value)
    raw_df = pd.DataFrame({
        "age": [25, 30, np.nan, 45, 50],
        "income": [50000, 60000, 55000, 70000, 80000]
    })
    print("\n📊 Raw DataFrame (Before):")
    print(raw_df)

    # 2. Simulate the exact code our Brain just generated
    mock_ai_code = """
# Handle 'age' column missing values
df['age'].fillna(df['age'].median(), inplace=True)
    """.strip()

    # 3. Run it through our Executor
    executor = CodeExecutor()
    cleaned_df = executor.apply_transformation(raw_df, mock_ai_code)

    print("\n✨ Cleaned DataFrame (After):")
    print(cleaned_df)