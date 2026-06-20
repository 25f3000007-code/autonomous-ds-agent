# src/brain.py
import os
import sys



HAS_GENAI = False

try:
    import google.genai as genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Native key bootstrap: IDE env space first, then local .env fallback
def _load_env_key(key: str) -> str:
    """Fetch an env key natively without external dependencies."""
    value = os.environ.get(key)
    if value:
        return value
    # Safe stdlib fallback: read a local .env file line-by-line
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
    except (FileNotFoundError, PermissionError, OSError):
        pass
    return ""

class AIBrain:
    """
    Communicates with Gemini to analyze the data profile and 
    generate precise python transformation scripts.
    """
    def __init__(self):
        self.is_mock = False
        
        # Check if genai is available
        if not HAS_GENAI:
            self.is_mock = True
            self.client = None
            self.model_name = "gemini-2.5-flash"
            print("⚠️ [Brain] GENAI not available. Running in mock mode.")
            return
        
        # Resolve key: IDE env → local .env file (native stdlib only, no dotenv)
        api_key = _load_env_key("GEMINI_API_KEY")
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        else:
            print("⚠️ [Brain] Warning: GEMINI_API_KEY not detected in env or .env file.")
            self.is_mock = True
            self.client = None
            self.model_name = "gemini-2.5-flash"
            return

        try:
            # Initialize the standard client (auto-reads GEMINI_API_KEY from os.environ)
            self.client = genai.Client()
            # Using the recommended model for general text/reasoning tasks
            self.model_name = "gemini-2.5-flash"
        except Exception as e:
            print(f"⚠️ [Brain] Client initialization failed: {e}. Running in mock mode.")
            self.is_mock = True
            self.client = None
            self.model_name = "gemini-2.5-flash"

    def generate_features(self, dataset_summary: str, target_column: str) -> str:
        """
        Generates feature optimization log. If in mock mode, returns static markdown.
        """
        if self.is_mock:
            return self._generate_mock_features(dataset_summary, target_column)
        
        return self.generate_transformation_code(dataset_summary)

    def _generate_mock_features(self, dataset_summary: str, target_column: str) -> str:
        """
        Returns a static markdown string detailing a simulated 3-iteration feature optimization log 
        with a 5-Fold CV score of 0.842.
        """
        mock_log = f"""# Feature Engineering Optimization Report

**Target Column:** {target_column}
**5-Fold Cross-Validation Score:** 0.842

## Optimization Iterations

### Iteration 1: Baseline Feature Analysis
- Applied standard scaling to numeric features
- Encoded categorical features using frequency encoding
- 5-Fold CV Score: 0.812
- Status: ✅ Baseline Established

### Iteration 2: Feature Interaction & Polynomial Expansion
- Created interaction terms between key numeric features
- Applied log transformation to skewed features
- Added polynomial features (degree 2) for non-linear relationships
- 5-Fold CV Score: 0.828
- Status: ✅ Improvement Detected (+0.016)

### Iteration 3: Advanced Feature Engineering
- Implemented target encoding for categorical variables
- Applied recursive feature elimination to reduce dimensionality
- Created domain-specific derived features from base features
- 5-Fold CV Score: 0.842
- Status: ✅ Optimal Performance Achieved (+0.014)

## Summary
- **Total Iterations:** 3
- **Initial Score:** 0.812
- **Final Score:** 0.842
- **Overall Improvement:** +0.030 (+3.7%)
- **Features Engineered:** 24
- **Execution Mode:** Mock (Simulated)
"""
        return mock_log

    def generate_transformation_code(self, profile_json: str) -> str:
        """
        Sends the data profile to Gemini and requests data cleaning/engineering code.
        """
        if self.is_mock:
            return "# Mock transformation: No code generation in mock mode"
        
        system_instruction = (
            "You are a Kaggle Grandmaster AI. Write clean, highly optimized Python code using pandas and numpy "
            "to perform advanced feature engineering and preprocessing.\n\n"
            "CRITICAL RULES:\n"
            "1. Assume the dataframe variable is named `df`.\n"
            "2. Handle missing values strategically (e.g., median for skewed, mode for categorical, flag columns for missingness).\n"
            "3. Implement advanced feature engineering transformations:\n"
            "   - Skewness mitigation: Check for skewness (>0.75 or <-0.75) and apply log/power transformations (e.g. `np.log1p()`).\n"
            "   - Interaction terms: Create promising feature interactions (products, ratios, differences) between numeric columns.\n"
            "   - Scaling: Apply scaling properties (StandardScaler or MinMaxScaler style scaling) to continuous numeric columns.\n"
            "   - Encoding: Encode categoricals strategically (frequency or target-like encoding) instead of basic LabelEncoder mappings.\n"
            "4. AVOID `inplace=True` operations. Use direct assignment: df['col'] = df['col'].fillna(...)\n"
            "5. Do NOT import modules other than `pandas` (as `pd`) and `numpy` (as `np`). Do NOT write file I/O operations.\n"
            "6. Output ONLY valid executable Python code. Do NOT wrap it in markdown block fences or explanations. "
            "Start directly with the code."
        )

        prompt = f"Here is the dataset profile JSON:\n{profile_json}\n\nGenerate the preprocessing code."

        try:
            print(f"🧠 [Brain] Requesting transformation strategy from {self.model_name}...")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                )
            )
            code = response.text.strip()
            # Clean up markdown code block wrapping if the model ignored instructions
            import re
            match = re.search(r"```python\s*(.*?)\s*```", code, re.DOTALL)
            if not match:
                match = re.search(r"```\s*(.*?)\s*```", code, re.DOTALL)
            if match:
                code = match.group(1).strip()
            return code
        except Exception as e:
            print(f"❌ [Brain] API call failed: {e}")
            return ""

if __name__ == "__main__":
    print("--- Testing Brain Module ---")
    
    mock_profile = """
    {
      "dataset_shape": {"rows": 6, "columns": 3},
      "column_metadata": {
        "age": {"dtype": "float64", "missing_count": 1, "skewness": 1.84},
        "income": {"dtype": "float64", "missing_count": 1, "skewness": 0.61},
        "category": {"dtype": "str", "missing_count": 0, "skewness": null}
      }
    }
    """
    
    brain = AIBrain()
    features = brain.generate_features(mock_profile, "target_col")
    print("\n💻 Feature Engineering Output:")
    print(features)