
# 🤖 Autonomous Data Science & Feature Engineering Agent

## _An Adaptive, Self-Healing Multi-Stage AutoML Agent Built for Enterprise Data Pipeline Optimization_

### 📌 Submission Details

-   **Selected Track:** Agents for Business _(Alternative: Freestyle Track)_

-   **Live Application Dashboard:** [https://autonomous-ds-agent--adityaroyiitm.replit.app](https://autonomous-ds-agent--adityaroyiitm.replit.app)

-   **Open-Source Repository:** [https://github.com/25f3000007-code/autonomous-ds-agent](https://github.com/25f3000007-code/autonomous-ds-agent)

-   **Demonstration Video:** [YouTube Walkthrough Link](https://youtube.com/watch?v=your_video_id "null")


## 1. Executive Summary

In enterprise machine learning, data scientists spend up to $80\%$ of their timeline on manual, iterative, and error-prone feature engineering. This represents a massive operational bottleneck that delays model deployment. The **Autonomous Data Science Agent** is an automated, closed-loop machine learning assistant that accepts raw, uncleaned, or adversarial datasets and autonomously engineers features, runs security-hardened transformations, evaluates performance, and dynamically pivots its ML strategy when hitting performance plateaus.

Operating as an active state machine, the agent integrates **Google Gemini 2.5 Flash** (via the modern `google-genai` SDK) to act as a reasoning brain, running within a strictly isolated, crash-resilient sandbox. Across validation sets, the agent demonstrated true autonomous intelligence: automatically pivoting from baseline models to an `ExtraTreesRegressor` pipeline to achieve a $73.8\%$ **reduction in prediction error** without any human intervention.

## 2. The Core Problem: The Feature Engineering Bottleneck

Before any machine learning model can extract signal from a dataset, the raw data must undergo extensive preparation. This includes:

1.  **Structural Cleansing:** Imputing missing values, handling extreme outliers, and verifying structural types.

2.  **Feature Generation:** Performing mathematical transformations, interaction terms, scaling, and encodings that match the underlying algorithms.

3.  **Continuous Validation:** Ensuring that engineered features actually yield generalized predictive power rather than inducing target leakage or overfitting.


Traditionally, this is a human-in-the-loop process of trial-and-error. Data scientists manually run transformations, evaluate scores, rewrite code, and repeat. When dealing with high-dimensional enterprise data, the permutation space is too vast for human manual coverage. Furthermore, manually executing arbitrary, dynamically generated transformation strings in production presents major system vulnerabilities and unhandled runtime crash risks.

## 3. System Architecture & The Closed-Loop Pipeline

The agent solves this problem by executing a highly structured, self-correcting **Monitor → Brain → Executor → Validator** runtime loop:

```
 ┌──────────────┐    profile JSON    ┌─────────────┐
 │  DataMonitor │ ─────────────────▶ │   AIBrain   │  (Gemini 2.5 Flash via
 │  (profiler)  │                    │ (reasoner)  │   Google AI Studio)
 └──────────────┘                    └──────┬──────┘
        ▲                                   │ Python code string
        │ current df                        ▼
 ┌──────┴───────┐    new_df          ┌─────────────────┐
 │  Validator   │ ◀───────────────── │  CodeExecutor   │
 │ (CV scoring) │                    │ (sandboxed run) │
 └──────────────┘                    └─────────────────┘
        │
        ▼
   Decision Matrix (Accept / Reject / Pivot / Stop)

```

### Module Breakdown:

-   **`DataMonitor` (The Senses):** Profiles the physical state of the active dataset. It extracts metadata including schema structures, missing value densities, column correlations, data distributions, and extreme outlier flags, compiling this into a structured JSON payload.

-   **`AIBrain` (The Reasoner):** Leverages **Gemini 2.5 Flash** to digest the JSON profile, analyze historical iterations, formulate statistical hypotheses, and write isolated, targeted Python data-transformation snippets.

-   **`CodeExecutor` (The Sandbox):** Compiles and runs the generated Python strings inside an isolated runtime container, enforcing strict safety limits and protecting the system.

-   **`ModelValidator` (The Judge):** Executes $k$-fold cross-validation on the modified dataset using robust metrics. It scores performance using Root Mean Squared Error (RMSE) for regression tasks:



    $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

## 4. The Core AI Reasoning Layer: Gemini 2.5 Flash

The agent's decision-making is powered by Gemini 2.5 Flash via Google AI Studio. Rather than sending raw, potentially massive CSV datasets to the LLM—which would exhaust token contexts and violate enterprise data privacy boundaries—the agent works exclusively with **metadata summaries**.

The prompt context is dynamically rebuilt on every single iteration to include:

1.  The structural metadata JSON generated by the `DataMonitor`.

2.  The exact chronological audit log of previous iterations (showing what code was written, whether it was accepted, and what score it yielded).

3.  The current best baseline score benchmark.


This creates an active feedback loop. If an AI-generated code snippet fails to improve the metric or throws an execution exception, the exact traceback or performance degradation is fed back to Gemini on the next turn. The model acts as an engineering debugger, analyzing why its previous code was rejected and writing a revised hypothesis.

## 5. Advanced Agent Skill: The Autonomous Pivot State Machine

Most traditional AutoML pipelines apply brute-force grid searches over static operations. The Autonomous DS Agent instead implements an intelligent **Autonomous Pivot State Machine** that dynamically modifies its modeling strategy based on learning progress:

```
             ┌──────────────────────────────────────────────┐
             ▼                                              │
┌────────────────────────┐  3 Rejections  ┌────────────────────────┐  3 Rejections  ┌────────────────────────┐
│        PHASE 1         │ ─────────────▶ │        PHASE 2         │ ─────────────▶ │        PHASE 3         │
│  HistGradientBoosting  │                │  ExtraTreesRegressor   │                │ Hyperparameter Tuning  │
└────────────────────────┘                └────────────────────────┘                └────────────────────────┘
                                                                                                │
                                                                                                │ All strategies
                                                                                                ▼ exhausted
                                                                                            ┌───────────┐
                                                                                            │ STOP &    │
                                                                                            │ EXPORT    │
                                                                                            └───────────┘

```

### Strategic States Explained:

1.  **Phase 1: Feature Engineering Baseline:** The agent uses an ultra-fast `HistGradientBoostingRegressor` to baseline the dataset and runs data-cleaning, scaling, and feature creation cycles.

2.  **Phase 2: Algorithmic Pivoting:** If the agent experiences **three consecutive rejections** (meaning data transformations have plateaued), the engine locks the current optimized state of the dataset and **pivots**. It dynamically swaps the underlying machine learning algorithm to an `ExtraTreesRegressor` (bagging ensemble) to capture different structural patterns.

3.  **Phase 3: Hyperparameter Optimization (HPO):** If performance plateaus again under the new model, the agent triggers an automated tuning layer, running randomized parameter space sweeps across pre-configured distributions to squeeze out remaining gains.

4.  **Early Stopping:** If all strategies are mathematically exhausted, the agent stops run operations immediately, avoiding unnecessary API calls or loop cycles.


### Real-World Validation Success:

When evaluated on a messy target regression housing dataset, the initial feature transformations yielded a solid baseline score. However, upon plateauing, the agent executed an automatic pivot to `ExtraTreesRegressor`, which immediately caused the RMSE error to plummet, culminating in a $73.8\%$ **reduction in total model error**:



$$\Delta_{Error} = \frac{Score_{Baseline} - Score_{Final}}{Score_{Baseline}} \times 100\% = \frac{121879.51 - 31927.96}{121879.51} \times 100\% \approx 73.8\%$$

## 6. Security, Sandboxing & Stress-Test Compliance

Executing dynamically generated LLM code strings presents extreme security risks (arbitrary system execution, data deletion, shell escapes). To ensure enterprise readiness, the agent is protected by the **Antigravity Sandbox Protocol** and underwent a rigorous 19-test penetration audit on June 21, 2026, achieving a $94.7\%$ **pass rate** with all critical and high-severity vulnerabilities resolved.

### Technical Containment Implementations:

-   **Import Whitelisting:** Before any code enters the compilation phase, a tokenizer scans the code structure. Unapproved module requests (such as `import os`, `subprocess`, `requests`, or `shutil`) are intercepted and blocked immediately.

-   **Overriding System Built-ins:** To prevent malicious write operations or workspace access, the execution namespace environment replaces native system hooks with a hand-curated safe whitelist. Critically, `open()` is absent — file I/O is structurally impossible inside the sandbox:

    ```python
    safe_builtins = {
        '__import__': restricted_import,  # Only pandas/numpy allowed
        'abs': abs, 'bool': bool, 'dict': dict, 'float': float,
        'int': int, 'len': len, 'list': list, 'max': max, 'min': min,
        'range': range, 'round': round, 'str': str, 'sum': sum,
        # open, eval, exec, compile intentionally absent
    }
    exec(compiled_code, {"__builtins__": safe_builtins, "pd": pd, "np": np})
    ```

-   **Resource Isolation & Exception Catching:** Loops are monitored for computation bounds. Deep recursion calls are isolated before stack overflow limits are hit. Any unhandled processing exceptions (e.g., mismatched vector shapes) are safely caught within the executor, reverting the DataFrame to its last known healthy state.

-   **Target Leakage Guard:** The validator structurally isolates the target vector. Any AI code attempting to scale, modify, or impute the target values (which would result in fake, artificially low RMSE scores) is overwritten and restored to its true historical state before evaluation.

-   **Fail-Safe Audit Logging:** A complete chronological `*_audit_trail.md` is generated at the conclusion of every successful run, giving administrators complete decision-level visibility into every accepted, rejected, pivoted, and early-stopped iteration.


## 7. Results & Key Deliverables

Upon completing its run, the agent writes two crucial artifacts directly to the workspace directory:

### 1. `*_optimized.csv`

The finalized, high-fidelity dataset. Every single missing value has been intelligently imputed, outliers managed, and feature relationships built and validated. It is fully clean and ready for direct production deployment.

### 2. `*_audit_trail.md`

A complete, step-by-step regulatory decision log detailing the chronological life of the pipeline:

**Iteration** | **Status** | **Metric Score (RMSE)** | **Change** | **Operational Decision**
---|---|---|---|---
0 | Baseline | 121,879.51 | — | Original dataset baseline
1 | Rejected ❌ | 121,879.51 | 0% | No improvement over baseline
2 | Rejected ❌ | 121,879.51 | 0% | Reverted to best state
3 | **Pivot 🔄** | 121,879.51 | Pivot | Plateau detected. Swapping to `ExtraTrees`
4 | **Approved ✅** | 31,927.96 | **−73.8%** | ExtraTrees unlocked massive gain
5 | Rejected ❌ | 31,927.96 | 0% | Reverted to best state
10 | **Early Stop 🛑** | 31,927.96 | Locked | Strategies exhausted. Stopping execution.

## 8. Conclusion & Setup Guide for Judges

The Autonomous Data Science Agent demonstrates a highly robust, secure, and production-ready solution to automated feature engineering. By combining the natural language understanding of Google Gemini 2.5 Flash with strict security boundaries and an adaptive, multi-model state machine, we have built a system that safely automates data science pipelines.

---

### 🔗 Project Links

| Resource | URL |
|---|---|
| **Live Application Dashboard** | [https://autonomous-ds-agent--adityaroyiitm.replit.app](https://autonomous-ds-agent--adityaroyiitm.replit.app) |
| **Open-Source Repository** | [https://github.com/25f3000007-code/autonomous-ds-agent](https://github.com/25f3000007-code/autonomous-ds-agent) |

---

### 🚀 Instant Preview — No Setup Required

The live application is deployed and running. Visit the dashboard directly in your browser with no account or installation:

**[https://autonomous-ds-agent--adityaroyiitm.replit.app](https://autonomous-ds-agent--adityaroyiitm.replit.app)**

Upload any CSV, declare your regression or classification target column, and click **Run Autonomous Optimization**. Your own `GEMINI_API_KEY` (free at [aistudio.google.com](https://aistudio.google.com)) is required to trigger AI inference.

---

### Option 1: Run via Replit (Recommended for Code Review)

Import the repository directly into Replit for a fully pre-configured cloud environment — no local Python installation required.

1. Log in to [replit.com](https://replit.com) and click **Create Repl**.
2. Choose **Import from GitHub** and paste:
   ```
   https://github.com/25f3000007-code/autonomous-ds-agent
   ```
3. Open the **Secrets** panel (padlock icon in the left sidebar) and add:
   - **Key:** `GEMINI_API_KEY`
   - **Value:** Your key from [aistudio.google.com](https://aistudio.google.com)
4. Click the green **Run** button. The pre-configured `.replit` file launches the Streamlit dashboard automatically in the built-in browser pane.
5. Upload any CSV dataset and click **🚀 Run Autonomous Optimization**.

---

### Option 2: Run Locally

For full local execution with a standard Python environment:

```bash
# 1. Clone the repository
git clone https://github.com/25f3000007-code/autonomous-ds-agent.git
cd autonomous-ds-agent

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Gemini API key
export GEMINI_API_KEY="your_api_key_here"
# Windows PowerShell: $env:GEMINI_API_KEY="your_api_key_here"

# 5. Launch the application
streamlit run app.py
```

The Streamlit dashboard opens at `http://localhost:8501`. Upload any CSV dataset and run.
