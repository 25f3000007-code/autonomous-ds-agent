import pandas as pd
import os
import time
import json
import uuid
import traceback
from datetime import datetime
from src.monitor import DataMonitor
from src.brain import AIBrain
from src.executor import CodeExecutor
from src.validator import ModelValidator

# Model upgrade sequence: HistGBT → ExtraTrees → HPO → STOP
_MODEL_SEQUENCE = ["HistGradientBoosting", "ExtraTrees", "HPO", "STOP"]
_MAX_CONSECUTIVE_REJECTIONS = 3


class AutonomousAgent:
    def __init__(self, filepath, target_column, max_iterations=3):
        self.filepath = filepath
        self.target_column = target_column
        self.max_iterations = max_iterations
        self.monitor = DataMonitor(filepath)
        self.brain = AIBrain()
        self.executor = CodeExecutor()
        self.validator = ModelValidator(target_column)

        self.best_score = None
        self.best_df = None
        self.metric_name = None
        self.is_classification = False
        self.audit_history = []

        # --- Pivot state machine ---
        self.consecutive_rejections = 0
        self.current_model = "HistGradientBoosting"
        self.feature_pipeline_locked = False
        self.stop_flag = False

        # --- Output delivery ---
        self.run_id = uuid.uuid4().hex[:6].upper()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.telemetry_lines = []
        self.execution_successful = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_improvement(self, new_score: float) -> bool:
        if self.is_classification:
            return new_score > self.best_score
        return new_score < self.best_score

    def _model_label(self) -> str:
        labels = {
            "HistGradientBoosting": "HistGBT",
            "ExtraTrees": "ExtraTrees",
            "HPO": "HPO-HistGBT",
        }
        return labels.get(self.current_model, self.current_model)

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _log(self, message: str):
        self.telemetry_lines.append(message)
        print(message)

    def _write_telemetry(self, data_dir: str):
        log_path = os.path.join(data_dir, "trial_execution.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"# Trial Execution Log — Run {self.run_id}\n")
            f.write(f"# Timestamp: {self.timestamp}\n\n")
            for line in self.telemetry_lines:
                f.write(line + "\n")

    def _write_manifest(self, data_dir: str):
        manifest = {
            "last_updated": self.timestamp,
            "run_id": self.run_id,
            "target_column": self.target_column,
            "final_shape": list(self.best_df.shape) if self.best_df is not None else None,
            "best_score": self.best_score,
            "best_performing_artifact": os.path.join(data_dir, f"optimized_deployment_{self.timestamp}_{self.run_id}.csv"),
            "competition_submission": os.path.join(data_dir, "submission.csv"),
            "execution_trace": os.path.join(data_dir, "trial_execution.log"),
        }
        manifest_path = os.path.join(data_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    # ------------------------------------------------------------------
    # Pivot state machine
    # ------------------------------------------------------------------

    def _next_model(self) -> str:
        idx = _MODEL_SEQUENCE.index(self.current_model)
        return _MODEL_SEQUENCE[idx + 1] if idx + 1 < len(_MODEL_SEQUENCE) else "STOP"

    def _handle_plateau(self, iteration: int):
        next_model = self._next_model()

        if next_model == "STOP":
            self._log(f"🛑 [Agent] Feature space exhausted for {self.current_model}. Terminating run.")
            self.audit_history.append({
                'iteration': iteration,
                'status': '🛑 Early Stop',
                'score': self.best_score,
                'delta': 0,
                'note': "All strategies exhausted. Best score locked in.",
            })
            self.stop_flag = True
            return

        self._log(f"🔄 [Agent] Plateau detected after {_MAX_CONSECUTIVE_REJECTIONS} rejections. "
                   f"Locking features and pivoting: {self.current_model} → {next_model}")

        self.feature_pipeline_locked = True
        self.current_model = next_model
        self.consecutive_rejections = 0

        self.audit_history.append({
            'iteration': iteration,
            'status': f'🔄 Pivot → {self._model_label()}',
            'score': self.best_score,
            'delta': 0,
            'note': (f"Feature pipeline locked. Switching model to {self.current_model} "
                     "to attempt further improvement."),
        })

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self):
        self._log("\nSTARTING AUTONOMOUS DATA SCIENCE AGENT")
        self._log(f"📋 Run ID: {self.run_id} | Timestamp: {self.timestamp}")

        if not self.monitor.load_data():
            self._log("❌ [Agent] Data loading failed. Aborting.")
            self._write_outputs()
            return

        current_df = self.monitor.df.copy()
        baseline = self.validator.evaluate(current_df)
        if "error" in baseline:
            self._log(f"❌ [Agent] Baseline evaluation failed: {baseline['error']}. Aborting.")
            self._write_outputs()
            return

        self.best_score = baseline['score']
        self.metric_name = baseline['metric']
        self.best_df = current_df.copy()
        self.is_classification = baseline['task_type'] == "Classification"
        baseline_score = self.best_score

        self._log(f"📊 [Agent] Baseline {self.metric_name}: {self.best_score}")

        for i in range(1, self.max_iterations + 1):
            if self.stop_flag:
                break

            # ----------------------------------------------------------
            # Decide action based on phase
            # ----------------------------------------------------------
            if self.feature_pipeline_locked:
                # PIVOT PHASE: evaluate locked best_df with the new model/HPO
                phase_label = self._model_label()
                self._log(f"🔬 [Agent] Iteration {i}: Evaluating locked features with {phase_label}...")
                eval_res = self.validator.evaluate(self.best_df, model_name=self.current_model)

                if "error" not in eval_res and self._is_improvement(eval_res['score']):
                    prev_score = self.best_score
                    self.best_score = eval_res['score']
                    self.consecutive_rejections = 0
                    self.audit_history.append({
                        'iteration': i,
                        'status': f'✅ {phase_label} Win',
                        'score': self.best_score,
                        'delta': round(self.best_score - prev_score, 4),
                        'note': f"Model {phase_label} improved score.",
                    })
                    self._log(f"✅ [Agent] Iteration {i} [{phase_label}]: {prev_score} → {self.best_score}")
                else:
                    self.consecutive_rejections += 1
                    self.audit_history.append({
                        'iteration': i,
                        'status': f'❌ {phase_label} No gain',
                        'score': eval_res.get('score', self.best_score),
                        'delta': 0,
                        'note': eval_res.get('error', 'No improvement over current best.'),
                    })
                    self._log(f"⚠️  [Agent] Iteration {i} [{phase_label}]: No improvement, "
                                f"consecutive rejections = {self.consecutive_rejections}")

                    if self.consecutive_rejections >= _MAX_CONSECUTIVE_REJECTIONS:
                        self._handle_plateau(i)

            else:
                # FEATURE ENGINEERING PHASE: AI generates transformation code
                self.monitor.df = current_df.copy()
                profile = self.monitor.generate_profile(self.target_column)
                self._log("⏳ Giving Gemini API a brief cooling period...")
                time.sleep(8)

                try:
                    code_string = self.brain.generate_transformation_code(
                        profile, self.target_column,
                        iteration=i, current_score=self.best_score,
                        metric_name=self.metric_name, history=self.audit_history
                    )
                except RuntimeError as e:
                    self._log(f"❌ [Agent] Brain failed after all retries: {e}")
                    self._log("⚠️  [Agent] Skipping this iteration and continuing with current best.")
                    self.consecutive_rejections += 1
                    self.audit_history.append({
                        'iteration': i,
                        'status': '❌ Brain Failure',
                        'score': self.best_score,
                        'delta': 0,
                        'note': f"API unreachable after retries: {e}",
                    })
                    if self.consecutive_rejections >= _MAX_CONSECUTIVE_REJECTIONS:
                        self._handle_plateau(i)
                    continue

                new_df = self.executor.apply_transformation(current_df, code_string)
                # Guard: never let AI corrupt the target column
                new_df[self.target_column] = current_df[self.target_column].values
                eval_res = self.validator.evaluate(new_df)

                if "error" not in eval_res and self._is_improvement(eval_res['score']):
                    prev_score = self.best_score
                    self.best_score = eval_res['score']
                    self.best_df = new_df.copy()
                    current_df = new_df.copy()
                    self.consecutive_rejections = 0
                    self.audit_history.append({
                        'iteration': i,
                        'status': 'Approved ✅',
                        'score': self.best_score,
                        'delta': round(self.best_score - prev_score, 4),
                        'note': '',
                    })
                    self._log(f"✅ [Agent] Iteration {i}: {prev_score} → {self.best_score}")
                else:
                    reason = eval_res.get('error', 'No improvement') if "error" in eval_res else 'No improvement'
                    self.consecutive_rejections += 1
                    self.audit_history.append({
                        'iteration': i,
                        'status': 'Rejected ❌',
                        'score': eval_res.get('score', self.best_score),
                        'delta': 0,
                        'note': '',
                    })
                    self._log(f"⚠️  [Agent] Iteration {i}: Rejected ({reason}), "
                                f"consecutive = {self.consecutive_rejections}")

                    if self.consecutive_rejections >= _MAX_CONSECUTIVE_REJECTIONS:
                        self._handle_plateau(i)

        self._write_outputs()
        self._generate_audit_report(baseline_score)

    # ------------------------------------------------------------------
    # Output delivery
    # ------------------------------------------------------------------

    def _write_outputs(self):
        """Write all production artifacts to disk."""
        data_dir = os.path.dirname(self.filepath)
        os.makedirs(data_dir, exist_ok=True)

        # 1. Standard optimized dataset
        self.best_df.to_csv(self.filepath.replace(".csv", "_optimized.csv"), index=False)
        self.best_df.to_csv(self.filepath, index=False)
        self._log(f"🎉 Fully optimized dataset written to disk at: {self.filepath}")

        # 2. Atomic experiment artifact (timestamped, unique)
        artifact_path = os.path.join(data_dir, f"optimized_deployment_{self.timestamp}_{self.run_id}.csv")
        self.best_df.to_csv(artifact_path, index=False)
        self._log(f"📦 Atomic artifact saved: {artifact_path}")

        # 3. Competition submission copy
        submission_path = os.path.join(data_dir, "submission.csv")
        self.best_df.to_csv(submission_path, index=False)
        self._log(f"🏆 Competition submission copied: {submission_path}")

        # 4. Telemetry log
        self._write_telemetry(data_dir)
        self._log(f"📝 Telemetry log saved: {os.path.join(data_dir, 'trial_execution.log')}")

        # 5. Metadata manifest
        self._write_manifest(data_dir)
        self._log(f"📄 Manifest saved: {os.path.join(data_dir, 'manifest.json')}")

    # ------------------------------------------------------------------
    # Audit report
    # ------------------------------------------------------------------

    def _generate_audit_report(self, baseline_score: float):
        report_path = self.filepath.replace(".csv", "_audit_trail.md")
        total_delta = round(self.best_score - baseline_score, 4)
        approved = sum(1 for h in self.audit_history if '✅' in h['status'] or 'Approved' in h['status'])
        rejected = sum(1 for h in self.audit_history if '❌' in h['status'] or 'Rejected' in h['status'])
        pivots   = sum(1 for h in self.audit_history if 'Pivot' in h['status'])

        if self.is_classification:
            improved = total_delta > 0
            pct_change = abs(total_delta / baseline_score * 100) if baseline_score else 0
            improvement_label = f"📈 +{pct_change:.1f}% better" if improved else f"📉 {pct_change:.1f}% worse"
            delta_note = "Higher is better (F1-Score)"
        else:
            improved = total_delta < 0
            pct_change = abs(total_delta / baseline_score * 100) if baseline_score else 0
            improvement_label = f"📈 {pct_change:.1f}% less error" if improved else f"📉 {pct_change:.1f}% more error"
            delta_note = "Lower is better (RMSE — negative delta = improvement)"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 🤖 Autonomous DS Agent — Audit Trail\n\n")
            f.write(f"**Metric:** {self.metric_name}\n\n")
            f.write("---\n\n")
            f.write("## 📊 Score Summary\n\n")
            f.write(f"| | Score |\n|---|---|\n")
            f.write(f"| **Baseline** | `{baseline_score}` |\n")
            f.write(f"| **Final (Best)** | `{self.best_score}` |\n")
            f.write(f"| **Change** | {improvement_label} |\n\n")
            f.write(f"> ℹ️ *{delta_note}*\n\n")
            f.write("---\n\n")
            f.write("## 🔄 Iteration Log\n\n")
            f.write("| Iteration | Status | Score | Change | Note |\n")
            f.write("|---|---|---|---|---|\n")
            for h in self.audit_history:
                if h['delta'] != 0:
                    iter_pct = abs(h['delta'] / baseline_score * 100) if baseline_score else 0
                    if self.is_classification:
                        delta_str = f"📈 +{iter_pct:.1f}%" if h['delta'] > 0 else f"📉 -{iter_pct:.1f}%"
                    else:
                        delta_str = f"📈 {iter_pct:.1f}% less error" if h['delta'] < 0 else f"📉 {iter_pct:.1f}% more error"
                else:
                    delta_str = "—"
                note = h.get('note', '')
                f.write(f"| {h['iteration']} | {h['status']} | `{h['score']}` | {delta_str} | {note} |\n")
            f.write("\n---\n\n")
            f.write("## 📝 Summary\n\n")
            f.write(f"- **Iterations run:** {len(self.audit_history)}\n")
            f.write(f"- **Approved (feature wins):** {approved}\n")
            f.write(f"- **Rejected:** {rejected}\n")
            f.write(f"- **Pivots triggered:** {pivots}\n")
            f.write(f"- **Result:** {improvement_label}\n")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root, "data", "uploaded_dataset.csv")
    target_col = os.environ.get("AGENT_TARGET_COLUMN", "price")
    max_iter = int(os.environ.get("AGENT_MAX_ITERATIONS", "3"))
    agent = AutonomousAgent(data_path, target_col, max_iterations=max_iter)
    agent.run()
