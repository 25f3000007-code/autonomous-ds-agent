import pandas as pd
import os
import time
from src.monitor import DataMonitor
from src.brain import AIBrain
from src.executor import CodeExecutor
from src.validator import ModelValidator

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

    def _is_improvement(self, new_score: float) -> bool:
        """Return True if new_score is genuinely better than the current best."""
        if self.is_classification:
            return new_score > self.best_score   # Higher F1 is better
        else:
            return new_score < self.best_score   # Lower RMSE is better

    def run(self):
        print("\nSTARTING AUTONOMOUS DATA SCIENCE AGENT")
        if not self.monitor.load_data(): return
        
        current_df = self.monitor.df.copy()
        baseline = self.validator.evaluate(current_df)
        if "error" in baseline:
            print(f"❌ [Agent] Baseline evaluation failed: {baseline['error']}. Aborting.")
            return
        self.best_score = baseline['score']
        self.metric_name = baseline['metric']
        self.best_df = current_df.copy()
        self.is_classification = baseline['task_type'] == "Classification"
        baseline_score = self.best_score

        print(f"📊 [Agent] Baseline {self.metric_name}: {self.best_score}")

        for i in range(1, self.max_iterations + 1):
            profile = self.monitor.generate_profile()
            code_string = self.brain.generate_transformation_code(profile)
            new_df = self.executor.apply_transformation(current_df, code_string)
            eval_res = self.validator.evaluate(new_df)
            
            if "error" not in eval_res and self._is_improvement(eval_res['score']):
                prev_score = self.best_score
                self.best_score = eval_res['score']
                self.best_df = new_df.copy()   # save the best dataframe
                current_df = new_df.copy()
                self.audit_history.append({
                    'iteration': i,
                    'status': 'Approved ✅',
                    'score': self.best_score,
                    'delta': round(self.best_score - prev_score, 4)
                })
                print(f"✅ [Agent] Iteration {i}: Improvement {prev_score} → {self.best_score}")
            else:
                reason = eval_res.get('error', 'No improvement') if "error" in eval_res else 'No improvement'
                self.audit_history.append({
                    'iteration': i,
                    'status': 'Rejected ❌',
                    'score': eval_res.get('score', self.best_score),
                    'delta': 0
                })
                print(f"⚠️  [Agent] Iteration {i}: Rejected ({reason}), best stays at {self.best_score}")
        
        self.best_df.to_csv(self.filepath.replace(".csv", "_optimized.csv"), index=False)
        self._generate_audit_report(baseline_score)

    def _generate_audit_report(self, baseline_score: float):
        report_path = self.filepath.replace(".csv", "_audit_trail.md")
        total_delta = round(self.best_score - baseline_score, 4)
        approved = sum(1 for h in self.audit_history if 'Approved' in h['status'])
        rejected = sum(1 for h in self.audit_history if 'Rejected' in h['status'])

        # For display: positive pct always means "got better"
        if self.is_classification:
            # Higher F1 is better — improvement is positive delta
            improved = total_delta > 0
            pct_change = abs(total_delta / baseline_score * 100) if baseline_score else 0
            improvement_label = f"📈 +{pct_change:.1f}% better" if improved else f"📉 {pct_change:.1f}% worse"
            delta_note = "Higher is better (F1-Score)"
        else:
            # Lower RMSE is better — improvement is negative delta
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
            f.write("| Iteration | Status | Score | Change |\n")
            f.write("|---|---|---|---|\n")
            for h in self.audit_history:
                if h['delta'] != 0:
                    iter_pct = abs(h['delta'] / baseline_score * 100) if baseline_score else 0
                    if self.is_classification:
                        delta_str = f"📈 +{iter_pct:.1f}%" if h['delta'] > 0 else f"📉 -{iter_pct:.1f}%"
                    else:
                        delta_str = f"📈 {iter_pct:.1f}% less error" if h['delta'] < 0 else f"📉 {iter_pct:.1f}% more error"
                else:
                    delta_str = "—"
                f.write(f"| {h['iteration']} | {h['status']} | `{h['score']}` | {delta_str} |\n")
            f.write("\n---\n\n")
            f.write("## 📝 Summary\n\n")
            f.write(f"- **Iterations run:** {len(self.audit_history)}\n")
            f.write(f"- **Approved:** {approved}\n")
            f.write(f"- **Rejected:** {rejected}\n")
            f.write(f"- **Result:** {improvement_label}\n")

if __name__ == "__main__":
    # Use absolute path relative to project root
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root, "data", "uploaded_dataset.csv")
    agent = AutonomousAgent(data_path, "price")
    agent.run()