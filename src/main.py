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

    def run(self):
        print("\nSTARTING AUTONOMOUS DATA SCIENCE AGENT")
        if not self.monitor.load_data(): return
        
        current_df = self.monitor.df.copy()
        baseline = self.validator.evaluate(current_df)
        self.best_score = baseline['score']
        self.metric_name = baseline['metric']
        self.best_df = current_df.copy()
        self.is_classification = baseline['task_type'] == "Classification"

        for i in range(1, self.max_iterations + 1):
            profile = self.monitor.generate_profile()
            code_string = self.brain.generate_transformation_code(profile)
            new_df = self.executor.apply_transformation(current_df, code_string)
            eval_res = self.validator.evaluate(new_df)
            
            if "error" not in eval_res and eval_res['score'] != self.best_score:
                self.best_score = eval_res['score']
                current_df = new_df.copy()
                self.audit_history.append({'iteration': i, 'status': 'Approved', 'metric_score': self.best_score})
            else:
                self.audit_history.append({'iteration': i, 'status': 'Rejected', 'metric_score': self.best_score})
        
        self.best_df.to_csv(self.filepath.replace(".csv", "_optimized.csv"), index=False)
        self._generate_audit_report()

    def _generate_audit_report(self):
        report_path = self.filepath.replace(".csv", "_audit_trail.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Audit Trail\n\nFinal Score: " + str(self.best_score))

if __name__ == "__main__":
    # Use absolute path relative to project root
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root, "data", "uploaded_dataset.csv")
    agent = AutonomousAgent(data_path, "price")
    agent.run()