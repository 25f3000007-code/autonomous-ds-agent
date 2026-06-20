import streamlit as st
import os
import subprocess
import sys
# 1. Update the import (this will now work with the src folder)
from src.main import AutonomousAgent

# Page Config
st.set_page_config(page_title="Autonomous DS Agent", page_icon="🤖", layout="wide")
st.title("Autonomous Data Science & Feature Engineering Agent")

# --- INITIALIZE CONTAINERS AT THE TOP LEVEL ---
status_container = st.container()
report_container = st.container()

# Sidebar
uploaded_file = st.sidebar.file_uploader("Upload Messy Dataset (CSV)", type=["csv"])
target_column = st.sidebar.text_input("Target Column Name", value="price")

if uploaded_file is not None:
    # Setup paths relative to the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "uploaded_dataset.csv")
    
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.sidebar.button("🚀 Run Autonomous Optimization"):
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        status_placeholder = status_container.empty()
        status_placeholder.info("Running optimization...")

        # Get absolute path to the project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
    
        # Define paths for module-based execution
        python_exe = os.path.join(current_dir, "venv", "Scripts", "python.exe")
    
        # Set the environment
        my_env = os.environ.copy()
        my_env["PYTHONIOENCODING"] = "utf-8"
        my_env["PYTHONPATH"] = current_dir # Add root to PYTHONPATH so 'src' is found

        try:
            # Execute as a module from the root
            result = subprocess.run(
                [python_exe, "-m", "src.main"],
                cwd=current_dir,
                env=my_env,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
        
            if result.returncode == 0:
                status_placeholder.success("Optimization finished!")
                st.session_state["optimized"] = True
            else:
                status_placeholder.error(f"Agent Error: {result.stderr}")
        except Exception as e:
            status_placeholder.error(f"System Error: {e}")

    # Render results
    audit_path = filepath.replace(".csv", "_audit_trail.md")
    if os.path.exists(audit_path):
        with open(audit_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())