import streamlit as st
import os
import subprocess
import sys
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
max_iterations = st.sidebar.slider("Optimization Iterations", min_value=1, max_value=10, value=3,
                                    help="More iterations give the AI more attempts to find improvements, but take longer.")

if uploaded_file is not None:
    # Setup paths relative to the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "uploaded_dataset.csv")
    optimized_path = filepath.replace(".csv", "_optimized.csv")
    audit_path = filepath.replace(".csv", "_audit_trail.md")

    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.sidebar.button("🚀 Run Autonomous Optimization"):
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        status_placeholder = status_container.empty()
        status_placeholder.info(f"⏳ Running {max_iterations} iteration(s) — this may take a minute...")

        python_exe = sys.executable

        my_env = os.environ.copy()
        my_env["PYTHONIOENCODING"] = "utf-8"
        my_env["PYTHONPATH"] = current_dir
        my_env["AGENT_TARGET_COLUMN"] = target_column
        my_env["AGENT_MAX_ITERATIONS"] = str(max_iterations)

        try:
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
                status_placeholder.success("✅ Optimization finished!")
                st.session_state["optimized"] = True
            else:
                status_placeholder.error(f"Agent Error: {result.stderr}")
        except Exception as e:
            status_placeholder.error(f"System Error: {e}")

    # --- DOWNLOAD BUTTON ---
    if os.path.exists(optimized_path):
        with open(optimized_path, "rb") as f:
            optimized_bytes = f.read()
        st.sidebar.divider()
        st.sidebar.download_button(
            label="⬇️ Download Optimized Dataset",
            data=optimized_bytes,
            file_name="optimized_dataset.csv",
            mime="text/csv",
        )

    # --- AUDIT TRAIL ---
    if os.path.exists(audit_path):
        with open(audit_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
