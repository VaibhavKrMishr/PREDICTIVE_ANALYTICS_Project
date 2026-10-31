# models.py - backward-compatible wrapper redirecting to main.py orchestrator

from main import run_pipeline, SEED

# alias run_all_models to run_pipeline for backward compatibility
run_all_models = run_pipeline

if __name__ == "__main__":
    res = run_pipeline()
    print("models.py: execution completed via main.py")
