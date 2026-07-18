from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
THETA_PATH = DATA_DIR / "model.json"
MAIN = PROJECT_ROOT / "main.py"

USAGE = """Usage:
    <command> [arguments]

Commands:
  train [csv_file]      Train the linear regression model.
  graph [csv_file]      Plot the dataset and regression line.
  predict [mileage]     Predict the price for a mileage.
  reset                 Reset the trained model.
  help                  Show this help message.
"""
