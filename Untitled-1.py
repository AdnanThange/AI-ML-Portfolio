import yaml
print(yaml.__version__)

# ===========================
# Project Configuration
# ===========================
project:
  name: house_price_prediction
  description: Linear Regression pipeline for house price prediction
  base_dir: "C:/Users/Adnan/OneDrive/Desktop/git_hub"

# ===========================
# Paths
# ===========================
paths:
  data: "C:/Users/Adnan/OneDrive/Desktop/DL_PROJECTS/house_price_prediction_dataset.csv"
  logs: "logs/pipeline.log"
  models_dir: "models"
  evaluation_dir: "evaluation"
  metrics_file: "evaluation/metrics.json"

# ===========================
# Logging Configuration
# ===========================
logging:
  level: DEBUG
  format: "%(asctime)s:%(levelname)s:%(message)s"
