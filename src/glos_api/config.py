import yaml

with open("src/glos_api/config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)