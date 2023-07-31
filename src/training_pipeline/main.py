from loguru import logger
from datetime import datetime

from train import train_model
from src.config import train_settings

# %%
def main():
    train_binary_model = True
    train_multiclass_model = False

    method = train_settings["ml-method"]

    dateTimeObj = datetime.now()
    timestamp = dateTimeObj.strftime("%d%m%Y_%H%M")

    folder_path = f"src/training_pipeline/trained_models/{method}_HyperparameterTuning_{timestamp}/"

    if train_binary_model:
        logger.info("Start training the binary models...")
        train_model(folder_path, binary_model=True, method=method)

    if train_multiclass_model:
        logger.info("Start training the multiclass models...")
        train_model(folder_path, binary_model=False, method=method)
    
# %%
if __name__ == "__main__":
    
    main()