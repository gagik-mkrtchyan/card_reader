import os
from torch.cuda import device
import wandb
from wandb.integration.ultralytics import add_wandb_callback

wandb.init(project='second_model', job_type='training')

from ultralytics import YOLO

# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt") # load a pretrained model (recommended for training)

add_wandb_callback(model, enable_model_checkpointing=False)
# Use the model
data_yaml_path = "s3://fastbank-ml-models-archive/OCR/Data/digits800,320.zip"
model.train(data=data_yaml_path,
            epochs=120,
            batch=64,
            name="credit_card",
            nbs=32,
            cos_lr=True,
            close_mosaic=20)
wandb.finish()

model.export(format='torchscript')
