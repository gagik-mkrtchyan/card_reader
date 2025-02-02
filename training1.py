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
data_yaml_path = "s3://fastbank-ml-models-archive/OCR/Data/credit_cards.zip"
model.train(data=data_yaml_path,
            epochs=200,
            batch=4,
            name="digis_letters",
            nbs=2,
            cos_lr=True,
            close_mosaic=10,
            imgsz=[800,320])
wandb.finish()

model.export(format='torchscript', imgsz=[320,800])
