# Card Reader Object Detection and Information Extraction
 
This repository contains code for a pipeline that performs object detection on a card (such as a credit card), extracts information (such as the cardholder's name, card number, and expiration date), and then processes this information through two different machine learning models. The pipeline is implemented in both C++ and Python.
 
## Files in the Repository
 
### 1. `pipeline.cpp`
This C++ file contains a pipeline for card reading and object detection. The steps are as follows:
- **Object Detection**: Detects specific objects such as card information on the card.
- **Concatenation**: Once the object is detected, the relevant data is concatenated.
- **Further Processing**: The concatenated data is passed to a second network that extracts the following information:
  - Cardholder's Name
  - Card Number
  - Expiration Date
 
### 2. `pipeline.py`
This is the Python equivalent of `pipeline.cpp`. It performs the same operations as the C++ pipeline:
- **Object Detection**: Detects objects on the card.
- **Concatenation**: Concatenates detected information.
- **Further Processing**: Feeds the data into the second network to extract the cardholder's name, card number, and expiration date.
 
### 3. `training1.py`
This script is responsible for training the first model in the pipeline. The first model is used for object detection on the card. It is designed to:
- Detect various objects on the card, such as the card number, cardholder's name, and expiration date.
- Perform feature extraction for downstream tasks.
 
### 4. `training2.py`
This script is used for training the second model in the pipeline. The second model focuses on:
- Extracting specific information such as the cardholder's name, card number, and expiration date from the detected objects.
- Post-processing the information obtained from the first model.
 
## How to Run

Finally, run the pipeline.py script to do inference for one image:

python pipeline.py

