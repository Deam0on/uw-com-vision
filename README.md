README is under development and might be outdated as well as ahead of the actual functionality. Read with caution.

Overview
UW-Com-Vision is a project designed to perform computer vision tasks, including data preparation, model training, evaluation, and inference. This branch (multi_set_V1) focuses on handling multiple datasets and provides a flexible framework for experimenting with different model architectures and configurations.

Features
Data Preparation: Load and preprocess datasets for training and evaluation.
Model Training: Train a deep learning model for vision tasks.
Model Evaluation: Evaluate model performance using various metrics.
Inference: Make predictions on new data and visualize results.
Utilities: Helper functions for logging, metrics, and more.
Installation
To use this repository, follow these steps:

Clone the Repository
```
git clone -b multi_set_V1 https://github.com/Deam0on/uw-com-vision.git
cd uw-com-vision
```

Install Dependencies
Make sure you have Python 3.8+ and the required packages installed. You can use pip to install the dependencies listed in the requirements.txt file:

```
pip install -r requirements.txt
```

Alternatively, if a requirements.txt file is not provided, manually install packages like torch, torchvision, opencv-python, numpy, pandas, matplotlib, and shapely.

Usage
Data Preparation
Prepare your datasets using the data_preparation.py script. This script will load, process, and split the data into training and validation sets.

```
python data_preparation.py --config configs/data_config.yaml
```

Model Training
Train the model using the train_model.py script. Configure training parameters in the configs/train_config.yaml file.

```
python train_model.py --config configs/train_config.yaml
```

Model Evaluation
Evaluate the trained model using the evaluate_model.py script to measure its performance on a test dataset.

```
python evaluate_model.py --config configs/eval_config.yaml
```

Inference
Make predictions on new data using the inference.py script.

```
python inference.py --config configs/inference_config.yaml --input /path/to/input/image
```

Configuration
Configuration files are located in the configs/ directory. These YAML files define parameters for data preparation, model training, evaluation, and inference. Adjust these files according to your specific needs.

Contributing
If you wish to contribute to this project, please fork the repository and create a new branch for your features or bug fixes. Submit a pull request for review.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For questions or suggestions, please open an issue or contact the repository owner.
