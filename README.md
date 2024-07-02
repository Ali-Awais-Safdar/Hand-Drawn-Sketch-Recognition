# Hand-Drawn-Sketch-Recognition

## Project Overview
This project focuses on recognizing hand-drawn sketches using deep learning techniques. The model leverages the VGG19 architecture pre-trained on ImageNet for feature extraction, followed by a custom dense network for classification. The project includes data preprocessing, training, and validation processes, as well as generating predictions on a test dataset.

## Dependencies
- Python 3.x
- TensorFlow
- Keras
- NumPy
- pandas
- OpenCV
- Matplotlib

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Hand-Drawn-Sketch-Recognition.git
cd Hand-Drawn-Sketch-Recognition
2. Install Required Packages
bash
Copy code
pip install -r requirements.txt
3. Prepare the Dataset
Place your dataset in the following directory structure:

lua
Copy code
Hand-Drawn-Sketch-Recognition/
|-- Train/
|   |-- class1/
|   |   |-- img1.png
|   |   |-- img2.png
|   |-- class2/
|       |-- img1.png
|       |-- img2.png
|-- Validation/
|   |-- class1/
|   |   |-- img1.png
|   |   |-- img2.png
|   |-- class2/
|       |-- img1.png
|       |-- img2.png
|-- Test/
|   |-- img1.png
|   |-- img2.png
|-- Train.csv
|-- Validation.csv
|-- modified_labels.csv
4. Train the Model
Run the following script to train the model:

bash
Copy code
python train.py
5. Generate Predictions
After training, you can generate predictions on the test dataset:

bash
Copy code
python predict.py
Training and Validation
The model is trained using an ImageDataGenerator for data augmentation and preprocessing. The training process includes:

Data Augmentation:

python
Copy code
train_datagen = ImageDataGenerator(
    rescale=1./255,
)
Data Generator:

python
Copy code
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='Train',
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
Model Architecture:

python
Copy code
model = Sequential([
    vgg_model1,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dense(250, activation='softmax')
])
Model Compilation:

python
Copy code
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
Model Training:

python
Copy code
history = model.fit_generator(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    verbose=1
)
Evaluation
Training and validation loss and accuracy are plotted to evaluate the model performance.

python
Copy code
import matplotlib.pyplot as plt

# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
Prediction
A CSV file is created with the filenames of the test set, and predictions are generated and stored in another CSV file.

python
Copy code
import os
import csv

# Define the folder path
test_dir = "Test"

# Get all filenames in the folder
filenames = os.listdir(test_dir)

# Create a CSV file and write filenames to it
csv_file = "testset.csv"
with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Filename"])
    writer.writerows([[filename] for filename in filenames])

print(f"CSV file '{csv_file}' created successfully.")
License
This project is licensed under the MIT License.
