# Coral Inference and Imprinting Toolkit

## Description

This toolkit facilitates running inference and imprinting training on Coral devices, leveraging Google's Edge TPU for fast and efficient image classification with minimal latency. It features the `CoralMachine` class, which allows for image inference using MobileNet models and the dynamic updating of these models with new data through imprinting, a subset of transfer learning. This is particularly useful in real-world applications that require rapid model adaptation to new data.

## Usage

### Preparation

- Ensure you have a TensorFlow Lite model file, ideally quantized for Edge TPU to enhance performance. This model is used for both inference and as the base for imprinting new data.
- Prepare your data:
  - **For inference:** Place your image files and the corresponding TensorFlow Lite model and labels file in a designated directory.
  - **For imprinting:** Create a folder for your new training data. Within this folder, organize your images into subfolders, where each subfolder's name is the label for the images contained within. This structure is used to imprint these labels onto the model.

### Running Inference

1. Place your TensorFlow Lite model (`mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite` or similar) and labels file (`imagenet_labels.txt`) in a directory.
2. Execute `main.py` with the path to an image for inference, e.g., `test_data/grace_hopper.bmp`. The script prints the top 5 inference results.

### Running Imprinting

1. Organize your training images in the prepared folder, following the structure mentioned above.
2. Create a script like `example_usage.py` to initiate the imprinting process. This will train the model with the new data and update the labels file accordingly.
3. After imprinting, run inference again to see the classification results updated with the newly trained model.

This toolkit simplifies applying machine learning models in environments where quick updates with new data are essential, making it an invaluable resource for developers working with Coral devices.
