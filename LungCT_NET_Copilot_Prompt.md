# Task: Create a Parameterized Proposed Deep Learning Model Notebook for Lung Cancer Classification

Act as a Senior Deep Learning Research Engineer and Python/TensorFlow Expert. I am working on a Lung Cancer Classification research project using Google Colab. I want to modify the Swin Transformer architecture, ResNet, MovileNetV2 and experiment with different configurations to develop my own proposed model.

I am still learning Deep Learning, so I need clean, beginner-friendly, well-commented, and modular code.

Your task is to create a complete, executable **Google Colab Jupyter Notebook (.ipynb)** for my experiments.

## 1. Main Objective

I want to experiment with a proposed model based on Swin Transformer, ResNet, MobileNet, EfficientNet and investigate whether using Categorical Cross-Entropy or other configurable loss functions can improve classification accuracy and other evaluation metrics.

Important:
- Do not assume that adding Cross-Entropy will automatically improve accuracy.
- Help me design a proper experimental setup to compare different configurations.
- Keep the architecture flexible so I can modify its components and parameters later.
- Avoid unnecessarily complicated code.

## 2. Centralized Configuration Section

At the beginning of the notebook, create a dedicated configuration cell where I can easily change all important parameters without modifying the main implementation.

### Model Selection

```python
MODEL_NAMES = [
    "ResNet50",
    "EfficientNetB0",
    "MobileNetV2",
    "DenseNet121",
    "SwinTransformer"
]

# Example:
# MODEL_NAMES = ["SwinTransformer"]
# MODEL_NAMES = ["ResNet50", "SwinTransformer"]
```

I should be able to train one or multiple models using the same configuration.

### Training Parameters

Keep all parameters in one place.

```python
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
SEED = 42

EARLY_STOPPING_PATIENCE = 7
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-7
```

Add beginner-friendly comments with suggested alternative values.

For example:

```python
LEARNING_RATE = 1e-4
# Suggested values to experiment with:
# 1e-3, 5e-4, 1e-4, 5e-5, 1e-5
```

### Loss Function Configuration

Create a separate configuration section for loss functions.

I want to experiment with:

- Categorical Cross-Entropy
- Sparse Categorical Cross-Entropy
- Categorical Cross-Entropy with Label Smoothing
- Other relevant loss functions if appropriate for this classification problem.

Example:

```python
LOSS_FUNCTION_NAME = "categorical_crossentropy"

LABEL_SMOOTHING = 0.0
# Suggested values: 0.0, 0.05, 0.1, 0.15
```

Keep loss-specific parameters separate and explain what each parameter does.

Ensure the loss function is compatible with the dataset labels and model output format.

### Proposed Swin Transformer Configuration

Create a separate section for Swin Transformer architecture parameters.

For example:

```python
SWIN_CONFIG = {
    "patch_size": 4,
    "window_size": 7,
    "embed_dim": 96,
    "num_heads": [3, 6, 12, 24],
    "depths": [2, 2, 6, 2],
    "mlp_ratio": 4.0,
    "dropout_rate": 0.1,
    "attention_dropout_rate": 0.0,
    "drop_path_rate": 0.1
}
```

These values should be configurable, with simple explanations and suggested alternatives.

Important:
- Do not use a nonexistent API such as `keras_hub.models.SwinTransformerImageClassifier`.
- First verify which Swin Transformer implementation is actually supported by the selected library and its version.
- If necessary, implement a lightweight modular Swin Transformer architecture using TensorFlow/Keras components.
- Clearly distinguish between the original Swin Transformer and my modified/proposed version.
- Keep the architecture easy to modify, especially the attention mechanism, feature extraction, classification head, and loss function.

## 3. Google Colab Environment Setup

The notebook must be designed specifically for Google Colab.

Include:

1. Google Drive mounting.
2. Dataset path configuration.
3. Required library installation.
4. TensorFlow and GPU verification.
5. Random seed configuration.
6. Output directory creation.

Use my existing project paths where applicable:

```python
DATASET_ROOT = "/content/drive/MyDrive/Dataset/Lung_Cancer_Preprocessed Dataset"

MODELS_DIR = "/content/drive/MyDrive/Lung_Cancer_Trained_Models"

CHECKPOINTS_DIR = "/content/drive/MyDrive/Lung_Cancer_Checkpoints"
```

Make paths configurable from the central configuration section.

Do not assume the dataset folder structure without checking it. Include a dataset inspection step and clearly explain how the notebook expects training, validation, and test data to be organized.

If the existing dataset is already split, reuse it rather than splitting it again.

## 4. Dataset Loading and Preprocessing

Create reusable functions for:

- Loading training data.
- Loading validation data.
- Loading test data.
- Resizing images.
- Normalization or model-specific preprocessing.
- Batching and prefetching.

Important:
- Prevent data leakage between training, validation, and test sets.
- Do not apply random augmentation to validation or test data.
- Ensure preprocessing is appropriate for each selected architecture.
- Handle class labels consistently across all models.
- Check class distribution and display a simple visualization.

## 5. Modular Model Architecture

Create a clean `build_model(model_name)` function.

It should support:

```python
MODEL_NAMES = [
    "ResNet50",
    "EfficientNetB0",
    "MobileNetV2",
    "DenseNet121",
    "SwinTransformer"
]
```

Use transfer learning for CNN architectures where appropriate.

For Swin Transformer, implement or use a verified compatible architecture.

For my proposed model, keep the following components modular:

1. Patch Embedding.
2. Window-based Multi-Head Self-Attention.
3. Shifted Window Attention.
4. MLP block.
5. Patch Merging.
6. Classification Head.

I want to experiment with modifications in these components in the future.

Keep the architecture parameterized instead of hardcoding values throughout the implementation.

## 6. Proposed Model Experimentation

I specifically want to experiment with loss functions and Swin Transformer modifications.

Make it easy to modify:

- Loss function.
- Learning rate.
- Optimizer.
- Weight decay.
- Dropout.
- Label smoothing.
- Number of attention heads.
- Window size.
- Embedding dimension.
- Transformer depths.
- Classification head configuration.

Keep all these values in centralized configuration dictionaries.

Use a simple and consistent coding pattern.

Avoid unnecessary abstraction and excessive classes.

## 7. Model Training

Create a reusable training loop.

Example:

```python
trained_models = {}
trained_histories = {}
training_seconds = {}
checkpoint_paths = {}

for model_name in MODEL_NAMES:
    # Build model
    # Compile model
    # Train model
    # Save best checkpoint
    # Restore best weights
    # Store training history
```

Requirements:

- EarlyStopping.
- ModelCheckpoint.
- ReduceLROnPlateau.
- Configurable optimizer and loss function.
- Track training time.
- Save best weights.
- Restore best validation-loss weights before evaluation.
- Handle multiple models sequentially to avoid unnecessary GPU memory consumption.
- Clear the TensorFlow session when appropriate.
- Keep training logs understandable.

## 8. Evaluation and Comparison

Evaluate every trained model using the same test dataset.

Calculate:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Macro F1-score.
- Confusion Matrix.
- Classification Report.
- ROC-AUC, if applicable.

Generate:

1. Training vs Validation Accuracy graph.
2. Training vs Validation Loss graph.
3. Confusion Matrix for each model.
4. Model Comparison Table.
5. Training Time Comparison.
6. Accuracy and Macro F1 Comparison Graph.

Save evaluation results and graphs to Google Drive.

Use Macro F1 as an important metric, especially if the dataset is imbalanced.

## 9. Experiment Tracking

I want to compare different configurations of the proposed model.

Create a simple CSV experiment log containing:

- Experiment ID.
- Model Name.
- Loss Function.
- Learning Rate.
- Optimizer.
- Batch Size.
- Epochs.
- Important Swin Transformer parameters.
- Best Validation Accuracy.
- Best Validation Loss.
- Test Accuracy.
- Test Macro F1.
- Training Time.

This will help me identify which configuration performs best.

## 10. Notebook Structure

Organize the notebook into clearly separated sections.

Use this order:

1. Project Overview and Experiment Objective.
2. Google Colab Environment Setup.
3. Centralized Configuration.
4. Dataset Path and Structure Inspection.
5. Dataset Loading and Preprocessing.
6. Data Visualization and Class Distribution.
7. Model Architecture Implementation.
8. Loss Function and Optimizer Configuration.
9. Model Building and Summary.
10. Model Training.
11. Model Evaluation.
12. Performance Visualization.
13. Experiment Comparison.
14. Save Results and Final Models.
15. Future Experimentation Guide.

Every section should have a Markdown explanation before its code cells.

## 11. Coding Guidelines

Follow these rules strictly:

- Use TensorFlow/Keras consistently.
- Keep code beginner-friendly.
- Use short, meaningful variable and function names.
- Add simple English comments explaining what each important part does.
- Avoid unnecessarily long functions.
- Avoid duplicated code.
- Keep configurable values in one place.
- Use modular architecture.
- Avoid unnecessary external dependencies.
- Ensure all cells run in the correct order in Google Colab.
- Do not leave incomplete code or placeholder functions.
- Verify library APIs before using them.
- Handle common runtime errors where reasonable.
- Do not silently ignore errors.

## 12. Research and Experimental Integrity

This is an academic research project, so experimental reliability is important.

Please ensure:

- All models use the same train/validation/test split.
- All models use consistent evaluation metrics.
- Test data is not used for hyperparameter tuning.
- Random seeds are controlled.
- The best checkpoint is selected using validation loss.
- Class imbalance is considered.
- Clearly explain that changing the loss function alone does not guarantee accuracy improvement.
- Do not make fabricated performance claims.
- Do not call the architecture novel or research-proven without evidence.

Also, explain how I can perform a fair baseline vs proposed model comparison.

## 13. Final Deliverable

I do not want only code snippets or instructions.

**Generate a complete downloadable `.ipynb` notebook file** that I can directly upload and run in Google Colab.

Before finalizing:

1. Check the code for syntax errors.
2. Verify TensorFlow/Keras API compatibility.
3. Ensure the model configurations are consistent.
4. Ensure the training and evaluation functions work with multiple selected models.
5. Ensure the notebook is logically organized and executable from top to bottom.
6. Ensure the coding is suitable for google colab. no need to install any library as colab has all the support. 
7. Show all the performance and learning graph in a collage mode.

If any required dataset information is missing, make the notebook inspect the dataset and clearly explain which configuration I need to update.

The final notebook should be easy enough for a beginner to understand but structured well enough for serious academic experimentation.