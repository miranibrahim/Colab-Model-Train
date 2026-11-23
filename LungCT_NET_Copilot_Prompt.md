# GitHub Copilot Prompt: Implement a LungCT-NET-Inspired 3-Class Model

Review my existing lung cancer classification project and create a **new, separate Jupyter notebook** named:

`LungCT_NET_Three_Class_Classification.ipynb`

Use the paper details I provided for **“LungCT-NET: An explainable transfer learning-based robust ensemble model for lung cancer diagnosis”** as the methodological reference, but adapt the implementation to my existing dataset and project.

## 1. My dataset — important
DATASET_ROOT = "/content/drive/MyDrive/Dataset/Lung_Cancer_Preprocessed_Dataset"
My dataset is **already preprocessed, split, and augmented**. Do not repeat preprocessing, splitting, or augmentation.

Use the existing `train`, `validation`, and `test` folders and preserve their contents.

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Benign | 735 | 70 | 71 |
| Malignant | 735 | 157 | 158 |
| Normal | 735 | 62 | 63 |
| **Total** | **2,205** | **289** | **292** |

Before training, inspect the actual folder structure and verify these counts. If the observed counts differ, report the difference instead of silently changing the dataset.

This is a **three-class classification problem: Benign, Malignant, and Normal**. The paper's binary labels (Benign/Malignant) must therefore be adapted. Do not discard or merge the Normal class.

## 2. Model architecture

Implement the four transfer-learning models used in the paper's final ensemble:

- VGG16
- VGG19
- MobileNetV2
- EfficientNetB0

Use ImageNet pretrained weights, remove the original classification head, and initially freeze the convolutional base.

Use the paper's described classification heads where appropriate:
- VGG16/VGG19: Flatten → Dense(4096, ReLU) → Dropout(0.5) → Dense(4096, ReLU) → Dropout(0.5) → Dense(3, softmax).
- MobileNetV2/EfficientNetB0: GlobalAveragePooling2D → Dense(128, ReLU) → Dense(3, softmax).

Make the output layer suitable for three classes. Keep architecture settings configurable and explain any practical adjustment needed for GPU memory.

Check each architecture's required preprocessing. Keep the dataset split and labels consistent across models.

## 3. Training

- Use categorical or sparse categorical cross-entropy according to the label format.
- Use Adam (or explain any justified alternative).
- Train for up to 100 epochs, with configurable batch size and learning rate.
- Use validation-based EarlyStopping and ModelCheckpoint.
- Save the best checkpoint for each model.
- Record training/validation loss, accuracy, learning rate, and training time.
- Set random seeds and enable deterministic operations where supported.
- Record library versions, configuration, and limitations to exact reproducibility.

Do not use the test set for training, model selection, or hyperparameter tuning.

## 4. Stacking ensemble — avoid data leakage

Build a stacking ensemble using the four CNNs' **three-class probability outputs**.

Use:
- Decision Tree
- Random Forest
- Extra Trees
- Logistic Regression as the final meta-classifier

Important: generate training features using **out-of-fold (OOF) predictions** from the training set, not predictions from CNNs on the same samples they were trained on. This prevents leakage into the stacking classifiers.

Fit the base tree classifiers on the OOF features, and train the Logistic Regression meta-classifier using appropriate cross-validated predictions from the base classifiers. Keep validation and test data out of fitting. Explain the stacking process clearly.

If the paper's exact stacking procedure cannot be reproduced from the available details, clearly state the implementation choice rather than claiming it is identical to the paper.

## 5. Evaluation and visualizations

Evaluate each CNN and the final ensemble on the untouched test set.

Report:
- Accuracy
- Macro and weighted precision, recall, and F1-score
- Per-class precision, recall, and F1-score
- Confusion matrix
- One-vs-rest ROC curves and macro/micro ROC-AUC
- Precision-Recall curves and average precision
- Specificity for each class (one-vs-rest)

Create clear, labeled graphs for:
- Training vs validation accuracy
- Training vs validation loss
- Learning rate over epochs, if recorded
- Confusion matrix
- Multiclass ROC curves
- Precision-Recall curves
- Model metric comparison
- Training time comparison

Explain briefly how to interpret each graph, including signs of overfitting and underfitting. Do not use binary-only evaluation for this three-class task.

## 6. Explainability

Add an optional, clearly separated explainability section:
- Generate Grad-CAM heatmaps for representative test images for the CNN models.
- If SHAP is practical, explain and implement an appropriate multiclass approach for CNN predictions and the stacking model.
- Do not claim that SHAP/Grad-CAM proves clinical validity. Clearly describe limitations.

## 7. Results and notebook quality

Save models, histories, metrics, predictions, plots, and experiment configuration in a timestamped results folder so previous runs are not overwritten.

Keep the notebook beginner-friendly:
- Use clear Markdown headings and short explanations.
- Break work into small, logical cells.
- Add useful comments without over-commenting.
- Avoid huge code cells and unnecessary abstractions.
- Make it executable from top to bottom.
- Explain important outputs in simple English.

## Before writing code

First inspect my existing project to identify the framework, dataset paths, folder naming, image format, label encoding, and current environment. Reuse the actual paths and compatible code. Do not invent paths or silently alter my dataset.

Create the new notebook only. Do not overwrite my existing notebooks or dataset. At the end, provide a short summary of the notebook, assumptions, and any issues I need to check before running a full training session.
