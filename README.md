# Brain Tumor Classification Using an Ensemble of CNNs and Transformers

## Introduction
This project presents an advanced deep learning approach for multiclass brain tumor classification from MRI images. By integrating the complementary strengths of two convolutional neural networks (Xception and Inception V3) and a transformer-based model (DeiT), we build an ensemble that improves accuracy, robustness, and interpretability. Transfer learning, targeted data augmentation, and class-balancing strategies are leveraged to handle dataset variability and limited medical imaging data. Grad-CAM is employed for model explainability, providing visual insights into the regions influencing predictions. Our experiments demonstrate that this lightweight yet powerful ensemble achieves performance competitive with larger, more complex transformer-only models, offering a clinically viable solution.

## Project Metadata

### Authors
- **Team:** [Hassan Algizani, Malik Ibrahim, Abdulrahman Kamili]
- **Supervisor Name:** [Dr.Muzammil Behzah]
- **Affiliations:** [Your Affiliations]

### Project Documents
- **Presentation:** [Project Presentation](https://github.com/BRAIN-Lab-AI/3DGen-Diffusion-Models-for-3D-Object-Generation/blob/main/Brain_Tumour_Final%20(1).pptx)
- **Report:** [Project Report](https://github.com/BRAIN-Lab-AI/3DGen-Diffusion-Models-for-3D-Object-Generation/blob/main/Brain%20Tumor%20Classification%20and%20Detection.pdf)

### Reference Papers
- [Pretrained DeIT for Brain Tumor Classification: A Fine-Tuning Approach with Label Smoothing]([https://arxiv.org/abs/2112.10752](https://ieeexplore.ieee.org/document/10725957))

### Reference Datasets
- [Nickparvar Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/guslovesmath/tumor-classification-99-7-tensorflow-2-16)
- [Bhuvaji Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

## Project Technicalities

### Terminologies
- **Convolutional Neural Network (CNN):** A deep learning model that extracts local features from input images.
- **Vision Transformer (ViT) and DeiT:** Transformer-based models adapted for image tasks.
- **Transfer Learning:** Using pre-trained weights from large datasets like ImageNet to fine-tune models for a specific task.
- **Ensemble Learning:** Combining multiple models to boost performance.
- **Grad-CAM:** A technique for visualizing important regions of an input image for model predictions.
- **Region of Interest (ROI) Extraction:** Cropping images to focus on relevant anatomical areas.
- **Cross-Entropy Loss:** A loss function used for multiclass classification tasks.
- **Label Smoothing:** A regularization technique to prevent models from becoming overconfident.
- **ReduceLROnPlateau:** Learning rate adjustment strategy during training.

### Problem Statements
- **Problem 1:** CNNs alone can overfit on small MRI datasets and lack long-range spatial context.
- **Problem 2:** Transformers, while powerful, require significant computational resources and data augmentation.
- **Problem 3:** Class imbalance and limited interpretability of black-box models hinder clinical deployment.

### Loopholes or Research Areas
- **Dataset Size:** Small sample size risks overfitting for both CNNs and transformers.
- **Model Complexity:** Vision transformer models are heavy and slow compared to CNNs.
- **Interpretability:** Lack of explanations in automated predictions undermines clinical trust.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Model Ensembling:** Fuse complementary architectures (CNNs + Transformers) to balance strengths.
2. **Explainability Integration:** Use Grad-CAM to visualize what the model focuses on during classification.
3. **Data-Centric Methods:** ROI cropping and data augmentation without introducing synthetic bias.

### Proposed Solution: Code-Based Implementation
This repository provides the following components:

- **CNN Ensemble (Xception + InceptionV3):** Extract diverse features from MRI scans.
- **Transformer Ensemble (ViT + DeiT):** Capture global spatial context for classification.
- **Dual Fusion Strategies:** Concatenate features with a lightweight dense fusion head.
- **Visualization Module:** Generate Grad-CAM visualizations for model predictions.

### Key Components
- **`brain_tumor_classification-paper-author-work.ipynb`**: Author Original Notebook.
- **`experiment1_ensemble_CNN.ipynb`**: Experiment 1:CNN Ensemble.
- **`experiment2_ensemble_transformer.ipynb`**: Experiment 2:Vision Transformer Ensemble.
- **`deit_baseline.pth`**: [fine tuned deit model](https://kfupmedusa-my.sharepoint.com/:u:/g/personal/g202403940_kfupm_edu_sa/EQPlNF4H-xBAhWKPjb4z2AIBP1FcFAEHsVnamQLd8L3HwQ?e=3Mn6oQ).
- **`Experiment1_CNN.h5`**:[Experiment 1: CNN Ensemble Model](https://kfupmedusa-my.sharepoint.com/:u:/g/personal/g202403940_kfupm_edu_sa/Ed1beUnfhWpKmoNTV0NK6QoB1Fd6XLIkVgY3ux3NIQCGXw?e=TYkb3k).
- **`Experiment2_Vision_Transformer.keras`**: [Experiment 2: Vision Transformer Ensemble Model](https://kfupmedusa-my.sharepoint.com/:u:/g/personal/g202403940_kfupm_edu_sa/EbelpNvC5Q9CsivckVyHxl4BxCURa-VGJ-kMRdr3KvWMRQ?e=ePv6sT).

## Model Workflow

1. **Input:**
   - **MRI Images:** Brain MRI slices preprocessed (ROI extraction, normalization).

2. **Feature Extraction:**
   - CNNs and Transformers extract complementary features.

3. **Fusion:**
   - Features are concatenated and passed through a small dense fusion head.

4. **Prediction:**
   - A final softmax layer predicts one of the four tumor categories.

5. **Interpretability:**
   - Grad-CAM visualizations highlight regions important for predictions.

## How to Run the Code
**Use kaggle and upload the jupyter notebooks and run the cells**

## Acknowledgments
- **Open-Source Communities:** Thanks to TensorFlow, PyTorch, and Kaggle Datasets contributors.
- **Mentors and Colleagues:** Gratitude to our supervisors and teammates for constant feedback and support.
- **Research Inspiration:** Thanks to prior work on CNNs, Vision Transformers, and medical imaging research communities.
