**TITLE**
Explainable AI-Based Federated Deep Learning for Kidney Stone Classification

**Description**
This project implements FCTKSD, a privacy-preserving, explainable, and scalable deep learning framework for automated kidney stone detection and classification from CT images. The framework uniquely integrates three core components. The first component is Federated Learning, which enables privacy-preserving collaborative training across distributed hospitals, ensuring that raw patient CT data never leaves local premises. The second component is a Deep Learning backbone based on ResNet-18, optimized with nine confusion-matrix-based evaluation metrics to ensure robust and reliable classification performance. The third component is Explainable AI, which employs five complementary techniques, namely Grad-CAM, LIME, Integrated Gradients, Occlusion Sensitivity, and SHAP, along with bounding-box localization, to provide clinically interpretable visual and numerical explanations that highlight the regions influencing model predictions. This framework addresses the critical research gap where existing studies focus on accuracy, explainability, or privacy independently, as FCTKSD unifies all three into a single clinically deployable pipeline for computer-aided kidney stone diagnosis.

**Dataset Information**
The model was trained on the Kaggle Axial CT Imaging Dataset for Kidney Stone Detection. The dataset contains two classes: stone (1,577 images) and non-stone (1,787 images). Several data augmentation approaches were applied to extend the dataset to approximately 7,000 CT images per class, resulting in a total of approximately 14,000 images. An 80:20 split was applied, with 11,200 images for training and 2,800 for testing, with preserved class distribution across all three simulated hospital clients.

**Code Overview**
The pipeline integrates:

1. Data Preprocessing & Augmentation
   - Resizing to 224×224 pixels, normalization using mean and standard deviation, denoising, and augmentation applied only to the training set (rotation, flipping, scaling, intensity variation) to preserve patient privacy while improving model robustness.
2. Federated Learning Setup
   - Three simulated hospitals (clients) with locally stored non-IID CT datasets.
   -ResNet-18 backbone trained independently at each client.
   -  Client-specific optimizers: AdamW at Hospital 1, SGD with Momentum at Hospital 2, and RMSprop at Hospital 3.
   - Eight communication rounds with batch size = 32 and 8 local epochs.
 - Only model parameters are transmitted to the central server, not raw CT data.
3. Reliability-Weighted Global Model Aggregation
   - Custom reliability scoring mechanism evaluates each client's contribution.
   - Uses Federated Averaging (FedAvg) with weighted aggregation that prioritizes trustworthy clients with larger datasets.
4. Performance Evaluation
   - Confusion matrix generation for the testing set.
   - Computation of accuracy, misclassification rate, specificity, recall, precision, negative predictive value, false positive rate, false negative rate, and F1-score.
   - ROC curve with AUC and Precision-Recall (PR) curve.
5. Robustness & Statistical Analysis
   - Convergence monitoring via global loss variation.
   - Adaptive learning-rate mechanism for stable convergence.
6. Explainable AI (XAI) Integration
   - Grad-CAM class activation heatmaps.
   - LIME local interpretable explanations.
   - Integrated Gradients pixel-wise attribution.
   - Occlusion Sensitivity region masking verification.
   - SHAP feature contribution scores.
   - Bounding Box explicit stone localization.

All analyses are implemented in PyTorch and TensorFlow, with plotting via Matplotlib and Seaborn.

**Usage Instructions**
1. Dataset Preparation
   - Organize the dataset directory at each hospital as:
      text
      /path/to/hospital_k/
      ├── stone/
      └── non_stone/
   - Update the path in the code:

      python
      data_dir = "/path/to/hospital_k"
2. Run Federated Training
   - Execute the federated server across three simulated hospitals:
   - python federated_server.py --rounds 8 --batch_size 32 --clients 3
3. Run Local Training (per client)
   - python local_training.py --hospital 1 --epochs 8 --lr 0.0001
   Or open and run FCTKSD_Workflow.ipynb in Jupyter/Colab.
4. Run Evaluation
   - python evaluate.py --model models/global_model.h5 --test_data /path/to/test
5. Run XAI Explanations
   - python xai_explainer.py --image path/to/ct_image.png --methods gradcam,lime,ig,occlusion,shap,bbox
6. Outputs Generated
   - Client-wise training and testing accuracy plots across federated rounds
   - Client-wise accuracy and loss across FL rounds
   - Global model accuracy across FL rounds
   -  Confusion matrix for the testing set
   - ROC curve with AUC
   - Precision-Recall (PR) curve
   - XAI heatmaps including Grad-CAM, LIME, Integrated Gradients, Occlusion Sensitivity, SHAP, and Bounding Box
   - Classification reports

**Requirements**
| Library | Version (Recommended) |
|----------|-----------------------|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.1 |
| torchvision | ≥ 0.16 |
| TensorFlow | ≥ 2.12 |
| numpy | ≥ 1.25 |
| scikit-learn | ≥ 1.3 |
| matplotlib | ≥ 3.8 |
| seaborn | ≥ 0.12 |
| OpenCV | ≥ 4.8 |

Install dependencies:
pip install torch torchvision tensorflow numpy scikit-learn matplotlib seaborn opencv-python

**Methodology Summary**
Component	Description
|Base Model	| ResNet-18 (federated backbone)|
|Training Strategy |	Federated Learning with reliability-weighted aggregation|
|Preprocessing |	Resizing (224×224), normalization, denoising|
|Augmentation |	Rotation, flipping, scaling, intensity variation|
|Dataset Split |	80% training / 20% testing with preserved class distribution|
|Clients | 3 simulated hospitals (non-IID data)|
|Optimizers |	AdamW (H1), SGD+Momentum (H2), RMSprop (H3)|
|Federated Rounds |	8 rounds, batch size = 32, 8 local epochs|
|Aggregation |	Reliability-weighted FedAvg|
|Evaluation | Metrics	Accuracy, Misclassification Rate, Specificity, Recall, Precision, NPV, FPR, FNR, F1-Score|
|Robustness Analysis |	Convergence monitoring, adaptive learning rate|
|nterpretability | 5 XAI techniques: Grad-CAM, LIME, Integrated Gradients, Occlusion Sensitivity, SHAP, and Bounding Box|

**Performance Summary**
|Metric | Value|
|Accuracy |	96.8%|
|Misclassification Rate |	3.2%|
|Specificity |	94.9%|
|Recall (Sensitivity) |	98.6%|
|Precision |	95.1%|
|Negative Predictive Value |	98.6%|
|False Positive Rate |	5.1%|
|False Negative Rate	| 1.4%|
|F1-Score |	0.97|
|AUC |	0.99| 

**Visualization Examples**
- Confusion Matrices – class-wise accuracy comparison for training and testing sets
- Training/Testing Loss & Accuracy Curves – convergence analysis across epochs
- Five-Fold Cross-Validation Plots – robustness and generalizability assessment
- Grad-CAM Heatmaps – highlight retinal regions (hemorrhages, exudates, microaneurysms) influencing predictions

**Evaluation Environment**
Developed and tested on:
   - OS: Windows 10 / Linux
   - Platform: Google Colab (GPU acceleration)
   - Frameworks: PyTorch and TensorFlow
   - Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, OpenCV, SHAP, LIME, Captum
   - Federated Setup: 3 simulated hospitals, 8 rounds, batch size = 32, 8 local epochs

**Results and Discussion**
  - ResNet-18 within the federated learning framework achieved 96.8% accuracy, 94.9% specificity, 98.6% recall, 95.1% precision, and 0.97 F1-score.
   - Federated learning enabled privacy-preserving collaboration across three hospitals without sharing raw CT data.
   - Reliability-weighted aggregation using FedAvg demonstrated fast convergence and stable performance across communication rounds.
   - All clients converged rapidly, reaching 97–98% testing accuracy within the first few communication rounds.
   - Optimizer diversity with AdamW, SGD with Momentum, and RMSprop demonstrated the framework's robustness to heterogeneous client configurations.
   - The ROC curve with AUC of 0.99 confirms high discrimination power between Stone and Non-Stone classes.
   - The Precision-Recall curve shows near-perfect precision across recall levels, indicating very low false positives.
   - Five XAI techniques provided multi-perspective visual explanations, highlighting stone regions and clinically relevant anatomy.
   - The framework is designed as a clinical decision support tool for radiologists, where avoiding false negatives is especially important.

**Limitations**
  - Current study is limited to a single dataset and binary classification, restricting generalizability to diverse clinical environments and multi-class stone severity grading.
  - Patient-level stratification was not possible due to unavailable patient identifiers.
  - XAI outputs remained qualitative only, as radiologist annotations were unavailable for quantitative validation.
  - The impact of individual preprocessing steps was not evaluated through ablation studies.
  - Ensemble and advanced architectures were not explored; only ResNet-18 was benchmarked within the FL framework.
  - Future work should address these gaps through multi-center and multi-vendor CT datasets, detailed stone characterization (type, size, location), differentially-private training and secure aggregation, adaptive federated algorithms for non-IID data, quantitative evaluation of explainability techniques based on clinician feedback, and smooth integration into clinical workflows such as PACS and decision-support systems.

**Conclusion**
The proposed FCTKSD framework delivers:
  - State-of-the-art accuracy (96.8%) with ResNet-18 for binary kidney stone classification
  - Privacy-preserving federated learning across three simulated hospitals without centralized data sharing
  - Reliability-weighted FedAvg aggregation improving robustness and generalization
  - Comprehensive benchmarking with nine confusion-matrix-based evaluation metrics
  - Robustness validation through convergence monitoring and adaptive learning rate
  - Five clinically interpretable XAI techniques (Grad-CAM, LIME, Integrated Gradients, Occlusion Sensitivity, SHAP) with bounding-box localization highlighting disease-related anatomical regions

This framework represents a practical, trustworthy AI pipeline for urological diagnostics, emphasizing accuracy, privacy, reliability, and transparency, with strong potential for clinical deployment as a screening and referral support tool in distributed healthcare systems.