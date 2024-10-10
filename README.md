# TikTok Video Claims Classification Project 
### Overview
This project focuses on building a machine learning model to classify TikTok videos as either "claims" or "opinions." The model helps TikTok mitigate misinformation and prioritize the review of reported videos, especially those containing claims that might violate the platform's terms of service.
**Business need and modeling objective**

TikTok users can report videos that they believe violate the platform's terms of service. Because there are millions of TikTok videos created and viewed every day, this means that many videos get reported—too many to be individually reviewed by a human moderator.

Analysis indicates that when authors do violate the terms of service, they're much more likely to be presenting a claim than an opinion. Therefore, it is useful to be able to determine which videos make claims and which videos are opinions.

TikTok wants to build a machine learning model to help identify claims and opinions. Videos that are labeled opinions will be less likely to go on to be reviewed by a human moderator. Videos that are labeled as claims will be further sorted by a downstream process to determine whether they should get prioritized for review. For example, perhaps videos that are classified as claims would then be ranked by how many times they were reported, then the top x% would be reviewed by a human each day.

A machine learning model would greatly assist in the effort to present human moderators with videos that are most likely to be in violation of TikTok's terms of service.

**Modeling design and target variable**

The data dictionary shows that there is a column called claim_status. This is a binary value that indicates whether a video is a claim or an opinion. This will be the target variable. In other words, for each video, the model should predict whether the video is a claim or an opinion.

This is a classification task because the model is predicting a binary class.

**Select an evaluation metric**

To determine which evaluation metric might be best, consider how the model might be wrong. There are two possibilities for bad predictions:

**False positives:** When the model predicts a video is a claim when in fact it is an opinion
**False negatives:** When the model predicts a video is an opinion when in fact it is a claim
2.  it's better for the model to predict false positives when it makes a mistake, and worse for it to predict false negatives. It's very important to identify videos that break the terms of service, even if that means some opinion videos are misclassified as claims. The worst case for an opinion misclassified as a claim is that the video goes to human review. The worst case for a claim that's misclassified as an opinion is that the video does not get reviewed and it violates the terms of service. A video that violates the terms of service would be considered posted from a "banned" author, as referenced in the data dictionary.

Because it's more important to minimize false negatives, the model evaluation metric will be recall.


Modeling workflow and model selection process


* Split the data into train/validation/test sets (60/20/20)

* Fit models and tune hyperparameters on the training set

* Perform final model selection on the validation set

* Assess the champion model's performance on the test set


**The project is structured into three parts:**

**Ethical Considerations:** Evaluating the ethical implications of the model and its potential consequences.
**Feature Engineering:** Selecting, transforming, and preparing features for the machine learning model.
**Modeling:** Building, evaluating, and comparing machine learning models to identify the best classifier.
Project Objectives
**Purpose:** Develop a machine learning model to classify videos into "claim" or "opinion" categories.
**Goal:** Assist TikTok in prioritizing videos for review by identifying claims that may violate terms of service.
**Modeling Approach:** Use classification models like Random Forest and XGBoost to predict whether a video presents a claim or an opinion.
### Project Workflow
This project follows the PACE (Plan, Analyze, Construct, Execute) problem-solving framework:

1. Plan
Business Problem: TikTok has too many videos to manually review each one. The goal is to automate the classification of videos to help identify those containing claims for faster human review.
Ethical Considerations: False negatives are more harmful than false positives. Minimizing false negatives ensures claims that violate terms of service are not overlooked.
Evaluation Metric: Recall is chosen as the key metric to prioritize minimizing false negatives.
2. Analyze
The dataset includes various features such as claim_status, video_transcription_text, and engagement metrics.
**Key columns include:**
claim_status: Binary target variable indicating whether a video is a claim.
video_transcription_text: Text-based feature, which is tokenized using CountVectorizer for feature extraction.
Basic data exploration and preprocessing, including handling text data and tokenizing video transcriptions, are performed.
3. Construct
Machine learning models are built, tuned, and validated using the following algorithms:
Random Forest Classifier
XGBoost Classifier
Hyperparameter tuning and cross-validation are used to optimize model performance.
4. Execute
Models are evaluated on the validation set, with the focus on recall.
Results are compared to identify the best-performing model.
Model performance is visualized using confusion matrices, recall scores, and other relevant metrics.
### Getting Started
Prerequisites
To run the project, the following libraries are required:

```python
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```
### Dataset
The dataset is automatically loaded in the notebook. However, ensure that the file tiktok_dataset.csv is available for manual runs if needed.

### Running the Project
To execute the project, run the following steps in a Jupyter notebook:

**Load Data:** Load the dataset into a Pandas dataframe.
**Data Preprocessing:** Preprocess and tokenize the video_transcription_text using CountVectorizer.
**Model Building:**
Random Forest
XGBoost
**Model Evaluation:** Evaluate both models on recall and other metrics.
**Next Steps:** Use the best model for classifying videos in production and fine-tune as needed.
Example: Model Building and Evaluation


```python
# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, recall_score
```
# Load dataset
```python
data = pd.read_csv('tiktok_dataset.csv')
```
# Preprocess data and tokenization
# ...

# Random Forest Model
```python
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
```
# XGBoost Model
```python
xgb = XGBClassifier(objective='binary:logistic', random_state=0)
xgb.fit(X_train, y_train)
```
# Model Evaluation
```python
y_pred_rf = rf.predict(X_test)
y_pred_xgb = xgb.predict(X_test)

print(classification_report(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_xgb))
```

**Evaluation Results**
The random forest model achieved an impressive recall score of 0.995 during cross-validation, making it highly effective in minimizing false negatives.

**Next Steps**
Fine-tune the best-performing model further.
Implement the model in production to classify reported videos.
Explore additional text-based features using more advanced NLP techniques.
**Contributing**
Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.

**License**
This project is licensed under the MIT License.

**Contact**
For any questions or feedback, please contact:

Author: Shumetie Tefera
Email: shumetie.tefera@example.com
