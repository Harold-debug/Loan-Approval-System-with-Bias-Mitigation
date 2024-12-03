# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# Load Dataset
def load_data(file_path):
    """Load the CSV file into a Pandas DataFrame."""
    df = pd.read_csv(file_path)
    return df

# Preprocessing
def preprocess_data(df):
    """Clean and preprocess the data."""
    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Convert categorical columns to numeric
    categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Define target and features
    target = 'Loan_Status'
    X = df.drop(columns=[target])
    y = df[target].map({'Y': 1, 'N': 0})  # Convert to binary
    return X, y

# Split Dataset
def split_data(X, y):
    """Split the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Baseline Model Training
def train_baseline(X_train, y_train, X_test, y_test):
    """Train a baseline model and evaluate performance."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Baseline Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model

# Bias Mitigation Techniques
def apply_preprocessing_bias_mitigation(X, y, protected_attribute):
    """Apply pre-processing bias mitigation techniques."""
    # Convert to AIF360 BinaryLabelDataset
    dataset = BinaryLabelDataset(df=pd.concat([X, y], axis=1),
                                 label_names=['Loan_Status'], 
                                 protected_attribute_names=[protected_attribute])
    reweighing = Reweighing(unprivileged_groups=[{protected_attribute: 0}],
                            privileged_groups=[{protected_attribute: 1}])
    transformed_dataset = reweighing.fit_transform(dataset)
    return transformed_dataset

def apply_inprocessing_bias_mitigation(X_train, y_train, X_test, y_test, protected_attribute):
    """Apply in-processing bias mitigation using adversarial debiasing."""
    privileged = [{protected_attribute: 1}]
    unprivileged = [{protected_attribute: 0}]

    # Convert to AIF360 datasets
    train_dataset = BinaryLabelDataset(df=X_train, label_names=['Loan_Status'], protected_attribute_names=[protected_attribute])
    test_dataset = BinaryLabelDataset(df=X_test, label_names=['Loan_Status'], protected_attribute_names=[protected_attribute])

    # In-processing with Adversarial Debiasing
    sess = tf.Session()
    adv_debias = AdversarialDebiasing(privileged_groups=privileged,
                                      unprivileged_groups=unprivileged,
                                      scope_name='debias',
                                      sess=sess)
    adv_debias.fit(train_dataset)
    pred_dataset = adv_debias.predict(test_dataset)
    return pred_dataset

def apply_postprocessing_bias_mitigation(y_test, y_pred, protected_attribute):
    """Apply post-processing bias mitigation."""
    # Convert to AIF360 BinaryLabelDataset
    test_dataset = BinaryLabelDataset(df=pd.concat([y_test, y_pred], axis=1),
                                      label_names=['Loan_Status'],
                                      protected_attribute_names=[protected_attribute])

    ceop = CalibratedEqOddsPostprocessing(privileged_groups=[{protected_attribute: 1}],
                                          unprivileged_groups=[{protected_attribute: 0}])
    ceop.fit(test_dataset, test_dataset)
    pred_transformed = ceop.predict(test_dataset)
    return pred_transformed

# Fairness Metrics Evaluation
def evaluate_fairness(dataset, privileged, unprivileged):
    """Evaluate fairness metrics."""
    metric = BinaryLabelDatasetMetric(dataset,
                                      privileged_groups=privileged,
                                      unprivileged_groups=unprivileged)
    print("Statistical Parity Difference:", metric.mean_difference())

# Main Workflow
if __name__ == "__main__":
    # Load the dataset
    file_path = "loan_data.csv"  # Update with your dataset path
    df = load_data(file_path)

    # Preprocess the data
    X, y = preprocess_data(df)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train baseline model
    baseline_model = train_baseline(X_train, y_train, X_test, y_test)

    # Bias Mitigation - Pre-processing
    transformed_dataset = apply_preprocessing_bias_mitigation(X, y, protected_attribute='Gender')
    print("Applied Pre-processing Bias Mitigation.")

    # Bias Mitigation - In-processing
    inprocessing_results = apply_inprocessing_bias_mitigation(X_train, y_train, X_test, y_test, protected_attribute='Gender')
    print("Applied In-processing Bias Mitigation.")

    # Bias Mitigation - Post-processing
    postprocessing_results = apply_postprocessing_bias_mitigation(y_test, baseline_model.predict(X_test), protected_attribute='Gender')
    print("Applied Post-processing Bias Mitigation.")

    # Fairness Metrics Evaluation
    evaluate_fairness(transformed_dataset, privileged=[{'Gender': 1}], unprivileged=[{'Gender': 0}])