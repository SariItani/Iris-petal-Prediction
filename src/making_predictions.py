import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.utils import shuffle
from ucimlrepo import fetch_ucirepo
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.metrics import accuracy_score

iris = fetch_ucirepo(id=53)

df = iris.data.features
t = iris.data.targets

n_samples = 35
n_rows = df.shape[0]

top_samples = df[:n_samples]
middle_samples = df[n_rows // 2 - n_samples // 2: n_rows // 2 + n_samples // 2]
end_samples = df[-n_samples:]

X_new = pd.concat([top_samples, middle_samples, end_samples], ignore_index=True)
y_true = pd.concat([t[:n_samples], t[n_rows // 2 - n_samples // 2: n_rows // 2 + n_samples // 2], t[-n_samples:]], ignore_index=True)

X_raw = X_new
y_raw = y_true

# Preprocessing data

# 1. Removing null data
print("Missing values in features:")
print(X_new.isnull().sum())
if X_new.isnull().sum().any():
    imputer = SimpleImputer(strategy='mean')
    X_new = imputer.fit_transform(X_new)
    print("Missing values handled using mean imputation")
else:
    print("No missing values found")
    
# encoding y_true
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_true['class'])
print("\nEncoded Target:")
print(y_encoded)

# 2. Removing Outliers
def remove_outliers(data, thresh=1.5):
    for col in X_new.columns:
        Q1 = np.percentile(X_new[col], 25)
        Q3 = np.percentile(X_new[col], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - thresh * IQR
        upper_bound = Q3 + thresh * IQR
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data

removed_indices = []
X_new = remove_outliers(X_new.copy())
removed_indices.extend(X_new.index.tolist())
if len(removed_indices) > 0:
    y_encoded = y_encoded[removed_indices]
    removed_indices = [x for x in np.arange(1, len(y_encoded), 1) if x not in removed_indices]
    print(f"Removed {len(removed_indices)} data points from y_encoded.")
    print("The indices are:", removed_indices)

# 3. Transforming Skewed Data
transformer = PowerTransformer(method='yeo-johnson')
transformed_data = transformer.fit_transform(X_new)
X_new = pd.DataFrame(transformed_data, columns=X_new.columns)

# 4. Scale the features using standardization
scaler_standardized = StandardScaler()
X_standardized = scaler_standardized.fit_transform(X_new.copy())
X_standardized = pd.DataFrame(X_standardized, columns=X_new.columns)
print("Standard Features:")
print(X_standardized.head())

# 5. Normalization
scaler_normalized = MinMaxScaler()
X_normalized = scaler_normalized.fit_transform(X_new.copy())
X_normalized = pd.DataFrame(X_normalized, columns=X_new.columns)
print("Normalized Features:")
print(X_normalized.head())

# Save preprocessed data
X_standardized.to_csv('../data/prediction_standardized.csv', index=False)
X_normalized.to_csv('../data/prediction_normalized.csv', index=False)
y = pd.DataFrame(y_encoded, columns=['class'])
y.to_csv('../data/true_predictions.csv', index=False)

# Comparative Data Analysis
for feature in X_standardized.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(x=X_standardized[feature], bins=35)
    title = f'Histogram {feature}'
    plt.title(title, fontsize=16)
    plt.xlabel(feature, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.savefig(f'../results/making_predictions/histogram_standardized/{title}.png')
    
for feature in X_normalized.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(x=X_normalized[feature], bins=35)
    title = f'Histogram {feature}'
    plt.title(title, fontsize=16)
    plt.xlabel(feature, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.savefig(f'../results/making_predictions/histogram_normalized/{title}.png')
    
for label in y.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(x=y[label])
    title = f'Count plot of {label}'
    plt.title(title, fontsize=16)
    plt.xlabel(label, fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.savefig(f'../results/making_predictions/countplots/{title}.png')
    
# Load the trained models
standardized_models = {
    "KNeighborsClassifier(Standardized)": joblib.load('../models/KNeighborsClassifier(standardized)".pkl'),
    "LogisticRegression(Standardized)": joblib.load('../models/LogisticRegression(standardized)".pkl'),
    "DecisionTreeClassifier(Standardized)": joblib.load('../models/DecisionTreeClassifier(standardized)".pkl'),
    "RandomForestClassifier(Standardized)": joblib.load('../models/RandomForestClassifier(standardized)".pkl'),
    "SVC(Standardized)": joblib.load('../models/SVC(standardized)".pkl'),
    "MLPClassifier(Standardized)": joblib.load('../models/MLPClassifier(standardized)".pkl'),
}

normalized_models = {
    "KNeighborsClassifier(Normalized)": joblib.load('../models/KNeighborsClassifier(normalized)".pkl'),
    "LogisticRegression(Normalized)": joblib.load('../models/LogisticRegression(normalized)".pkl'),
    "DecisionTreeClassifier(Normalized)": joblib.load('../models/DecisionTreeClassifier(normalized)".pkl'),
    "RandomForestClassifier(Normalized)": joblib.load('../models/RandomForestClassifier(normalized)".pkl'),
    "SVC(Normalized)": joblib.load('../models/SVC(normalized)".pkl'),
    "MLPClassifier(Normalized)": joblib.load('../models/MLPClassifier(normalized)".pkl'),
}

raw_models = {
    "KNeighborsClassifier(Raw)": joblib.load('../models/KNeighborsClassifier(raw)".pkl'),
    "LogisticRegression(Raw)": joblib.load('../models/LogisticRegression(raw)".pkl'),
    "DecisionTreeClassifier(Raw)": joblib.load('../models/DecisionTreeClassifier(raw)".pkl'),
    "RandomForestClassifier(Raw)": joblib.load('../models/RandomForestClassifier(raw)".pkl'),
    "SVC(Raw)": joblib.load('../models/SVC(raw)".pkl'),
    "MLPClassifier(Raw)": joblib.load('../models/MLPClassifier(raw)".pkl'),
}

standardized_predictions = pd.DataFrame(index=X_standardized.index)
normalized_predictions = pd.DataFrame(index=X_normalized.index)
raw_predictions = pd.DataFrame(index=X_raw.index)

for model_name, model in standardized_models.items():
    predictions = model.predict(X_standardized)
    standardized_predictions[f'{model_name}_standardized'] = predictions
for model_name, model in normalized_models.items():
    predictions = model.predict(X_normalized)
    normalized_predictions[f'{model_name}_normalized'] = predictions
for model_name, model in raw_models.items():
    predictions = model.predict(X_raw)
    raw_predictions[f'{model_name}_raw'] = predictions

for column in standardized_predictions.columns:
    standardized_predictions[column].to_csv(f'../data/standardized_prediction_{column}.csv', index=False)
for column in normalized_predictions.columns:
    normalized_predictions[column].to_csv(f'../data/normalized_prediction_{column}.csv', index=False)
for column in raw_predictions.columns:
    raw_predictions[column].to_csv(f'../data/raw_prediction_{column}.csv', index=False)

for label in standardized_predictions.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(x=standardized_predictions[label])
    title = f'Count plot of {label}'
    plt.title(title, fontsize=16)
    plt.xlabel(label, fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.savefig(f'../results/making_predictions/countplots/{title}.png')
for label in normalized_predictions.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(x=normalized_predictions[label])
    title = f'Count plot of {label}'
    plt.title(title, fontsize=16)
    plt.xlabel(label, fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.savefig(f'../results/making_predictions/countplots/{title}.png')
for label in raw_predictions.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(x=raw_predictions[label])
    title = f'Count plot of {label}'
    plt.title(title, fontsize=16)
    plt.xlabel(label, fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.savefig(f'../results/making_predictions/countplots/{title}.png')

# Prediction Correctness Percentage
accuracy_results = {}
for column in standardized_predictions.columns:
    y_pred = standardized_predictions[column]
    accuracy = accuracy_score(y_encoded, y_pred)
    accuracy_results[column] = accuracy

accuracy_df = pd.DataFrame(list(accuracy_results.items()), columns=['Model', 'Accuracy'])
accuracy_df.to_csv('../results/making_predictions/standardized_models_accuracy.csv', index=False)

print("\nStandardized Models Accuracy:")
for model, accuracy in accuracy_results.items():
    print(f"{model}: {accuracy}")
    
accuracy_results = {}
for column in normalized_predictions.columns:
    y_pred = normalized_predictions[column]
    accuracy = accuracy_score(y_encoded, y_pred)
    accuracy_results[column] = accuracy

accuracy_df = pd.DataFrame(list(accuracy_results.items()), columns=['Model', 'Accuracy'])
accuracy_df.to_csv('../results/making_predictions/normalized_models_accuracy.csv', index=False)

print("\nNormalized Models Accuracy:")
for model, accuracy in accuracy_results.items():
    print(f"{model}: {accuracy}")
    
accuracy_results = {}
for column in raw_predictions.columns:
    y_pred = raw_predictions[column]
    accuracy = accuracy_score(y_true=y_raw, y_pred=y_pred)
    accuracy_results[column] = accuracy

accuracy_df = pd.DataFrame(list(accuracy_results.items()), columns=['Model', 'Accuracy'])
accuracy_df.to_csv('../results/making_predictions/raw_models_accuracy.csv', index=False)

print("\nRaw Models Accuracy:")
for model, accuracy in accuracy_results.items():
    print(f"{model}: {accuracy}")
    