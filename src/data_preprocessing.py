from matplotlib import pyplot as plt
from scipy.stats import f_oneway, chi2_contingency
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer, MinMaxScaler
import numpy as np
from sklearn.impute import SimpleImputer

# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets

# metadata
print(iris.metadata)

# variable information
print(iris.variables) 

print(X.head())
print(y.head())

# Exploratory Data Analysis
for feature in X.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(x=X[feature], bins=35)
    title = f'Histogram {feature}'
    plt.title(title, fontsize=16)
    plt.xlabel(feature, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.savefig(f'../results/data_analysis/histogram_raw/{title}.png')

for label in y.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(x=y[label])
    title = f'Count plot of {label}'
    plt.title(title, fontsize=16)
    plt.xlabel(label, fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.savefig(f'../results/data_analysis/countplot_raw/{title}.png')
    
# Perform ANOVA F-test for each feature
f_values, p_values = f_classif(X, y)
anova_results = pd.DataFrame({'feature': X.columns, 'f_value': f_values, 'p_value': p_values})

anova_results.to_csv('../results/data_analysis/anova_results.csv', index=False)

plt.figure(figsize=(10, 6))
plt.barh(anova_results['feature'], anova_results['f_value'])
plt.xlabel('F-Value', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('ANOVA F-Values for Iris Features', fontsize=16)
plt.tight_layout()
plt.savefig('../results/data_analysis/anova/anova_fvalues.png')

plt.figure(figsize=(10, 6))
plt.barh(anova_results['feature'], anova_results['p_value'])
plt.xlabel('P-Value', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('ANOVA F-Values for Iris Features', fontsize=16)
plt.tight_layout()
plt.savefig('../results/data_analysis/anova/anova_pvalues.png')

plt.figure(figsize=(10, 6))
anova_results.boxplot(column='f_value', by='feature', vert=False)
plt.xlabel('F-Value', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('ANOVA F-Values Distribution for Iris Features', fontsize=16)
plt.tight_layout()
plt.savefig('../results/data_analysis/anova/anova_fboxplot.png')

plt.figure(figsize=(10, 6))
anova_results.boxplot(column='p_value', by='feature', vert=False)
plt.xlabel('P-Value', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('ANOVA P-Values Distribution for Iris Features', fontsize=16)
plt.tight_layout()
plt.savefig('../results/data_analysis/anova/anova_pboxplot.png')

# Loop through each feature and perform Chi-Square test
chi2_results = pd.DataFrame(columns=['target_class', 'feature', 'chi2_statistic', 'p_value'])
for column in y.columns:
    y_target = y[column]
    for feature in X.columns:
        contingency_table = pd.crosstab(y_target, X[feature])
        chi2_statistic, p_value, _, _ = chi2_contingency(contingency_table)
        chi2_results = chi2_results.append({'target_class': column,
                                            'feature': feature,
                                            'chi2_statistic': chi2_statistic,
                                            'p_value': p_value}, ignore_index=True)
        print(f"Chi-Square test for feature: {feature}")
        print(f"Chi-Square statistic: {chi2_statistic}")
        print(f"p-value: {p_value}")
        
        plt.figure(figsize=(8, 6))
        plt.pcolor(contingency_table)
        plt.xlabel(feature)
        plt.ylabel(f'Target Class: {column}')
        title = f'Heatmap for {feature} vs. Target Class'
        plt.title(title)
        plt.colorbar(label='Count')
        plt.tight_layout()
        plt.savefig(f'../results/data_analysis/chi2/{title}.png')
        
chi2_results.to_csv('../results/data_analysis/chi2_results.csv', index=False)

# Preprocessing data

# 0. Removing null data
print("Missing values in features:")
print(X.isnull().sum())
print("Missing values in labels:")
print(y.isnull().sum())
if X.isnull().sum().any():
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    print("Missing values handled using mean imputation")
else:
    print("No missing values found")

# 1. Encode the categorical target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y['class'])
print("\nEncoded Target:")
print(y_encoded)

# 2. Removing Outliers
def remove_outliers(data, thresh=1.5):
    for col in X.columns:
        Q1 = np.percentile(X[col], 25)
        Q3 = np.percentile(X[col], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - thresh * IQR
        upper_bound = Q3 + thresh * IQR
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data

removed_indices = []
X = remove_outliers(X.copy())
removed_indices.extend(X.index.tolist())
if len(removed_indices) > 0:
    y_encoded = y_encoded[removed_indices]
    removed_indices = [x for x in np.arange(1, len(y_encoded), 1) if x not in removed_indices]
    print(f"Removed {len(removed_indices)} data points from y_encoded.")
    print("The indices are:", removed_indices)

# 3. Transforming Skewed Data
transformer = PowerTransformer(method='yeo-johnson')
transformed_data = transformer.fit_transform(X)
X = pd.DataFrame(transformed_data, columns=X.columns)

# 4. Scale the features using standardization
scaler_standardized = StandardScaler()
X_standardized = scaler_standardized.fit_transform(X.copy())
X_standardized = pd.DataFrame(X_standardized, columns=X.columns)
print("Standard Features:")
print(X_standardized.head())

# 5. Normalization for KNN Model
scaler_normalized = MinMaxScaler()
X_normalized = scaler_normalized.fit_transform(X.copy())
X_normalized = pd.DataFrame(X_normalized, columns=X.columns)
print("Normalized Features:")
print(X_normalized.head())

# Save preprocessed data
X_standardized.to_csv('../data/standardized.csv', index=False)
X_normalized.to_csv('../data/normalized.csv', index=False)
y_train = pd.DataFrame(y_encoded, columns=['class'])
y_train.to_csv('../data/target.csv', index=False)

# Exploratory Data Analysis
for feature in X_standardized.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(x=X_standardized[feature], bins=35)
    title = f'Histogram {feature}'
    plt.title(title, fontsize=16)
    plt.xlabel(feature, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.savefig(f'../results/data_analysis/histogram_standardized/{title}.png')

for label in y_train.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(x=y_train[label])
    title = f'Count plot of {label}'
    plt.title(title, fontsize=16)
    plt.xlabel(label, fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.savefig(f'../results/data_analysis/countplot_processed/{title}.png')
    
# Perform ANOVA F-test for each feature
f_values, p_values = f_classif(X_standardized, y_train)
anova_results = pd.DataFrame({'feature': X_standardized.columns, 'f_value': f_values, 'p_value': p_values})

anova_results.to_csv('../results/data_analysis/anova_results_standardized.csv', index=False)

plt.figure(figsize=(10, 6))
plt.barh(anova_results['feature'], anova_results['f_value'])
plt.xlabel('F-Value', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('ANOVA F-Values for Iris Features', fontsize=16)
plt.tight_layout()
plt.savefig('../results/data_analysis/anova_standardized/anova_fvalues.png')

plt.figure(figsize=(10, 6))
plt.barh(anova_results['feature'], anova_results['p_value'])
plt.xlabel('P-Value', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('ANOVA F-Values for Iris Features', fontsize=16)
plt.tight_layout()
plt.savefig('../results/data_analysis/anova_standardized/anova_pvalues.png')

plt.figure(figsize=(10, 6))
anova_results.boxplot(column='f_value', by='feature', vert=False)
plt.xlabel('F-Value', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('ANOVA F-Values Distribution for Iris Features', fontsize=16)
plt.tight_layout()
plt.savefig('../results/data_analysis/anova_standardized/anova_fboxplot.png')

plt.figure(figsize=(10, 6))
anova_results.boxplot(column='p_value', by='feature', vert=False)
plt.xlabel('P-Value', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('ANOVA P-Values Distribution for Iris Features', fontsize=16)
plt.tight_layout()
plt.savefig('../results/data_analysis/anova_standardized/anova_pboxplot.png')

# Loop through each feature and perform Chi-Square test
chi2_results = pd.DataFrame(columns=['target_class', 'feature', 'chi2_statistic', 'p_value'])
for column in y_train.columns:
    y_target = y_train[column]
    for feature in X_standardized.columns:
        contingency_table = pd.crosstab(y_target, X_standardized[feature])
        chi2_statistic, p_value, _, _ = chi2_contingency(contingency_table)
        chi2_results = chi2_results.append({'target_class': column,
                                            'feature': feature,
                                            'chi2_statistic': chi2_statistic,
                                            'p_value': p_value}, ignore_index=True)
        print(f"Chi-Square test for feature: {feature}")
        print(f"Chi-Square statistic: {chi2_statistic}")
        print(f"p-value: {p_value}")
        
        plt.figure(figsize=(8, 6))
        plt.pcolor(contingency_table)
        plt.xlabel(feature)
        plt.ylabel(f'Target Class: {column}')
        title = f'Heatmap for {feature} vs. Target Class'
        plt.title(title)
        plt.colorbar(label='Count')
        plt.tight_layout()
        plt.savefig(f'../results/data_analysis/chi2_standardized/{title}.png')
        
chi2_results.to_csv('../results/data_analysis/chi2_results_standardized.csv', index=False)

# Exploratory Data Analysis
for feature in X_normalized.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(x=X_normalized[feature], bins=35)
    title = f'Histogram {feature}'
    plt.title(title, fontsize=16)
    plt.xlabel(feature, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.savefig(f'../results/data_analysis/histogram_normalized/{title}.png')
    
# Perform ANOVA F-test for each feature
f_values, p_values = f_classif(X_normalized, y_train)
anova_results = pd.DataFrame({'feature': X_normalized.columns, 'f_value': f_values, 'p_value': p_values})

anova_results.to_csv('../results/data_analysis/anova_results_normalized.csv', index=False)

plt.figure(figsize=(10, 6))
plt.barh(anova_results['feature'], anova_results['f_value'])
plt.xlabel('F-Value', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('ANOVA F-Values for Iris Features', fontsize=16)
plt.tight_layout()
plt.savefig('../results/data_analysis/anova_normalized/anova_fvalues.png')

plt.figure(figsize=(10, 6))
plt.barh(anova_results['feature'], anova_results['p_value'])
plt.xlabel('P-Value', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('ANOVA F-Values for Iris Features', fontsize=16)
plt.tight_layout()
plt.savefig('../results/data_analysis/anova_normalized/anova_pvalues.png')

plt.figure(figsize=(10, 6))
anova_results.boxplot(column='f_value', by='feature', vert=False)
plt.xlabel('F-Value', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('ANOVA F-Values Distribution for Iris Features', fontsize=16)
plt.tight_layout()
plt.savefig('../results/data_analysis/anova_normalized/anova_fboxplot.png')

plt.figure(figsize=(10, 6))
anova_results.boxplot(column='p_value', by='feature', vert=False)
plt.xlabel('P-Value', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('ANOVA P-Values Distribution for Iris Features', fontsize=16)
plt.tight_layout()
plt.savefig('../results/data_analysis/anova_normalized/anova_pboxplot.png')

# Loop through each feature and perform Chi-Square test
chi2_results = pd.DataFrame(columns=['target_class', 'feature', 'chi2_statistic', 'p_value'])
for column in y_train.columns:
    y_target = y_train[column]
    for feature in X_normalized.columns:
        contingency_table = pd.crosstab(y_target, X_normalized[feature])
        chi2_statistic, p_value, _, _ = chi2_contingency(contingency_table)
        chi2_results = chi2_results.append({'target_class': column,
                                            'feature': feature,
                                            'chi2_statistic': chi2_statistic,
                                            'p_value': p_value}, ignore_index=True)
        print(f"Chi-Square test for feature: {feature}")
        print(f"Chi-Square statistic: {chi2_statistic}")
        print(f"p-value: {p_value}")
        
        plt.figure(figsize=(8, 6))
        plt.pcolor(contingency_table)
        plt.xlabel(feature)
        plt.ylabel(f'Target Class: {column}')
        title = f'Heatmap for {feature} vs. Target Class'
        plt.title(title)
        plt.colorbar(label='Count')
        plt.tight_layout()
        plt.savefig(f'../results/data_analysis/chi2_normalized/{title}.png')
        
chi2_results.to_csv('../results/data_analysis/chi2_results_normalized.csv', index=False)