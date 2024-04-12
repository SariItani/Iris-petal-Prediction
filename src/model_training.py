import csv
import os
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve, train_test_split, validation_curve
from sklearn.metrics import accuracy_score, precision_recall_curve, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from ucimlrepo import fetch_ucirepo

iris = fetch_ucirepo(id=53)

if os.path.isfile('../results/model_training/model_summary.csv'):
    os.remove('../results/model_training/model_summary.csv')
    with open('../results/model_training/model_summary.csv', 'w') as file:
        pass

X_standardized = pd.read_csv('../data/standardized.csv')
X_normalized = pd.read_csv('../data/normalized.csv')
y = pd.read_csv('../data/target.csv')['class']

X_raw = iris.data.features
y_raw = iris.data.targets

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_standardized, y, test_size=0.2, random_state=42)
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(f'../results/model_training/{title}.png')

def plot_roc(y_true, y_pred_probs, classes, title='ROC Curve'):
    plt.figure(figsize=(10, 8))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_true == classes[i], y_pred_probs[:, i])
        plt.plot(fpr, tpr, label=f'Class {classes[i]} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'../results/model_training/{title}.png')

def plot_precision_recall(y_true, y_pred_probs, classes, title='Precision-Recall Curve'):
    plt.figure(figsize=(10, 8))
    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(y_true == classes[i], y_pred_probs[:, i])
        plt.plot(recall, precision, label=f'Class {classes[i]}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.savefig(f'../results/model_training/{title}.png')

def plot_feature_importance(importance, features, title='Feature Importance'):
    feature_importance = pd.Series(importance, index=features)
    feature_importance.nlargest(10).plot(kind='barh')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.savefig(f'../results/model_training/{title}.png')

def plot_learning_curve(estimator, title, X, y, cv=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.title(title)
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(f'../results/model_training/{title}.png')

def plot_validation_curve(estimator, title, X, y, param_name, param_range, cv=None):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range, cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(f'../results/model_training/{title}.png')

def train_model(model, X_train, X_test, y_train, y_test, data_type="standardized"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Model: {model.__class__.__name__} ({data_type})")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("-" * 40)
    
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)
    
    metrics = {
        'Model': model.__class__.__name__,
        'Data Type': data_type,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Classification Report': report
    }

    file_path = '../results/model_training/model_summary.csv'
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
        if not file_exists or os.path.getsize(file_path) == 0:
            writer.writeheader()
        writer.writerow(metrics)
    
    joblib.dump(model, f'../models/{model.__class__.__name__}({data_type})".pkl')
    
    plot_confusion_matrix(y_test, y_pred, classes=model.classes_, title=f'confusion_matrix_{model.__class__.__name__}({data_type})')
    plot_roc(y_test, model.predict_proba(X_test), classes=model.classes_, title=f'roc_curve_{model.__class__.__name__}({data_type})')
    plot_precision_recall(y_test, model.predict_proba(X_test), classes=model.classes_, title=f'precision_recall_{model.__class__.__name__}({data_type})')
    if hasattr(model, 'feature_importances_'):
        plot_feature_importance(model.feature_importances_, X_train.columns, title=f'feature_importance_{model.__class__.__name__}({data_type})')
    plot_learning_curve(model, f'Learning Curve {model.__class__.__name__} ({data_type})', X_train, y_train, cv=5)
    param_range = np.arange(1, 11)
    doit = False
    if isinstance(model, RandomForestClassifier):
        param_name = 'n_estimators'
        doit = True
    elif isinstance(model, KNeighborsClassifier):
        param_name = 'n_neighbors'
        doit = True
    if doit:
        plot_validation_curve(model, f'Validation Curve {model.__class__.__name__} ({data_type})', X_train, y_train, param_name=param_name, param_range=param_range, cv=5)
    
train_model(KNeighborsClassifier(n_neighbors=3), X_train_s, X_test_s, y_train_s, y_test_s)
train_model(LogisticRegression(multi_class='multinomial', solver='lbfgs'), X_train_s, X_test_s, y_train_s.values.ravel(), y_test_s)
train_model(DecisionTreeClassifier(), X_train_s, X_test_s, y_train_s, y_test_s)
train_model(RandomForestClassifier(), X_train_s, X_test_s, y_train_s, y_test_s)
train_model(SVC(probability=True), X_train_s, X_test_s, y_train_s, y_test_s)
train_model(MLPClassifier(), X_train_s, X_test_s, y_train_s, y_test_s)

train_model(KNeighborsClassifier(n_neighbors=3), X_train_n, X_test_n, y_train_n, y_test_n, data_type="normalized")
train_model(LogisticRegression(multi_class='multinomial', solver='lbfgs'), X_train_n, X_test_n, y_train_n.values.ravel(), y_test_n, data_type="normalized")
train_model(DecisionTreeClassifier(), X_train_n, X_test_n, y_train_n, y_test_n, data_type="normalized")
train_model(RandomForestClassifier(), X_train_n, X_test_n, y_train_n, y_test_n, data_type="normalized")
train_model(SVC(probability=True), X_train_n, X_test_n, y_train_n, y_test_n, data_type="normalized")
train_model(MLPClassifier(), X_train_n, X_test_n, y_train_n, y_test_n, data_type="normalized")

train_model(KNeighborsClassifier(n_neighbors=3), X_train_r, X_test_r, y_train_r, y_test_r, data_type="raw")
train_model(LogisticRegression(multi_class='multinomial', solver='lbfgs'), X_train_r, X_test_r, y_train_r.values.ravel(), y_test_r, data_type="raw")
train_model(DecisionTreeClassifier(), X_train_r, X_test_r, y_train_r, y_test_r, data_type="raw")
train_model(RandomForestClassifier(), X_train_r, X_test_r, y_train_r, y_test_r, data_type="raw")
train_model(SVC(probability=True), X_train_r, X_test_r, y_train_r, y_test_r, data_type="raw")
train_model(MLPClassifier(), X_train_r, X_test_r, y_train_r, y_test_r, data_type="raw")
