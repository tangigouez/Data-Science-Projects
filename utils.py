# Data manipulation
import pandas as pd

# Start Python Imports
import math

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')

# Machine learning
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.linear_model import LogisticRegression


def count_missing_values(df):
    """
    Function to count the number of missing values in each columns of the dataframe
        Parameters:
            df : dataframe that we want to scan to identify missing values
        Return:
            missing_vals (dict) : dictionary listing the number of missing values per column

    """
    print('Number of missing or NaN value for each column:')
    keys = []
    values = []

    for element in df.columns:
        keys.append(element)
        values.append(
            str(df[element].isnull().sum()) + ' (' + str(round(df[element].isnull().sum() / len(df) * 100, 2)) + '%)')

    missing_vals = {keys[i]: values[i] for i in range(len(keys))}

    return missing_vals


def bar_chart_compare(dataset, chart_title, feature1, feature2, target, y_axis_label):
    """
    Function to plot a bar graph of the target feature value segmented by categories
        Parameters
            dataset: target dataframe
            chart_title: title of the chart
            feature1: feature that we want to compare
            feature2: name of the feature we will use to segment feature1
            target: name of the target feature
            y_axis_label: title of the y axis
        Return
            bar_plot: bar graph of the target feature value segmented by categories
    """
    plt.figure(figsize = [5, 5])
    plt.title(chart_title)
    show_plot = sns.barplot(x=feature1, y=target, hue=feature2, ci=None, data=dataset).set_ylabel(y_axis_label)

    return show_plot


def show_countplot_compare(dataset, feature1, feature2, title):
    """
        Function to plot a countplot of the target feature value segmented by categories
            Parameters
                dataset: target dataframe
                feature1: feature that we want to compare
                feature2: name of the feature we will use to segment feature1
                title: title of the chart
            Return
                countplot: bar graph of the target feature value segmented by categories
        """
    fig, ax = plt.subplots(figsize= (5,5))
    show_plot = sns.countplot(x=feature1, hue=feature2, data=dataset, ax=ax).set_title(title)

    return show_plot


def fit_ml_algo(algo, X_train, y_train, cv):
    """
    Function that runs the requested algorithm and returns the accuracy metrics.
        Parameters
            algo: the name of the algorithm we will run from sklearn library
            X_train: the input data of the model
            y_train: the real values of the target feature from training dataset
            cv: k-fold cross validation, the number of blocks in which we divide our data
            n_jobs: number of CPU cores used during cross validation loop (-1 means using all processors)
        Return
            acc: model accuracy score
            acc_cv: the adjusted accuracy score after k-fold cross validation

    """
    # One Pass
    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train) * 100, 2)

    # Cross Validation
    train_pred = model_selection.cross_val_predict(algo,
                                                   X_train,
                                                   y_train,
                                                   cv=cv,
                                                   n_jobs=-1)
    # Cross-validation accuracy metric
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)

    return train_pred, acc, acc_cv


def show_swarm_plot(dataset, target, feature1, feature2, title):
    """
    Function that runs the requested algorithm and returns the accuracy metrics.
        Parameters
            dataset: the name of the algorithm we will run from sklearn library
            target: the input data of the model
            feature1: feature that we want to compare
            feature2: name of the feature we will use to segment feature1
            title: title of the chart
        Return
            swarmplot: plot the distribution of data accross features
    """
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.grid(True)
    plt.xticks(list(range(0, 100, 2)))

    return sns.swarmplot(y=feature1, x=feature2, hue=target, data= dataset).set_title(title)


def plot_logistic_regression_feature_importance(X_train, y_train):
    """
    Function that runs the requested algorithm and returns the accuracy metrics.
        Parameters
            X_train: the input data of the model
            y_train: the real values of the target feature from training dataset
        Return
            feature_importance: bar plot of the relative contribution of the features to the prediction made by the Logistic regression.
        """
    model_lr = LogisticRegression().fit(X_train, y_train)

    # Build a dataframe with the predictive power of each feature
    feature_names = list(X_train.columns)
    feature_importance = pd.DataFrame(feature_names, columns=["Feature"])
    feature_importance['Importance'] = pow(math.e, model_lr.coef_[0])
    feature_importance = feature_importance.sort_values(by=['Importance'], ascending=False)

    return feature_importance
