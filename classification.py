import pandas as pd
import numpy as np
from optims.best_classifier import best_model
from utils.parse_args import parse_args
from preprocessing import get_training_data, get_testing_data, prepare_training_data
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from optims.classifier_testing import test_classifier
from sklearn.inspection import permutation_importance


data_prep_args = {
    'rm_empty': 'off',
    'smote': 'on',
    'fill_nan': {'method': 'mean'},
    'correlation': {'threshold': 0.999, 'method': 'kendall'},
    'outliers': {'threshold': 3 },
    'normalize': {'method': 'robust'},
}

# Best with the unlabeled data 
data_prep_args = {'rm_empty': 'off', 'smote': 'on', 'fill_nan': {'method': 'median'}, 'correlation': 'off', 'outliers': {'threshold': 3}, 'normalize': {'method': 'robust'}}

# rfLambda = lambda: RandomForestClassifier(max_depth=20, n_estimators=100, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=0, class_weight='balanced')

def getTrainingData():
    global data_prep_args
    df = get_training_data(args=data_prep_args, ret=True, df=None)
    target = df['X65']
    df = df.drop('X65', axis=1)
    return df, target

def getUnlabeledData(labels=None):
    global data_prep_args
    return get_testing_data(labels, args=data_prep_args, ret=True)



def classifier_test(iterations=10, threads=1, verbose=True, ptd_args=None):
    df, target = getTrainingData()

    classifierLamda = lambda: HistGradientBoostingClassifier(class_weight='balanced', max_iter=200, max_depth=20, early_stopping=False, learning_rate=0.2, l2_regularization=0.2)
    # classifierLamda = lambda: RandomForestClassifier(max_depth=20, n_estimators=100, min_samples_split=2, max_features='log2', random_state=0, class_weight='balanced')

    results = test_classifier(classifierLamda, df, target, iterations=iterations, threads=threads, verbose=verbose, ptd_args=ptd_args)
    print("Model: ", classifierLamda().__class__.__name__, "\n\tMean F1-Score: ", results.mean(axis=0), "\n\tStandard Deviation: ", results.std(axis=0)) if verbose else None

def evaluate_model(model, fold_size=3, repeats=10):
    X, y = getTrainingData()
    X, y = X.values, y.values
    rskf = RepeatedStratifiedKFold(n_splits=fold_size, n_repeats=repeats, random_state=42)
    accuracies = []
    f1_scores = []    

    for train_index, test_index in rskf.split(X, y):
        X_train, y_train, = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        X_train, y_train = prepare_training_data(X_train, y_train, data_prep_args)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print("Accuracy: ", acc)
        print("F1: ", f1)
        
        accuracies.append(acc)
        f1_scores.append(f1)

    print(f"Mean Accuracy: {float(np.mean(accuracies)):.4f}")
    print(f"Mean F1 Score: {float(np.mean(f1_scores)):.4f}")
    print(f"Accuracy Scores: {list(map(float, accuracies))}")
    print(f"F1 Scores: {list(map(float, f1_scores))}")

def simple_test():
    df, target = getTrainingData()

    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, shuffle=True, random_state=420)

    model = HistGradientBoostingClassifier(class_weight='balanced', max_iter=150, max_depth=20, early_stopping=False, learning_rate=0.2, l2_regularization=0.1)
    # model = RandomForestClassifier(max_depth=20, n_estimators=100, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=0, class_weight='balanced')
    X_train, y_train = prepare_training_data(X_train, y_train, data_prep_args)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print('\n')
    print(classification_report(y_test, y_pred))    
    
def grid_search():
    df, target = getTrainingData()

    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, shuffle=True, random_state=420)

    # param_grid = { 'n_estimators': [100, 150], 'max_depth': [10, 20], 'min_samples_split': [2, 5, 10, 15], 'max_features': ['sqrt'], 'class_weight': ['balanced'] }
    # model = GridSearchCV(RandomForestClassifier(random_state=420), param_grid, cv=5, n_jobs=6)
    
    param_grid = { 'max_iter': [100, 150, 200], 'max_depth': [10, 20], 'early_stopping': [False, True], 'learning_rate': [0.1, 0.2], 'l2_regularization': [None, 0.1, 0.2], 'class_weight': [None, 'balanced'] }
    model = GridSearchCV(HistGradientBoostingClassifier(random_state=420), param_grid, cv=5, n_jobs=6)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    best_params = model.best_params_
    best_rf = model.best_estimator_
    
    print("Best Parameters: ", best_params)
    print("Best Random Forest: ", best_rf)

    y_pred = best_rf.predict(X_test)

    accuracy = best_rf.score(X_test, y_test)
    print("Accuracy: ", accuracy)

    print(confusion_matrix(y_test, y_pred))
    print('\n')
    print(classification_report(y_test, y_pred))

def test_with_top10_features():
    df, target = getTrainingData()

    # Initialize the model
    model = HistGradientBoostingClassifier(class_weight='balanced', max_iter=200, max_depth=20, early_stopping=False, learning_rate=0.2, l2_regularization=0.2)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, shuffle=True)

    model.fit(X_train, y_train)

    # Select the top 10 features
    importance = permutation_importance(model, X_train, y_train, n_repeats=50, n_jobs=-1)

    print("Feature Importance: ", importance.importances_mean)

    # Filter the array if the importance is greater than 0
    topFeatures = [i for i in range(0, len(importance.importances_mean)) if importance.importances_mean[i] > 0]

    print("Top Features: ", X_train.columns[topFeatures])

    X_train = X_train.iloc[:, topFeatures]
    X_test = X_test.iloc[:, topFeatures]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print('\n')
    print(classification_report(y_test, y_pred))

def test_unlabeled_data():
    X_train, y_train = getTrainingData()
    X_test = getUnlabeledData(labels=X_train.columns)

    model = best_model

    # Preprocess training data
    X_train, y_train = prepare_training_data(X_train, y_train, data_prep_args)

    model.fit(X_train, y_train)

    # Save the predictions
    predictions = model.predict(X_test)
    print("Predictions: ", sum(predictions))
    results_df = pd.Series(predictions)
    results_df.to_csv('./data/unlabeled_predictions.csv', index=False, header=False)
    print("Unlabeled predictions saved to ./data/unlabeled_predictions.csv")
    
    # Save the top 50 most likely bankrupt companies
    probabilities = model.predict_proba(X_test)[:, 1]

    top_50_indices = probabilities.argsort()[-50:][::-1]

    results_df = pd.Series(top_50_indices + 1)

    results_df.to_csv('./data/top_50_predictions.csv', index=False, header=False)
    print("Top 50 predictions saved to ./data/top_50_predictions.csv")

args = parse_args({
    'simple': [bool, False],
    'grid_search': [bool, False],
    'classifier': [bool, False],
    'top10': [bool, False],
    'unlabeled': [bool, False]
})


if __name__ == '__main__':
    if args['simple']:
        simple_test()
    elif args['grid_search']:
        grid_search()
    elif args['classifier']:
        classifier_test(iterations=60, threads=12, verbose=True, ptd_args=data_prep_args)
        # evaluate_model(best_model, fold_size=10, repeats=10)
    elif args['unlabeled']:
        test_unlabeled_data()
    elif args['top10']:
        test_with_top10_features()
    else:
        print("No arguments provided. Exiting...")