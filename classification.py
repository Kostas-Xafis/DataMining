import pandas as pd
from parse_args import parse_args
from preprocessing import prepare_data, prepare_unlabeled_data, prepare_training_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from classifier_testing import test_classifier
from sklearn.inspection import permutation_importance

data_prep_args = {
    'rm_empty': {'threshold': 0.1},
    'smote': 'on',
    'fill_nan': {'method': 'mean'},
    'correlation': {'threshold': 0.999, 'method': 'kendall'},
    'outliers': {'threshold': 2.75 },
    'deskew': 'off',
    'normalize': {'method': 'robust'},
}

# Best with the unlabeled data 
data_prep_args = {'rm_empty': 'off', 'smote': 'on', 'fill_nan': {'method': 'median'}, 'correlation': 'off', 'outliers': {'threshold': 3}, 'deskew': 'off', 'normalize': {'method': 'robust'}}

def getData():
    global data_prep_args
    df = prepare_data(args=data_prep_args, ret=True, df=None)
    target = df['X65']
    df = df.drop('X65', axis=1)
    return df, target

def getunlabeledData(labels):
    global data_prep_args
    return prepare_unlabeled_data(labels, args=data_prep_args, ret=True)

histLambda = lambda: HistGradientBoostingClassifier(class_weight='balanced', max_iter=150, max_depth=20, early_stopping=False, learning_rate=0.2, l2_regularization=0.2)
rfLambda = lambda: RandomForestClassifier(max_depth=20, n_estimators=100, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=0, class_weight='balanced')

def classifier_test(iterations=10, threads=1, verbose=True, ptd_args=None):
    df, target = getData()

    classifierLamda = lambda: HistGradientBoostingClassifier(class_weight='balanced', max_iter=200, max_depth=20, early_stopping=False, learning_rate=0.2, l2_regularization=0.2)
    # classifierLamda = lambda: RandomForestClassifier(max_depth=20, n_estimators=100, min_samples_split=2, max_features='log2', random_state=0, class_weight='balanced')
    results = test_classifier(classifierLamda, df, target, iterations=iterations, threads=threads, verbose=verbose, ptd_args=ptd_args)
    print("Model: ", classifierLamda().__class__.__name__, "\n\tMean F1-Score: ", results.mean(axis=0), "\n\tStandard Deviation: ", results.std(axis=0)) if verbose else None

def simple_test():
    df, target = getData()

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
    df, target = getData()

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
    df, target = getData()

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
    X_train, y_train = getData()
    X_test = getunlabeledData(labels=X_train.columns)

    model = HistGradientBoostingClassifier(
        class_weight='balanced',
        max_iter=200,
        max_depth=20,
        early_stopping=False,
        learning_rate=0.2,
        l2_regularization=0.2
    )

    # Preprocess training data
    X_train, y_train = prepare_training_data(X_train, y_train, data_prep_args)

    model.fit(X_train, y_train)

    # Save the predictions
    predictions = model.predict(X_test)
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
        classifier_test(iterations=100, threads=12, verbose=True, ptd_args=data_prep_args)
    elif args['unlabeled']:
        test_unlabeled_data()
    elif args['top10']:
        test_with_top10_features()
    else:
        print("No arguments provided. Exiting...")