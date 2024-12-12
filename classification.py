from preprocessing import prepare_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from classifier_testing import test_classifier

data_prep_args = {
    'rm_empty': 'off',
    'fill_nan': {'method': 'zero'},
    'binning': 'off',
    'high_zero': 'off',
    'correlation': 'off',
    'outliers': {'threshold': 3},
    'deskew': 'off',
    'normalize': 'off'
}


df = prepare_data(data_prep_args, ret=True)
bancrupt_column = df['X65']
df = df.drop('X65', axis=1)

def classifier_test(df, target, iterations=10, threads=1, verbose=True, smote=False):
    # classifierLamda = lambda: AdaBoostClassifier(DecisionTreeClassifier(max_depth=50), n_estimators=100, random_state=0)
    # classifierLamda = lambda: GradientBoostingClassifier(loss='log_loss', max_depth=10, n_estimators=100, random_state=0)
    classifierLamda = lambda: HistGradientBoostingClassifier(class_weight='balanced', max_iter=100, max_depth=10, early_stopping=False)
    # classifierLamda = lambda: RandomForestClassifier(max_depth=75, n_estimators=50, random_state=0)
    results = test_classifier(classifierLamda, df, target, iterations=iterations, threads=threads, verbose=verbose, smote=smote, sample_size=0.1)
    print("Model: ", classifierLamda().__class__.__name__, "\n\tMean F1-Score: ", results.mean(axis=0), "\n\tStandard Deviation: ", results.std(axis=0)) if verbose else None

def simple_test(df, target):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, shuffle=True)

    model = HistGradientBoostingClassifier(class_weight='balanced', max_iter=100, max_depth=5, early_stopping=False, random_state=0)

    # Train the model
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print('\n')
    print(classification_report(y_test, predictions))



classifier_test(df, bancrupt_column, iterations=10, threads=8)
# simple_test(df, bancrupt_column)