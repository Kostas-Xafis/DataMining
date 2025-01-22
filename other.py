import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def get_labeled_test():
    df = pd.read_csv('./data/test_labeled.csv', na_values=['?', ''])
    
    target = df['X65']
    df = df.drop('X65', axis=1)
    return df, target

def predictions_test():
    data, target = get_labeled_test()
    target = target.isnull().replace(0)
    print(target)
    predictions = pd.read_csv('./data/unlabeled_predictions.csv', header=None)

    print(confusion_matrix(target, predictions))
    print('\n')
    print(classification_report(target, predictions))

if __name__ == '__main__':
    predictions_test()