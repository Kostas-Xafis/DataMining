from time import time
from utils import ignore_warnings, formatETATime
from sklearn.ensemble import HistGradientBoostingClassifier
from itertools import product
from preprocessing import prepare_data
from classifier_testing import test_classifier


ignore_warnings()

arg_permutations = {
    "rm_empty": [{'threshold': [0.1]}, 'off'],
    "smote": ['off', 'on'],
    "fill_nan": [{'method': ['mean', 'zero']}],
    "binning": ['off', {'threshold': [0.1], 'bins': [3]}],
    "high_zero": ['off', {'threshold': [0.1]}],
    "correlation": ['off', {'threshold': [0.999, 0.99], 'method': ['pearson']}],
    "outliers": ['off', {'threshold': [2.5, 3.0, 3.5]}],
    "deskew": ['off', 'on'],
    "normalize": ['off', {'method': ['z-score', 'minmax']}],
}


# Helper function to expand the inner lists
def expand_options(options):
    if isinstance(options, list):
        expanded = []
        for opt in options:
            if isinstance(opt, dict):
                # Convert lists in the dictionary to individual combinations
                keys, values = zip(*[
                    (k, v if isinstance(v, list) else [v])
                    for k, v in opt.items()
                ])
                for combination in product(*values):
                    expanded.append(dict(zip(keys, combination)))
            else:
                expanded.append(opt)
        return expanded
    return options

# Prepare the expanded arguments
expanded_args = {key: expand_options(value) for key, value in arg_permutations.items()}

# Generate all combinations
keys, values = zip(*expanded_args.items())
all_combinations = [dict(zip(keys, combination)) for combination in product(*values)]

print("Number of combinations:", len(all_combinations))
input("Press Enter to continue...\n")

def test_classifiers(model, combinations, iterations=1, threads=1, verbose=False):
    combinationRanking = []
    lapsed_time = 0
    comb_len = len(combinations)
    threshold = 0 # Variable to discard any combination that doesn't meet it
    for i in range(0, comb_len):
        t1 = time()
        if lapsed_time != 0:
            print("ETA: ", formatETATime((comb_len - i) * lapsed_time / (i + 1)), '\n')
        print("Combination: ", i + 1, "/", comb_len, ':', combinations[i])
        comb = combinations[i]
        smote = comb['smote'] == 'on'
        try:
            df = prepare_data(comb, ret=True)
            bancrupt_column = df['X65']
            df = df.drop('X65', axis=1)
            results = test_classifier(model, df, bancrupt_column, iterations=iterations, threads=threads, verbose=verbose, smote=smote, threshold=threshold)
            
            mean = results.mean(axis=0)
            std = results.std(axis=0)
            combinationRanking.append((comb, mean, std))
            print('Mean F1-score: ', mean, '±', std, '\n')
            
            lapsed_time += time() - t1

            cur_threshold = (mean[1] - std[1]) * 0.9
            if cur_threshold > threshold:
                threshold = cur_threshold
        except Exception as e:
            if 'Test stop' == str(e):
                print('Test was forcefully stopped')
            elif 'Stop process' == str(e):
                print('Process was stopped')
                break
            elif 'Threshold reached' == str(e):
                print('Combination reached threshold')
            else:
                print("Failed to test combination: ", comb)
                print(e)
    combinationRanking.sort(key=lambda x: x[1][1], reverse=True)

    n = min(10, len(combinationRanking))
    print(f"Top {n} combinations:")
    for i in range(0, n):
        print(combinationRanking[i][0])
        print(combinationRanking[i][1][1], "±", combinationRanking[i][2][1])
        print('\n')

    # Store to file
    with open('combination_ranking.txt', 'w') as f:
        for comb in combinationRanking:
            f.write(str(comb[0]) + '\n')
            f.write('Mean: ' + str(comb[1]) + '\n')
            f.write('Std: ' + str(comb[2]) + '\n')
            f.write('\n')
    f.close()


model = lambda: HistGradientBoostingClassifier(class_weight='balanced', max_depth=10, max_iter=100, early_stopping=False, random_state=0)

test_classifiers(model, all_combinations, iterations=30, threads=8, verbose=True)
