from time import time
from utils import ignore_warnings, formatETATime
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from preprocessing_permutations import all_combinations
from preprocessing import prepare_data
from classifier_testing import test_classifier

ignore_warnings()

print("Number of combinations:", len(all_combinations))
input("Press Enter to continue...\n")

def test_classifiers(model, combinations, iterations=1, threads=1, verbose=False, apply_threshold=False):
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
        try:
            df = prepare_data(args=comb, ret=True)
            bankrupt_column = df['X65']
            df = df.drop('X65', axis=1)
            results = test_classifier(model, df, bankrupt_column, iterations=iterations, threads=threads, verbose=verbose, threshold=threshold, ptd_args=comb)
            
            mean = results.mean(axis=0)
            std = results.std(axis=0)
            combinationRanking.append((comb, mean, std))
            print('Mean F1-score: ', mean, '±', std, '\n')
            
            lapsed_time += time() - t1
            if apply_threshold:
                cur_threshold = (mean[1] - std[1]) * 0.75
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


model = lambda: HistGradientBoostingClassifier(class_weight='balanced', max_iter=200, max_depth=20, early_stopping=False, learning_rate=0.2, random_state=420)
# model = lambda: RandomForestClassifier(
#     max_depth=20, 
#     n_estimators=100,
#     min_samples_split=5, 
#     min_samples_leaf=2, 
#     max_features='sqrt', 
#     random_state=0,   
#     class_weight='balanced'
# )


test_classifiers(model, all_combinations, iterations=1, threads=1, verbose=True, apply_threshold=False)