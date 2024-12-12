import threading
import time
import numpy as np
from utils import flattenUnevenArray, TestEnv
from imblearn.over_sampling import SMOTE
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import all_estimators

def get_all_classifiers_sklearn():
    estimators = all_estimators(type_filter='classifier')
    classifiers = DataFrame(columns=['classifier', 'name'])
    print("Number of classifiers: ", len(estimators))
    print(classifiers)
    for name, ClassifierClass in estimators:
        try:
            if name == 'GaussianProcessClassifier':
                continue
            classifiers.add_row([ClassifierClass, name])
        except Exception as e:
            pass
    print("Number of classifiers that can be used: ", classifiers.shape)
    return classifiers

def test_all_classifiers(df, target, iterations=1, test_size=0.2):
    datasets = []
    results = []
    for i in range(0, iterations):
        try:
            datasets.append(train_test_split(df, target, test_size=test_size, shuffle=True))
        except Exception as e:
            print("Failed to split the dataset. Attempting again...")

    classifiers = get_all_classifiers_sklearn()
    classifier_names = classifiers['name']
    classifiers = classifiers['classifier']
    for i in range(0, classifiers.shape[0]):
        avg_f1 = np.array([])
        try:
            model = classifiers[i]()
            once: bool = False
            for X_train, X_test, y_train, y_test in datasets:
                if (not once):
                    modelName = model.__class__.__name__
                    print("Training", modelName)
                    once = True
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                report = classification_report(y_test, predictions, output_dict=True, zero_division=1)
                avg_f1 = np.append(avg_f1, [[report['0']['f1-score'], report['1']['f1-score']]])
            
            results.append((classifier_names[i], avg_f1.mean()))
        except Exception as e:
            print("Classifier: ", classifier_names[i], "failed to train")
            print(e)

testEnv = TestEnv({
    'pause': False,
    'force_stop': False,
    'status': 'Running',
    'iterations': 0
})
def test_pause(testEnv):
    while testEnv['status'] != 'Finished':
        try:
            inp = input("[C]ontinue/[P]ause/[S]top: ")
            if inp == 'P':
                testEnv['pause'] = True
                testEnv['status'] = 'Paused'
            elif inp == 'C':
                testEnv['pause'] = False
                testEnv['status'] = 'Running'
            elif inp == 'S':
                testEnv['force_stop'] = True
                raise Exception('Test stop')
            else:
                print("Incorrect command...")
        except Exception as e:
            if 'Test stop' == str(e):
                raise e
            break

test_pause_thread = None
def init_test_pause_thread():
    global testEnv
    global test_pause_thread
    if test_pause_thread is not None:
        testEnv.reset()
    else:
        test_pause_thread = threading.Thread(target=test_pause, args=(testEnv,))
        test_pause_thread.start()

def test_classifier(model, df, target, iterations=10, test_size=0.2, threads=1, verbose=True, smote=False, threshold=1):
    global testEnv

    init_test_pause_thread()

    # if sample_size < 1.0:
    #     tCol = 'X65'
    #     df = pd.concat([df, target], axis=1).sample(frac=sample_size).reset_index(drop=True)
    #     target = df[tCol]
    #     df = df.drop(tCol, axis=1)


    # Define the functions to be used in the threads
    def track_progress(totalIterations, testEnv):
        prev_total = 0
        while testEnv['force_stop'] == False:
            cur_total = testEnv['iterations']
            if prev_total != cur_total:
                prev_total = cur_total
                print("Progress: ", cur_total, "/", totalIterations)
                if cur_total == totalIterations:
                    break

            if testEnv['pause']:
                while testEnv['pause']:
                    time.sleep(1)
            else: 
                time.sleep(1)

    tts = lambda: train_test_split(df, target, test_size=test_size, shuffle=True)

    def train_and_evaluate(model, tts, f1_scores, thread_id, iters, testEnv):
        model_instance = None
        try:
            model_instance = model()
            for i in range(0, iters):
                if testEnv['force_stop'] == True:
                    break
                while testEnv['pause']:
                    time.sleep(1)
                X_train, X_test, y_train, y_test = tts()
                if smote:
                    X_train, y_train = SMOTE(sampling_strategy='minority', random_state=39).fit_resample(X_train, y_train)

                model_instance.fit(X_train, y_train)

                report = classification_report(y_test, model_instance.predict(X_test), output_dict=True, zero_division=1)

                f1_scores[thread_id][i] = [report['0']['f1-score'], report['1']['f1-score']]
                testEnv['iterations'] += 1
                if f1_scores[thread_id][i][1] <= threshold:
                    testEnv['force_stop'] = True
        except Exception as e:
            print("Classifier: ", model_instance.__class__.__name__, " failed to train")
            testEnv['force_stop'] = True

    # Initialize thread variables
    threads_list = []
    f1_scores = [None] * threads
    chunk_size = iterations // threads
    remaining = iterations % threads

    # Start the progress tracker
    if verbose:
        progress_thread = threading.Thread(target=track_progress, args=(iterations, testEnv,))
        progress_thread.start()
        threads_list.append(progress_thread)
        testEnv['status'] = 'Running'


    for i in range(threads):
        iters = (chunk_size + 1) if i < remaining else chunk_size
        f1_scores[i] = [np.nan] * iters
        thread = threading.Thread(target=train_and_evaluate, args=(model, tts, f1_scores, i, iters, testEnv,))
        threads_list.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads_list:
        thread.join()

    if testEnv['force_stop']:
        raise Exception('Threshold reached')
    
    return np.array(flattenUnevenArray(f1_scores, 1))


__all__ = ['get_all_classifiers_sklearn', 'test_all_classifiers', 'test_classifier']