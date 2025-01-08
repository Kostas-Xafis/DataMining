import threading
import time
import numpy as np
from utils import flattenUnevenArray, TestEnv
from preprocessing import prepare_training_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

def threshold_test(model, tts, threshold=1, ptd_args=None):
    global testEnv
    # Test the classifier first with 1 thread to check if it goes below the threshold
    thresh_thread = threading.Thread(target=train_and_evaluate, args=(model, tts, [[None]], 0, 1, testEnv, threshold, ptd_args,))
    thresh_thread.start()
    thresh_thread.join()
    testEnv.reset()

def train_and_evaluate(model, tts, f1_scores, thread_id, iters, testEnv, threshold=1, ptd_args=None):
    model_instance = None
    try:
        model_instance = model()
        for i in range(0, iters):
            if testEnv['force_stop'] == True:
                break
            while testEnv['pause']:
                time.sleep(1)
            X_train, X_test, y_train, y_test = tts()
            X_train, y_train = prepare_training_data(X_train, y_train, ptd_args)

            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)

            report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)

            f1_scores[thread_id][i] = [report['0']['f1-score'], report['1']['f1-score']]
            testEnv['iterations'] += 1
            if f1_scores[thread_id][i][1] <= threshold:
                testEnv['force_stop'] = True
    except Exception as e:
        print("Classifier: ", model_instance.__class__.__name__, " failed to train")
        print(e)
        testEnv['force_stop'] = True

def test_classifier(model, df, target, iterations=10, test_size=0.2, threads=1, verbose=True, threshold=0, ptd_args=None):
    global testEnv

    init_test_pause_thread()

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
    # udf, utarget = prepare_unlabeled_data(df.columns, ptd_args, ret=True)
    # tts = lambda: (df, udf, target, utarget)
    
    # Test threshold
    if threshold > 0:
        threshold_test(model, tts, threshold, ptd_args)
        if testEnv['force_stop']:
            raise Exception('Threshold reached')


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
        thread = threading.Thread(target=train_and_evaluate, args=(model, tts, f1_scores, i, iters, testEnv, threshold, ptd_args))
        threads_list.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads_list:
        thread.join()

    if testEnv['force_stop']:
        raise Exception('Threshold reached')

    return np.array(flattenUnevenArray(f1_scores, 1))

__all__ = ['get_all_classifiers_sklearn', 'test_all_classifiers', 'test_classifier']