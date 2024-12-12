import warnings
import threading
from time import sleep
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

def ignore_warnings():
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
    warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    warnings.filterwarnings(action='ignore', category=FutureWarning)

positive_infinity = float('inf')
def flattenUnevenArray(arr, depth=positive_infinity):
    newArr = []
    for sublist in arr:
        if isinstance(sublist, list) and depth > 0:
            newArr.extend(flattenUnevenArray(sublist, depth - 1))
        else:
            newArr.append(sublist)
    return newArr

def formatETATime(n):
    if n is None or n < 0:
        return "N/A"
    if n is positive_infinity:
        return "+∞ s"


    if n < 60:
        return str(n // 1) +  "s"
    elif n < 3600:
        return str(n // 60) + "m " + formatETATime(n % 60)
    elif n < 86400:
        return str(n // 3600) + "h" + formatETATime(n % 3600)
    else:
        return str(n // 86400) + "d" + formatETATime(n % 86400)


class Atomic:
    """A class that provides atomic/thread safe access to a value."""
    def __init__(self, initial_value=None):
        self.value = initial_value
        self._lock = threading.Lock()

    def getLock(self):
        return self._lock

    def get(self):
        with self._lock:
            return self.value

    def set(self, new_value):
        with self._lock:
            self.value = new_value

class TestEnv:
    def __init__(self):
        self.pause = Atomic(False)
        self.force_stop = Atomic(False)
        self.status = Atomic('Running')
        self.iterations = Atomic(0)

    def reset(self):
        self.pause.set(False)
        self.force_stop.set(False)
        self.status.set('Running')
        self.iterations.set(0)

    def getAll(self):
        return {
            'pause': self.pause,
            'force_stop': self.force_stop,
            'status': self.status,
            'iterations': self.iterations
        }
    
    def __getitem__(self, key):
        match key:
            case 'pause':
                return self.pause.get()
            case 'force_stop':
                return self.force_stop.get()
            case 'status':
                return self.status.get()
            case 'iterations':
                return self.iterations.get()   

    def __setitem__(self, key, value):
        match key:
            case 'pause':
                self.pause.set(value)
            case 'force_stop':
                self.force_stop.set(value)
            case 'status':
                self.status.set(value)
            case 'iterations':
                self.iterations.set(value)

__atomic__ = Atomic()
__test__ = TestEnv()

__all__ = ['ignore_warnings', 'flattenUnevenArray', '__atomic__', 'formatETATime', '__test__'] 


