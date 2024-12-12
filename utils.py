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

    def lock(self):
        self._lock.acquire(blocking=True)
        return self    
    
    def unlock(self):
        self._lock.release()
        return self

    def get(self):
        return self.value

    def set(self, new_value):
        self.lock()
        if callable(new_value):
            self.value = new_value()
        else:
            self.value = new_value
        self.unlock()

class TestEnv:
    env = {}
    initial_values = None
    def __init__(self, env_dict=None):
        if env_dict is None:
            raise ValueError("env_dict must be a dictionary")
        
        for key, value in env_dict.items():
            self.env[key] = Atomic(value)
        self.initial_values = env_dict

    def reset(self):
        for key, value in self.initial_values.items():
            self.env[key].set(value)
            
    def get(self):
        return self.env
    
    def __getitem__(self, key):
        return self.env[key].get()

    def __setitem__(self, key, value):
        self.env[key].set(value)

__atomic__ = Atomic()
__test__ = TestEnv({})

__all__ = ['ignore_warnings', 'flattenUnevenArray', '__atomic__', 'formatETATime', '__test__'] 


