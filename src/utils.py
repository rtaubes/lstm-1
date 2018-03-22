"""
    Helpers
"""

import time
from functools import wraps

class TimeRec:
    """
        Calculate time of excecution. How to use it:
        tm = TimeRec()  # (1)
        ...
        exec_time = tm.step()  # (2) exec_time is time from (1) to (2)
        ...
        exec_time2 = tm.step()  # (3) exec_time is time from (2) to (3)
        ...
        tot_time = tm.total()  # (4) exec time from (4) to (1)
        print(tm) # print total time
    """
    def __init__(self):
        self._time0 = time.monotonic()
        self._last = self._time0

    def step(self):
        delta = time.monotonic() - self._last
        self._last = time.monotonic()
        return delta

    def total(self):
        return time.monotonic() - self._time0

    def __str__(self):
        return 'Total: {:.3f} sec'.format(self.total())


def frozen_cls(cls):
    """ Well-known solution how to prevent addition of new attributes
        to a class without using 'slot'.
        This is a decorator and should be used like this:
        @frozen_cls
        class Klass:
            ...
        cls = Klass()
        cls.x = 'X'  # it is possible only for members defined in class.
    """
    cls.__frozen = False

    def frozen_attr(self, key, value):
        if not hasattr(self, key) and self.__frozen:
            raise AttributeError("Could not set new attribute '{}' "
                               "for frozen class '{}'".format(key, cls.__name__))
        else:
            object.__setattr__(self, key, value)

    def init_wrapper(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.__frozen = True
        return wrapper

    cls.__setattr__ = frozen_attr
    cls.__init__ = init_wrapper(cls.__init__)

    return cls