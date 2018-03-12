"""
    Generators that works as iterators and can be used in a 'while' loop
"""

import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def _noise(ampl, size):
    return np.random.normal(loc=0.0, scale=ampl / 2, size=size)


def _sin_gen(ampl, period_steps, offset, size):
    """ Generate a noisy sin sequence """
    x_step = np.pi * 2 / period_steps
    xarg = [ i % period_steps * x_step for i in range(size)]
    x_sin = np.sin(np.array(xarg))
    return x_sin


def _lin_gen(step_ampl, size):
    return np.array([i * step_ampl for i in range(size)])


def _pandas_gen(df, val_col, date_col=None):
    """ Generate data Pandas DataFrame column names """
    if val_col not in df.columns:
        raise ValueError("value column '{}' not in columns '{}'"
                         .format(val_col, df.columns))
    if date_col is not None and date_col not in df.columns:
        raise ValueError("date column '{}' not in columns '{}'"
                         .format(date_col, df.columns))
    if date_col is None:
        date_vals = None
    else:
        date_vals = df[date_col].values
    return df[val_col].values, date_vals


def csv_reader(fname, val_col, date_col):
    """ Generate data from CSV file using column name """
    def csv_date(x):
        return pd.datetime.fromtimestamp(float(x))
    df = pd.read_csv(fname, parse_dates=True, date_parser=csv_date)
    return _pandas_gen(df, val_col, date_col)


def _pulse_gen(pulse_width, period, size):
    """ Generate pulse sequence """
    data = np.zeros(size, dtype=np.float)
    for p_idx in range(pulse_width):
        idx = 0
        while idx+p_idx < size-1:
            data[idx+p_idx] = 1
            idx += period
    return data


class BatchGenerator:
    """ Generate batches for using in a 'while' cycle.
        Using 'align_size'=True, data size will be aligned to the whole number of batches by
        removing elements from beginning. This value may be reassigned to '1' or to a value
        which didn't change the data size. Otherwise a ValueError will be raised.
        Number of output features is always 1 and this is an 'y'.
        Assume that x is a sequence 'x0 x1 x2 x3 x4 x5 x6'
        The example of batches X and Y with batch_size=4, x_steps=3, y_steps=3
        X:               Y:
        [x0, x1, x2] [x1, x2, x3]
        [x1, x2, x3] [x2, x3, x4]
        [x2, x3, x4] [x3, x4, x5]
        [x3, x4, x5] [x4, x5, x6]
        number of features is 1.
        An output batch shape is (4, 3, 1)
        The example of batches X and Y with batch_size=3, x_steps=3, y_steps=1
        X:               Y:
        [x0, x1, x2] [x3]
        [x1, x2, x3] [x4]
        [x2, x3, x4] [x5]
        number of features is 1.
        An output batch shape is (4, 3, 1)
        Note that number of columns for Y depends on a neural network structure and usually
        either x_steps or 1.
    """
    def __init__(self, data=None, dates=None, batch_size=1, x_steps=1, y_steps=1, noise_std=0,
                 align_size=True):
        """
            :param data: data as numpy array
            :param dates: np.array of dates for values
            :param batch_size: size of batch.
            :param x_steps: X steps in a batch row. The second dimension for x_batch
            :param y_steps: Y steps in a batch row. The second dimension for y_batch
            :param noise_std: std of added noise.
            :param align_size: align size to the whole number of batches
        """
        if data is None:
            raise ValueError("Could not make BatchGenerator with data as None")
        if isinstance(data, list):
            data = np.array(data)
        data = data.reshape(-1, 1).astype(np.float)
        rand_add = np.random.normal(loc=0.0, scale=noise_std, size=len(data)).reshape(-1, 1)
        self._data = data + rand_add
        if dates is not None:
            if isinstance(dates, list):
                self._dates = np.array(dates)
            self._dates = dates.reshape(-1, 1)
        else:
            self._dates = np.array([np.nan] * len(data)).reshape(-1, 1)
        self._size = len(data)
        self._raw_data = None  # save original data if they are scaled
        self._batch_generator = None
        self._idx0 = 0
        self._offset = 0
        self._x_steps = x_steps
        self._y_steps = y_steps
        self._batch_size = batch_size
        if align_size:
            new_size = self._align_data(batch_size)
            self._data = self._data[-new_size:]
            self._dates = self._dates[-new_size:]
            self._size = new_size
        self._scaler = None
        self._align_size = align_size

    def append_elem(self, elem, elem_date=np.nan):
        """ append data element to historical data """
        self._data = np.roll(self._data, -1)
        if self._scaler:
            self._raw_data = np.roll(self._raw_data, -1)
            self._raw_data[-1] = elem
            np_elem = self._scaler.transform(np.array([[elem]]))
            self._data[-1] = np_elem
        else:
            self._data[-1] = elem
        if self._dates is not None:
            self._dates = np.roll(self._dates, -1)
            self._dates[-1] = elem_date
            self._dates = self._dates[1:]
        self._idx0 = max(self._idx0-1, 0)

    def _align_data(self, batch_size):
        ssz = self._size - self._x_steps
        return ssz - ssz % batch_size + self._x_steps

    def reset(self):
        """ Reset values """
        self._idx0 = 0

    @property
    def batch_size(self):
      return self._batch_size
    
    @batch_size.setter
    def batch_size(self, batch_size):
        if self._align_size:
            new_size = self._align_data(batch_size)
            if new_size != self._size:
                raise ValueError("Could not use batch_size {} for aligned data,"
                                 "because it realigned the current data size {} to {}"
                                 .format(batch_size, self._size, new_size))
        self._batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        y_bg = self._idx0 + self._x_steps + 1 - self._y_steps
        last_idx = self._idx0 + self._x_steps + self._batch_size - 1
        # The last batch must have full length.
        if last_idx >= self._size:
            raise StopIteration()
        batches = []
        for cnt in range(self._batch_size):
            row = self._data[self._idx0 : self._idx0 + self._x_steps + 1]
            self._idx0 += 1
            batches.append(row)
        np_batch = np.array(batches)
        x_batch = np_batch[:, :-1].reshape(-1, self._x_steps, 1)
        y_batch = np_batch[:, -self._y_steps:].reshape(-1, self._y_steps, 1)
        return y_bg, x_batch, y_batch

    def scale(self, range=None):
        """ scale data.
            :param range: a tuple(min, max) to setup min and(or) max differ than in 'data'
        """
        if self._scaler:
            raise RuntimeError("data already scaled")
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        custom_limits = False
        self._raw_data = self._data.copy()
        if range is not None:
            custom_limits = True
            self._data += range
        data_f = np.array(self._data).reshape(-1, 1).astype(np.float)
        self._data = self._scaler.fit_transform(self._data)
        if custom_limits:
            self._data = self._data[:-2]

    def unscale(self, data):
        if self._scaler is None:
            return data
        if isinstance(data, list):
            data2 = np.array(data).reshape(-1, 1)
        else:
            data2 = data
        unscaled = self._scaler.inverse_transform(data2)
        if isinstance(data, list):
            return unscaled.flatten().tolist()
        return unscaled

    @property
    def data(self):
        return self._data

    @property
    def dates(self):
        return self._dates

    @property
    def raw_data(self):
        if self._scaler:
            return self._raw_data
        return self._data

    @property
    def data_size(self):
        return self._size


class SinBatches(BatchGenerator):
    def __init__(self, sin_ampl=0.5, period_steps=10, offset=0.45,
                 size=100, batch_size=10, x_steps=1, y_steps=1, align_size=True, noise_std=0):
        data = _sin_gen(sin_ampl, period_steps, offset, size)
        super(SinBatches, self).__init__(data=data, batch_size=batch_size, x_steps=x_steps,
                                         y_steps=y_steps, noise_std=noise_std,
                                         align_size=align_size)


class LinBatches(BatchGenerator):
    def __init__(self, step_ampl=1, size=100, batch_size=10, x_steps=1, y_steps=1, align_size=True,
                 noise_std=0.0):
        data = _lin_gen(step_ampl, size)
        super().__init__(data=data, batch_size=batch_size, x_steps=x_steps, y_steps=y_steps,
                         noise_std=noise_std, align_size=align_size)


class PulseBatches(BatchGenerator):
    def __init__(self, pulse_width=1, period=30, batch_size=10, x_steps=1, y_steps=1, size=100,
                 align_size=True, noise_std=0):
        data = _pulse_gen(pulse_width, period, size)
        super().__init__(data=data, batch_size=batch_size, x_steps=x_steps, y_steps=y_steps,
                         noise_std=noise_std, align_size=align_size)

class CSVBatches(BatchGenerator):
    def __init__(self, fname, date_col, val_col, batch_size=10, x_steps=1, y_steps=1,
                 align_size=True, noise_std=0):
        data, dates = csv_reader(fname, date_col, val_col)
        super().__init__(data=data, batch_size=batch_size, x_steps=x_steps, y_steps=y_steps,
                         noise_std=noise_std, align_size=align_size)

class PandasBatches(BatchGenerator):
    def __init__(self, data_frame, val_col, date_col, batch_size=10, x_steps=1, y_steps=1,
                 align_size=True, noise_std=0):
        data, dates = _pandas_gen(data_frame, val_col=val_col, date_col=date_col)
        super().__init__(data=data, dates=dates, batch_size=batch_size,
                         x_steps=x_steps, y_steps=y_steps, noise_std=noise_std,
                         align_size=align_size)

if __name__ == '__main__':

    BATCH_SIZE = 10
    TIME_STEPS = 4
    DATA_SIZE = 40
    pb = LinBatches(step_ampl=0.01, size=DATA_SIZE, batch_size=BATCH_SIZE,
                    x_steps=TIME_STEPS, y_steps=1, align_size=True, noise_std=0.0)
    # pb = PulseBatches(pulse_width=1, period=5, batch_size=9, time_steps=64, size=1000)
    print("Data:", pb.data)
    for idx, xb, yb in pb:
        print(idx, ", xb / ", xb.shape, ":\n", xb.reshape(-1, TIME_STEPS))
        print(idx, ", yb / ", yb.shape, ":\n", yb.reshape(-1, 1), " orig data:", pb.data[idx])
    print("Got {} batches".format(idx))
    pb.append_elem(101)
    for idx, xb, yb in pb:
        print(idx, ", xb:", xb.shape)
        print(idx, ", yb:", yb.shape)
    print("Got {} batches".format(idx))
    print('data tail:', pb.data[-10:])
