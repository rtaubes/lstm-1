#!/usr/bin/env python
"""
    Unittests for class DataSet
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime as dt
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# print(sys.path)

import data_generator

BATCH_SIZE = 4
X_STEPS = 3
Y_FEATURES = 1
ORIG_DATA_SIZE = 41
ALIGNED_DATA_SIZE = 40
N_BATCHES = 9

DFRAME = pd.DataFrame({'Vals': [i for i in range(ORIG_DATA_SIZE)],
                       'Dt': [1.0 + i/10.0 for i in range(ORIG_DATA_SIZE)]})

DFRAME_VALS = DFRAME.Vals.values.reshape(-1, 1)
DFRAME_DATES = DFRAME.Dt.values.reshape(-1, 1)

mms = MinMaxScaler(feature_range=(0, 1))
X_SCALED = mms.fit_transform(DFRAME['Vals'].values.reshape(-1, 1).astype(np.float))

pb = data_generator.PandasBatches(
    data_frame=DFRAME, val_col="Vals", date_col="Dt",
    batch_size=BATCH_SIZE, x_steps=X_STEPS, y_features=Y_FEATURES,
    align_size=True, noise_std=0)

XAB = [
    1,  2,  3,   2,  3,  4,   3, 4, 5,     4,  5,  6,   5,  6,  7,
    6,  7,  8,   7,  8,  9,   8, 9, 10,    9,  10, 11,  10, 11, 12,
    11, 12, 13,  12, 13, 14,  13, 14, 15,  14, 15, 16,  15, 16, 17,
    16, 17, 18,  17, 18, 19,  18, 19, 20,  19, 20, 21,  20, 21, 22,
    21, 22, 23,  22, 23, 24,  23, 24, 25,  24, 25, 26,  25, 26, 27,
    26, 27, 28,  27, 28, 29,  28, 29, 30,  29, 30, 31,  30, 31, 32,
    31, 32, 33,  32, 33, 34,  33, 34, 35,  34, 35, 36,  35, 36, 37,
    36, 37, 38
]

X_ALIGN_BATCHES = np.array(XAB).reshape(N_BATCHES, BATCH_SIZE, X_STEPS, 1)

YAB_STEPS = [
    4,  5,  6,  7,  8,
    9,  10, 11, 12, 13,
    14, 15, 16, 17, 18,
    19, 20, 21, 22, 23,
    24, 25, 26, 27, 28,
    29, 30, 31, 32, 33,
    34, 35, 36, 37, 38, 39
]

Y_ALIGN_BATCHES_STEPS = np.array(YAB_STEPS).reshape(N_BATCHES, BATCH_SIZE, Y_FEATURES)

YAB_1 = np.linspace(4, 39, 36)

# Y_ALIGN_BATCHES_1 = YAB_1.reshape(N_BATCHES, BATCH_SIZE, Y_FEATURES)

# Y_1_IDX = [3, 8, 13, 18, 23, 28, 33]

LAST_X_BATCH_1 = np.array([37, 38, 39]).reshape(1, -1, 1)

LAST_X_BATCH_2 = np.array([[36, 37, 38], [37, 38, 39]]).reshape(2, -1, 1)


def make_aligned(y_features=Y_FEATURES):
    pb = data_generator.PandasBatches(
        data_frame=DFRAME, val_col="Vals", date_col="Dt",
        batch_size=BATCH_SIZE, x_steps=X_STEPS, y_features=y_features,
        align_size=True, noise_std=0)
    return pb

def make_unaligned(y_features=Y_FEATURES):
    pb = data_generator.PandasBatches(
        data_frame=DFRAME, val_col="Vals", date_col="Dt",
        batch_size=BATCH_SIZE, x_steps=X_STEPS, y_features=y_features,
        align_size=False, noise_std=0)
    return pb


class BatchGenTest(unittest.TestCase):
    def test_aligned_data_size(self):
        pb = make_aligned()
        print("pb.data:", pb.data)
        self.assertEqual(pb.data_size, ALIGNED_DATA_SIZE)
        self.assertEqual(pb.data[0], DFRAME.Vals[1])
        self.assertEqual(pb.dates[0], DFRAME.Dt[1])

    def test_unaligned_data_size(self):
        pb = make_unaligned()
        print("pb.data:", pb.data)
        self.assertEqual(pb.data_size, ORIG_DATA_SIZE)
        self.assertEqual(pb.data[2], DFRAME.Vals[2])
        self.assertEqual(pb.dates[2], DFRAME.Dt[2])

    def test_scale(self):
        pb = make_unaligned()
        pb.scale(range=None)
        is_similar = np.isclose(pb.data, X_SCALED).all()
        is_similar2 = np.isclose(pb.raw_data, DFRAME_VALS).all()
        self.assertTrue(is_similar, "Scaled data are not the same")
        self.assertTrue(is_similar2, "Raw data has been changed to {}".format(pb.raw_data))

    def test_unscale(self):
        pb = make_unaligned()
        pb.scale(range=None)
        unscaled = pb.unscale(pb.data)
        is_similar = np.isclose(pb.unscale(pb.data), DFRAME_VALS).all()
        self.assertTrue(is_similar, "unscaled data are not the same as RAW. Unscaled:"
                        .format(unscaled))

    def test_x_iter(self):
        pb = make_aligned()
        cnt = 0
        for _, x_batch, _ in pb:
            print("test_x_iter({}): x_batch: -----------------".format(cnt))
            print(x_batch)
            print("test_x_iter({}): X_ALIGN: -----------------".format(cnt))
            print(X_ALIGN_BATCHES[cnt])
            x_batch_eq = np.isclose(x_batch, X_ALIGN_BATCHES[cnt]).all()
            self.assertTrue(x_batch_eq, "Not the same data for x_batch number {}".format(cnt))
            cnt += 1
        self.assertEqual(cnt, N_BATCHES, "Expected {} batches. Got {}"
                         .format(N_BATCHES+1, cnt))

    def test_y_iter(self):
        pb = make_aligned(y_features=Y_FEATURES)
        cnt = 0
        for _, _, y_batch in pb:
            print("test_y_iter({}): y_batch: -----------------".format(cnt))
            print(y_batch)
            print("test_y_iter({}): Y_ALIGN: -----------------".format(cnt))
            print(Y_ALIGN_BATCHES_STEPS[cnt])
            y_batch_eq = np.isclose(y_batch, Y_ALIGN_BATCHES_STEPS[cnt]).all()
            self.assertTrue(y_batch_eq, "Not the same data for y_batch number {}".format(cnt))
            cnt += 1
        self.assertEqual(cnt, N_BATCHES, "Expected {} batches. Got {}"
                         .format(N_BATCHES+1, cnt))

    # def test_y_iter_1(self):
    #     pb = make_aligned(y_features=Y_FEATURES)
    #     cnt = 0
    #     idxs = []
    #     for idx, _, y_batch in pb:
    #         print("test_y_iter_1({}): y_batch: -----------------".format(cnt))
    #         print(y_batch)
    #         print("test_y_iter_1({}): Y_ALIGN: -----------------".format(cnt))
    #         print(Y_ALIGN_BATCHES_1[cnt])
    #         y_batch_eq = np.isclose(y_batch, Y_ALIGN_BATCHES_1[cnt]).all()
    #         idxs.append(idx)
    #         self.assertTrue(y_batch_eq, "Not the same data for y_batch number {}".format(cnt))
    #         cnt += 1
    #     self.assertEqual(cnt, N_BATCHES, "Expected {} batches. Got {}"
    #                      .format(N_BATCHES + 1, cnt))
    #     self.assertEqual(idxs, Y_1_IDX, "Expected {} indexes. Got {}"
    #                      .format(Y_1_IDX, idxs))

    def test_last_x_batch1(self):
        pb = make_aligned()
        x_b1 = pb.get_last_x_batches()
        print("x_b1:", x_b1)
        x_eq = np.isclose(x_b1, LAST_X_BATCH_1).all()
        self.assertEqual(x_b1.shape, LAST_X_BATCH_1.shape)
        self.assertTrue(x_eq)

    def test_last_x_batch2(self):
        pb = make_aligned()
        x_b2 = pb.get_last_x_batches(batch_size=2)
        print("x_b2:", x_b2)
        x_eq = np.isclose(x_b2, LAST_X_BATCH_2).all()
        self.assertEqual(x_b2.shape, LAST_X_BATCH_2.shape)
        self.assertTrue(x_eq)

    def test_append(self):
        """
            Get all data, append one value, append one value, check changes in data, check
            that it is possible to get one batch with size 1.
        """
        pb = make_aligned()
        for _, _, _ in pb:
            pass
        old_data = pb.data.copy()
        expected = np.roll(old_data, -1)
        expected[-1] = 144
        pb.append_elem(144)
        cnt = 0
        pb.batch_size = 1
        self.assertEqual(pb.batch_size, 1)
        for _, _, _ in pb:
            cnt += 1
        print("cnt:", cnt)
        print("pb.data:", pb.data.shape)
        print("expected:", expected.shape)
        is_eq = np.isclose(pb.data, expected).any()
        self.assertTrue(is_eq)
        self.assertTrue(cnt, 1)


if __name__ == '__main__':
    unittest.main()
