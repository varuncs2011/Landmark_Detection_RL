import pandas as pd
import numpy as np
import math


class ParseData:
    def __init__(self):
        pass

    def get_Data(self):
        train_df = pd.read_csv("training.csv", skiprows=700, nrows=700, sep=",", usecols=[21, 20, 30])
        test_df = pd.read_csv("training.csv", skiprows=700, nrows=300, sep=",", usecols=[21, 20, 30])
        return train_df, test_df

    def convert_to_image(self, x, y, img):
        words = img.split()
        results = list(map(float, words))
        x1_new = np.array(results)
        x1 = x1_new.reshape(96, 96)
        x2 = x1[24:74, 24:74]

        landmark1 = [0, 0]
        if math.isnan(x) or math.isnan(y):
            return x2, landmark1
        else:
            landmark = [int(x) - 24, int(y) - 24]
            return x2, landmark

    def transform(self, train_df, test_df):
        train_images, train_landmarks = zip(
            *[self.convert_to_image(row[0], row[1], row[2]) for index, row in train_df.iterrows()])
        test_images, test_landmarks = zip(
            *[self.convert_to_image(row[0], row[1], row[2]) for index, row in test_df.iterrows()])

        return train_images, test_images, train_landmarks, test_landmarks

