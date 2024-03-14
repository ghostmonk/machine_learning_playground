from typing import List

import pandas as pd
from pandas import DataFrame
from pandas._typing import FilePath
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


class RandomForestTrainingModel:
    def __init__(self, file_path: FilePath, random_state=1):
        """
        :param file_path: Location of the data file
        :param random_state: Seed for making the model reproducible
        """
        self.val_y = None
        self.train_y = None
        self.val_X = None
        self.train_X = None
        self.X = None
        self.y = None
        # read the data and create a DataFrame
        self.data = pd.read_csv(file_path)
        self.random_state = random_state

    def describe(self) -> DataFrame:
        # return a description of the data (stats)
        return self.data.describe()

    def set_up_training(self, prediction_target: str, feature_columns: List[str]) -> None:
        """
        Will set the data up for training
        :param prediction_target: The column in the data that we want to predict
        :param feature_columns: The columns in the data that we want to use to make the prediction
        """
        # a list of features to be used in the model for prediction
        self.X = self.data[feature_columns]

        # specify the prediction target
        self.y = self.data[prediction_target]

        # split the data into training and validation data
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(
            self.X, self.y, random_state=self.random_state
        )

    def get_mean_error(self, max_leaf_nodes: int | None = None) -> float:
        """
        Will return the mean error of the model
        :param max_leaf_nodes: The maximum number of leaf nodes to use with the DecisionTreeRegressor
        :return: a float representing the mean error
        """
        model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=self.random_state)
        model.fit(self.train_X, self.train_y)
        predictions = model.predict(self.val_X)
        return mean_absolute_error(predictions, self.val_y)

    def find_best_leaf_num(self) -> int:
        """
        Will return the best number of leaf nodes to use with the DecisionTreeRegressor iterating through a range of
        possible values from 5 to 500 in increments of 5
        :return: The number of leaves that resulted in the lowest mean error
        """
        leaf_nodes = [i for i in range(5, 500, 5)]
        return min(leaf_nodes, key=lambda x: self.get_mean_error(x))
