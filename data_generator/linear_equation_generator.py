import numpy as np


class LinearEquationGenerator:

    __data_values = np.empty((1, 0))

    def __init__(self, m, n):
        self.__m = m
        self.__n = n

    def generate_data(self, start_value, end_value, input_resolution=0.1):
        total_samples = int((end_value - start_value) / input_resolution)
        for current_sample in range(total_samples):
            x_value = start_value + (current_sample * input_resolution)
            self.__data_values = np.append(self.__data_values, self.__generate_instant_value(x_value))

    def __generate_instant_value(self, x_value):
        return (self.__m * x_value) + self.__n

    def get_data_values(self):
        return self.__data_values

    def return_plot_data(self, start_value, end_value, input_resolution=0.1):
        total_samples = int((end_value - start_value) / input_resolution)
        x_value = np.empty((1, 0))

        for current_sample in range(total_samples):
            x_value = np.append(x_value, start_value + (current_sample * input_resolution))

        return x_value, self.__data_values
