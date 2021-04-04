import numpy as np

__author__ = "Ariel Barreiros and Richar Sosa"
__status__ = "Development"


class model_parameter:

    def __init__(self, lower_limit, upper_limit, resolution):
        self.__lower_limit = lower_limit
        self.__upper_limit = upper_limit
        self.__resolution = resolution

        possible_values_count = int((self.__upper_limit - self.__lower_limit) / self.__resolution)
        self.__q_values = np.zeros(possible_values_count)

    def get_q_values(self):
        return self.__q_values

    def get_parameter_value_by_selected_q_value(self, selected_q_value, resolution=2):
        return (self.__lower_limit + selected_q_value * self.__resolution).__round__(resolution)

    def update_q_value(self, position, value):
        self.__q_values[position] = value
