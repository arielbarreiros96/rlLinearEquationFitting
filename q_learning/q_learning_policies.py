import numpy as np

__author__ = "Ariel Barreiros and Richar Sosa"
__status__ = "Development"


class QLearningPolicies:

    @staticmethod
    def random_position(data_list):
        rnd = np.random.uniform(0, 1)
        portions = 1 / len(data_list)
        return int(rnd // portions)

    def epsilon_greedy(self, epsilon_trade_off, q_values):

        # This random number is for selecting or not the greedy action
        rnd = np.random.uniform(0, 1)

        # Select a greedy action with a probability of 1 - epsilon_trade_off
        if rnd < (1 - epsilon_trade_off):
            # Find the higher value, and the positions where it's located are contained in the max_value_positions array
            max_value = max(q_values)
            max_count = 0
            max_value_positions = np.empty((1, 0), int)

            # This snippet is for filling max_value_positions array with the positions of the maximum value if repeated
            for i in range(len(q_values)):
                if q_values[i] == max_value:
                    max_count += 1
                    max_value_positions = np.append(max_value_positions, i)

            # if the max value is repeated then any position with this value is equally probable to be selected
            if max_count > 1:
                # Return any random position of the max value
                return max_value_positions[self.random_position(max_value_positions)]

            # if the max value is repeated only once then the best action is in the max_value_positions at position '0'
            elif max_count == 1:
                # Return the position of the max value
                return max_value_positions[0]

        # else select a complete random action
        else:
            return self.random_position(q_values)
