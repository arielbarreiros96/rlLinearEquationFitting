import numpy as np
import math
import data_generator.linear_equation_generator as linear_gen
import q_learning.q_learning_policies as policies
import q_learning.model_parameter as mp
import matplotlib.pyplot as plt


def generate_reward(error, resolution=8):
    return ((math.pow(0.3, error)) * 1000).__round__(resolution)


def main():

    # generating a dataset for later learning algorithm
    dataset_lower_time = 0
    dataset_upper_time = 50
    dataset_input_resolution = 1
    dataset = linear_gen.LinearEquationGenerator(1.5, 2.6)
    dataset.generate_data(dataset_lower_time, dataset_upper_time, dataset_input_resolution)

    # Actual learning algorithm, ONLY for 2 values
    linear_equation_model = np.array([mp.model_parameter(0, 5, 0.05), mp.model_parameter(0, 5, 0.05)])

    # Iterate several times to adjust the q_values in order to select the best model at the very end of adjustment
    planned_iterations = 10000
    policy = policies.QLearningPolicies()
    epsilon_trade_off = 0.15
    for current_iteration in range(planned_iterations):

        print(f"Iteration: {current_iteration}")

        # Select a set of parameters with RL and evaluate the model performance
        selected_parameters = np.empty((1, 0))
        selected_parameter_positions = np.empty((1, 0), dtype=np.int64)
        for parameter in linear_equation_model:
            # Select a position by its q_value
            selected_parameter_position = policy.epsilon_greedy(epsilon_trade_off, parameter.get_q_values())
            selected_parameter_positions = np.append(selected_parameter_positions, selected_parameter_position)

            # Find the parameter value by the position of the q_value
            selected_parameter = parameter.get_parameter_value_by_selected_q_value(selected_parameter_position)
            selected_parameters = np.append(selected_parameters, selected_parameter)

        # Evaluate selected parameters by comparing its values with a dataset
        estimated_model = linear_gen.LinearEquationGenerator(selected_parameters[0], selected_parameters[1])
        estimated_model.generate_data(dataset_lower_time, dataset_upper_time, dataset_input_resolution)

        # Calculate the mse
        mse_vector = dataset.get_data_values() - estimated_model.get_data_values()
        mse_vector = mse_vector**2
        mse_vector = mse_vector/2
        # Generate a reward based on the maximum value of the mse vector
        reward = generate_reward(max(mse_vector), 8)

        # Update the q_values and restart the process
        for position in range(len(linear_equation_model)):
            # I'm using the linear_equation_model length because it always matches with selected_parameter_positions's
            iterable_q_value = linear_equation_model[position].get_q_values()[selected_parameter_positions[position]]
            iterable_q_value = (iterable_q_value + 0.1 * (reward - iterable_q_value))
            linear_equation_model[position].get_q_values()[selected_parameter_positions[position]] = iterable_q_value.__round__(4)

    # After the process ends show the resulting m and n variables
    m = policy.epsilon_greedy(0, linear_equation_model[0].get_q_values())
    m = linear_equation_model[0].get_parameter_value_by_selected_q_value(m)

    n = policy.epsilon_greedy(0, linear_equation_model[1].get_q_values())
    n = linear_equation_model[0].get_parameter_value_by_selected_q_value(n)

    print(f"\nThe resulting equation was y = {m}*x + {n}")
    x, y = dataset.return_plot_data(dataset_lower_time, dataset_upper_time, dataset_input_resolution)
    plt.plot(x, y)
    estimated_model = linear_gen.LinearEquationGenerator(m, n)
    estimated_model.generate_data(dataset_lower_time, dataset_upper_time, dataset_input_resolution)
    x, y = estimated_model.return_plot_data(dataset_lower_time, dataset_upper_time, dataset_input_resolution)
    plt.plot(x, y)
    plt.legend(["dataset", "model"])
    plt.show()


if __name__ == "__main__":
    main()
