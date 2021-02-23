"""
LU Decomposition and Forward Substitution to solve for x.
Currently working on a fixing a few errors with regards to recieving the wrong solution.
"""

import numpy as np

matrix_a = np.array([[3, 18, 9], [2, 3, 3], [4, 1, 2]], dtype=float)
solution = np.array([[18, 117, 283]], dtype=float)
solution = solution.transpose()


def lu_decomposition_algorithm(input_matrix):
    a_length = len(input_matrix)
    matrix_l = np.zeros([a_length, a_length])

    matrix_p = np.identity(a_length)

    if np.linalg.matrix_rank(input_matrix) != len(input_matrix):
        print("Ranks do not match!")

    elif np.linalg.matrix_rank(input_matrix) < np.linalg.matrix_rank(input_matrix):
        print("No answer!")

    for k in range(a_length - 1):

        maximum_index = abs(input_matrix[k:, k]).argmax() + k
        if input_matrix[maximum_index, k] == 0:
            raise ValueError("Matrix is singular.")

        if maximum_index != k:
            input_matrix[[k, maximum_index]] = input_matrix[[maximum_index, k]]
            matrix_l[[k, maximum_index]] = matrix_l[[maximum_index, k]]
            matrix_p[[k, maximum_index]] = matrix_p[[maximum_index, k]]

        else:
            if input_matrix[k, k] == 0:
                raise ValueError("Pivot element is zero.")

        for row in range(k + 1, a_length):
            matrix_l[row, k] = matrix_a[row, k] / matrix_a[k, k]

            multiple = input_matrix[row, k] / input_matrix[k, k]
            input_matrix[row, k:] = input_matrix[row, k:] - multiple * input_matrix[k, k:]

    matrix_u = matrix_a

    i_counter = 0
    while i_counter <= a_length - 1:

        matrix_l[i_counter, i_counter] = 1
        i_counter += 1

    return matrix_l, matrix_u, matrix_p


def solution_algorithm(matrix_lower, matrix_upper, matrix_iden, matrix_b):

    new_length = len(matrix_upper)
    matrix_y = np.zeros(new_length, float)
    matrix_combined = np.dot(matrix_iden, matrix_b)
    augmented_matrix = np.concatenate((matrix_lower, matrix_combined), 1)

    for i in range(1, new_length):
        sum_first = 0
        for j in range(0, i):
            sum_first += matrix_lower[i, j] * matrix_y[j]

        matrix_y[i] = (matrix_b[i] - sum_first) / matrix_lower[i, i]

    matrix_y = matrix_y.transpose()

    matrix_x = np.zeros(new_length, float)

    k = new_length - 1
    matrix_x[k] = matrix_y[k] / matrix_upper[k, k]

    while k >= 0:
        matrix_x[k] = (matrix_y[k] - np.dot(matrix_upper[k, k + 1:], matrix_x[k + 1:])) / matrix_upper[k, k]
        k = k - 1

    matrix_x = matrix_x.transpose()

    return matrix_x


result_l, result_u, result_p = lu_decomposition_algorithm(matrix_a)

answer = solution_algorithm(result_l, result_u, result_p, solution)
print(answer)
