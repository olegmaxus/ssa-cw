import numpy as np
import numpy.linalg as npl
import scipy as sp
import scipy.linalg as spl
from cmath import *
from math import floor

'''
/// comments on trajectory_matrix ///
i use it to understand the inner working of the parts of algorithm, i am implementing

'''


def trajectory_matrix(input_array, window_size): # window_size = L
    hankel_T = np.array([k for k in input_array[0 : window_size]])
    for i in range(1, len(input_array) - window_size + 1): # range(1, K = N - L + 1)
        temp_row = np.array([j for j in input_array[i:i+window_size]])
        hankel_T = np.vstack((hankel_T, temp_row))
    return np.real(np.transpose(hankel_T))


def eigenvectors(input_matrix):
    res = npl.eig(input_matrix)
    return np.real(np.array(res[1]))


def eigenvalues(input_matrix):
    res = npl.eig(input_matrix)
    return np.real(np.array(res[0]))


def rank(x):
    return npl.matrix_rank(x)


def specific_singular_value_decomposition(input_matrix):

    s = np.dot(input_matrix, np.transpose(input_matrix))
    spectrum, u = eigenvalues(s), eigenvectors(s)

    sort_index = spectrum.argsort()[::-1]
    spectrum = spectrum[sort_index]
    u = u[:, sort_index]

    rank_x = npl.matrix_rank(input_matrix)
    v_0 = (np.dot(np.transpose(input_matrix), u[:, [0]]))
    v_0 /= np.sqrt(spectrum[0])
    for d in range(1, rank_x):
        v_d = (np.dot(np.transpose(input_matrix), u[:, [d]]))
        v_d /= np.sqrt(spectrum[d])
        v_0 = np.hstack((v_0, v_d))

    return np.real(u), np.real(v_0), np.real(spectrum)


def singular_elementary_decomposition(u, v, sorted_spectrum, rank_x):

    collection = np.array([np.sqrt(sorted_spectrum[0]) * np.dot(u[:, [0]], np.transpose(v[:, [0]]))])

    for i in range(1, rank_x):
        x_i = np.sqrt(sorted_spectrum[i]) * np.dot(u[:, [i]], np.transpose(v[:, [i]])) #!
        collection = np.concatenate((collection, [x_i]))
    return np.real(collection)


def eigentriples(input_matrix):

    u, v, spec = specific_singular_value_decomposition(input_matrix)
    rank_x = rank(input_matrix)
    collection = [tuple((np.sqrt(spec[0]), u[:, 0], v[:, 0]))]

    for i in range(1, rank_x):
        triplet_i = tuple((np.sqrt(spec[i]), u[:, i], v[:, i]))
        collection.append(triplet_i)
    return collection


def book_diagonal_averaging(matrix):

    window_length, row_length = matrix.shape # L, K
    averaged_result = np.zeros((window_length, row_length))#np.zeros(window_length + row_length - 1)

    for i in range(window_length):
        for j in range(row_length):
            k = i + j
            if k in range(0, window_length):
                for m in range(0, k + 1):
                    averaged_result[i, j] += (matrix[m, k - m])/(k + 1)
                    #averaged_result[i + j] += (matrix[m, k - m])/(k + 1)
            elif k in range(window_length, row_length):
                for m in range(0, window_length - 1):
                    averaged_result[i, j] += (matrix[m, k - m])/window_length
                    #averaged_result[i + j] += (matrix[m, k - m]) / window_length
            elif k in range(row_length, window_length + row_length - 1):
                for m in range(k - row_length + 1, window_length):
                    averaged_result[i, j] += (matrix[m, k - m])/(window_length + row_length - k - 1)
                    #averaged_result[i + j] += (matrix[m, k - m]) / (window_length + row_length - k - 1)
    #for i in range(len(averaged_result)):
    #    if i <= (len(averaged_result) // 2):
    #        averaged_result[i] = averaged_result[i] / (i + 1)
    #    else:
    #        averaged_result[i] = averaged_result[i] / (len(averaged_result) - i)
    return averaged_result


def fast_diagonal_averaging(matrix):

    straight_diag = matrix[::-1]
    res_g_series = np.zeros(straight_diag.shape[0] + straight_diag.shape[1] - 1)

    for k in range(-straight_diag.shape[0] + 1, straight_diag.shape[1]):
        res_g_series[k + straight_diag.shape[0] - 1] = straight_diag.diagonal(k).mean()
    return res_g_series
    #X_rev = matrix[::-1]
    # Full credit to Mark Tolonen at https://stackoverflow.com/a/6313414 for this one:
    #return np.array([X_rev.diagonal(i).mean() for i in range(-matrix.shape[0] + 1, matrix.shape[1])])

