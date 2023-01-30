import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import qiskit as qt
from qiskit.visualization import plot_bloch_vector
import qutip as qut

def normalize_state_vec(state_vec):
    if type(state_vec) != np.array:
        state_vec = np.array(state_vec)
    norm_fact = 0
    for x in state_vec:
        norm_fact += x * np.conj(x)
    return 1 / np.sqrt(np.real(norm_fact)) * state_vec

def density_mat_from_state_vec(state_vec):
    if type(state_vec) != np.array:
        state_vec = np.array(state_vec)
    return np.dot(state_vec.reshape(state_vec.shape[0], 1), np.conj(state_vec.reshape(1, state_vec.shape[0])))

def stokes_vec_from_dens_mat(dens_mat):
    if type(dens_mat) != np.array:
        dens_mat = np.array(dens_mat)
    if nxn_valid_quantum(dens_mat):
        s0 = np.real(trace(dens_mat))
        s1 = np.real((trace(np.array([[0, 1], [1, 0]]) @ dens_mat)))
        s2 = np.real((trace(np.array([[0, -1j], [1j, 0]]) @ dens_mat)))
        s3 = np.real(trace(np.array([[1, 0], [0, -1]]) @ dens_mat))
        return [s0, s1, s2, s3]
    else: return False

# _, x, y, z = stokes_vec_from_dens_mat()

def plot_bloch_vector_from_dm(dens_mat):
    if stokes_vec_from_dens_mat(dens_mat) and nxn_valid_quantum(dens_mat):
        _, x, y, z = stokes_vec_from_dens_mat(dens_mat)
        st.pyplot(plot_bloch_vector([x, y, z]))

def trace(sqr_mat):
    if type(sqr_mat) != np.array:
        sqr_mat = np.array(sqr_mat)
    if sqr_mat.shape[0] == sqr_mat.shape[1]:
        return np.sum([sqr_mat[i][i] for i in range(sqr_mat.shape[0])])
    else:
        return "Entered matrix is not a square matrix."

def complex_conjugate(dens_mat):
    if type(dens_mat) != np.array:
        dens_mat = np.array(dens_mat)
    for i in range(dens_mat.shape[0]):
        for j in range(dens_mat.shape[1]):
            dens_mat[i][j] = np.conj(dens_mat[i][j])
    return dens_mat.T

def nxn_valid_quantum(sqr_mat):
    if type(sqr_mat) != np.array:
        sqr_mat = np.array(sqr_mat)
    flag = True
    if np.sum(sqr_mat == complex_conjugate(sqr_mat)) != (sqr_mat.shape[0] * sqr_mat.shape[1]): 
        raise ValueError("The given matrix is not Hermitian")
    for x in np.linalg.eigvals(sqr_mat):
        if x < (-1 * 10**-4): flag = False; raise ValueError("Negative eigen values")
    if trace(sqr_mat) < 0.999 or trace(sqr_mat) > 1.0001: flag = False; raise ValueError("Trace is not equal to 1")
    if trace(np.dot(sqr_mat, sqr_mat)) > 1.0001: flag = False; raise ValueError("Trace of rho squared is greater than 1")
    return flag

def arr2num(arr):
    num = 0
    arr = arr[::-1]
    for i in range(len(arr)):
        num += arr[i] * 2**i
        # print(num, arr[i], i)
    return num

def dec2bin_num(num):
    bin = []
    while num > 0:
        bin.append(num % 2)
        num //= 2
    while len(bin) < n:
        bin.append(0)
    return bin[::-1]

def generate_bin(n):
    arr = []
    for x in range(2**n):
        arr.append(dec2bin_num(x))
    return arr
st.title("Partial Transform of a Single Qubit")
n = st.text_input("Enter the number of qubits(less than 8)")
if n != "" and int(n) < 8:
    n = int(n)
    st.text("Enter the value of A (greater than 1) such that:")
    st.latex("|\phi> = \sum_{n = 1}^{2^n} n\, mod\, A\,(Unnormalized)")

    val = st.text_input("Enter A")
    if val != "":
        val = int(val)

        qubit = st.text_input("Enter which qubit you wish to partially transform as an integer")
        if qubit != "":
            if int(qubit) < n:
                qubit = int(qubit)
                if st.button("Do it!"):
                    dens_mat = density_mat_from_state_vec(normalize_state_vec([x % val for x in range(2**n)]))

                    all_binary = generate_bin(n)
                    all_binary_2 = generate_bin(n)
                    # qubit = 0
                    part_qubit = n - qubit - 1
                    cnt = 0
                    ppt_dens = dens_mat
                    # ppt_dens = np.zeros((2**n, 2**n))
                    for i in range(2**n):
                        for j in range(2**n):
                            if all_binary[i][part_qubit] != all_binary[j][part_qubit]:
                                dens_mat = density_mat_from_state_vec(normalize_state_vec([x % val for x in range(2**n)]))
                                # old_i, old_j = arr2num(all_binary[i]), arr2num(all_binary[j])
                                all_binary_2[i][part_qubit], all_binary_2[j][part_qubit] = all_binary[j][part_qubit], all_binary[i][part_qubit]
                                # print(arr2num(all_binary[i]), arr2num(all_binary[j]), "New",arr2num(all_binary_2[i]), arr2num(all_binary_2[j]))
                                ppt_dens[arr2num(all_binary[i])][arr2num(all_binary[j])] = dens_mat[arr2num(all_binary_2[i])][arr2num(all_binary_2[j])]
                                # print(dens_mat, "\n", ppt_dens)
                            else:
                                # print(i, j)
                                ppt_dens[i][j] = dens_mat[i][j]
                    fig, ax = plt.subplots()
                    ax.imshow(ppt_dens)
                    ax.axis('off')
                    st.pyplot(fig)
            else:
                st.text("The qubit value you have entered is more than the number of qubits \"n\" ")
else:
    st.text("Please enter a value less than 8")
