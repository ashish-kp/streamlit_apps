import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import qiskit as qt
from qiskit.visualization import plot_bloch_vector
import qutip as qut

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

def nxn_valid_quantum(sqr_mat):
    if type(sqr_mat) != np.array:
        sqr_mat = np.array(sqr_mat)
    flag = True
    for x in np.linalg.eigvals(sqr_mat):
        if x < (-1 * 10**-4): flag = False; raise ValueError("Negative eigen values")
    if trace(sqr_mat) < 0.999 or trace(sqr_mat) > 1.0001: flag = False; raise ValueError("Trace is not equal to 1")
    if trace(np.dot(sqr_mat, sqr_mat)) > 1.0001: flag = False; raise ValueError("Trace of rho squared is greater than 1")
    return flag


def upper(rng):
    cnt = 0
    while rng > 0:
        rng //= 2
        cnt += 1
    return cnt 

def dec2bin(num, upper_rng):
    arr = []
    while num > 0:
        arr.append(str(num % 2))
        num //= 2
    while len(arr) < upper_rng:
        arr.append('0')
    return "".join(arr)[::-1]

# Create equal superposition of all qubits

def create_all_initial(solutions, rng):
    arr = []
    upper_rng = upper(rng)
    for i in range(2**upper_rng):
        arr.append(str(dec2bin(i, upper_rng)))
        # arr.append(i)
    return dict.fromkeys(arr, 1 / np.sqrt(2**upper_rng))

def negate_sols(solutions, state):
    for x in solutions:
        state[dec2bin(x, upper(rng))] = -state[dec2bin(x, upper(rng))]

# negate_sols(solutions, state)

def calc_mean(state):
    return sum(list(state.values())) / len(list(state.values()))

# print(calc_mean(state))

def invmean(state):
    state_mean = calc_mean(state)
    upper_rng = upper(rng)
    for i in range(2**upper_rng):
        state_bin_val = dec2bin(i, upper_rng)
        state[state_bin_val] = 2 * state_mean - state[state_bin_val]

def grover(solutions, state, number):
    for i in range(number):
        negate_sols(solutions, state)
        solns.append(state[dec2bin(solutions[0], upper(rng))])
        invmean(state)
        solns.append(state[dec2bin(solutions[0], upper(rng))])

def plot_point(a, b):
    theta = np.arccos(a)
    return [ 0, np.sin(2 * theta), np.cos(2 * theta)]

def rotate_vec(vector, angle):
    return [np.cos(angle) * vector[0] - vector[1] * np.sin(angle), vector[0] * np.sin(angle) + vector[1] * np.cos(angle)]


st.title("Grover's Search Algorithm")

sols = st.text_input("Enter the numbers:")
solutions = []
if sols != "":
    for c in sols:
        if c.isnumeric():
            solutions.append(int(c))

rng = st.text_input("Give the range")
if rng != "":
    rng = int(rng)
    solns = []
    state = create_all_initial(solutions, rng)
    grover(solutions, state, int(np.pi * np.sqrt(2**upper(rng) / len(solutions)) / 4))
    x = list(state.keys())
    y = list(state.values())

    if st.button("Do it"):
        nn = f"Max probability of solution is {solns[-1]} in {len(solns) // 2} iterations."
        st.text(nn)
        N = 2**upper(rng)
        M = len(solutions)
        all_angles = []

        ang_by_2 = (1 * np.arcsin(np.sqrt(M / N)))

        ini_vec = [np.cos(ang_by_2), np.sin(ang_by_2)]

        all_angles.append(ini_vec)

        n = int(np.pi / (4 * np.arcsin(np.sqrt(M / N))) - 0.5)
        # print(N, M, n)
        next_vec = rotate_vec(ini_vec, ang_by_2 * 2)
        all_angles.append(next_vec)
        # y = rotate_vec(x, ang_by_2 * 2)
        # all_angles.append(y)

        while n > 0:
            prev_vec = next_vec
            if prev_vec[1] > next_vec[1]:
                break
            else:
                next_vec = rotate_vec(next_vec, ang_by_2 * 2)
                all_angles.append(next_vec)
            n -= 1
        # b_x = []
        # b_y = []
        # b_z = []
        # for i in range(len(all_angles)):
        #     x, y = all_angles[i]
        #     a, b, c = plot_point(x, y)
        #     b_x.append(a)
        #     b_y.append(b)
        #     b_z.append(c)
        # b = qut.Bloch()
        # b.add_points([b_x, b_y, b_z])
        # st.pyplot(b.render())
        plt.figure(figsize = (5, 5))
        fig, ax = plt.subplots()
        for i in range(len(all_angles)):
            ax.scatter(all_angles[i][0], all_angles[i][1], s = 100)
            ax.plot([0, all_angles[i][0]], [0, all_angles[i][1]])
        ax.grid()
        ax.set_xlim((-0.4, 1.2))
        ax.set_ylim((-0.4, 1.2))
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$\beta$")
        st.pyplot(fig)
