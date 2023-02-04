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

def dens_mat_from_stokes_vec(vec):
    s = [np.eye(2, dtype = 'complex128'), np.array([[0, 1], [1, 0]], dtype = 'complex128'), np.array([[0, -1j], [1j, 0]], dtype = 'complex128'), np.array([[1, 0], [0, -1]], dtype = 'complex128')]
    if len(vec) == 4:
        b = np.zeros((2, 2))
        for i in range(4):
            b = b + vec[i] * s[i]
        b = b / 2
    ans_txt = f"The density matrix is {b}"
    st.text(ans_txt)
    plot_bloch_vector_from_dm(b)

def single_qubit_tomo(arr):
    if len(arr) == 4:
        tot = arr[0] + arr[1]
        single_qubit_tomo((2 * arr[2] - 1) / tot, (2 * arr[3] - 1) / tot, (arr[0] - arr[1]) / tot)


st.title("(Ideal) Single Qubit Tomography")
st.text("We assume there is no absorption of the photons by the optical elements.")
Nh = st.text_input("Enter the number of counts detected in H (Nh)")
Nv = st.text_input("Enter the number of counts detected in V (Nv)")
Nd = st.text_input("Enter the number of counts detected in D (Nd)")
Nr = st.text_input("Enter the number of counts detected in R (Nr)")
if Nh != "" and Nv != "" and Nd != "" and Nr != "":
    if st.button("Do it!"):
        single_qubit_tomo([int(Nh), int(Nv), int(Nd), int(Nr)])
