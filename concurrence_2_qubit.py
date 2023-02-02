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
    return np.conj(dens_mat).T

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

def concurrence(dens_mat):
    if type(dens_mat) != np.array:
        dens_mat = np.array(dens_mat)
    if nxn_valid_quantum(dens_mat) and dens_mat.shape[0] == 4 and dens_mat.shape[1] == 4:
        eta = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])
        fin = dens_mat @ eta @ complex_conjugate(dens_mat) @ eta 
        eig_val_arr = np.linalg.eigvals(fin)
        r_v = sorted(eig_val_arr, reverse = True)
        val = np.sqrt(r_v[0])
        for x in range(1, 4):
            val -= np.sqrt(r_v[x])
        return max(0. , np.real(np.round(val, 4)))

def bin_ent(x):
    if x == 1:
        return 0
    else:
        return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

def ent(dens_mat):
    return bin_ent((1 + np.sqrt(1 - concurrence(dens_mat)**2)) / 2)

# ent(density_mat_from_state_vec(normalize_state_vec([1, -1j, -2j, 1])))
st.title("Concurrence and Entanglement of a general 2 qubit.")
df = pd.DataFrame({
    'first column': ['Select', 'Pure', 'Mixed'],
    })

option = st.selectbox(
    'Is the quantum state pure or mixed?',
     df['first column'])
dens_mat = []
if option == 'Pure':
    st.text("Enter the four co-efficients of the state vector")
    st.latex(r""" a |00> + b |01> + c |10> + d |11> """)
    a = st.text_input("Enter a")
    b = st.text_input("Enter b")
    c = st.text_input("Enter c")
    d = st.text_input("Enter d")
    if a != "" and b != "" and c != "" and d != "":
        alpha, beta, gamma, delta = complex(a.replace(' ', '')), complex(b.replace(' ', '')), complex(c.replace(' ', '')), complex(d.replace(' ', ''))
        dens_mat_con = density_mat_from_state_vec(normalize_state_vec([alpha, beta, gamma, delta]))
        if st.button("Do it!"):
            ans_str = f"The concurrence is {concurrence(dens_mat_con)} and the entanglement is {ent(dens_mat_con)}"
            st.text(ans_str)
elif option == 'Mixed':
    st.text("Enter the values p1, p2, p3 (p4 is inferred) to create a mixed state from Bell States in the following manner.")
    st.latex(r"p1 |\phi^+><\phi^+| + p2 |\phi^-><\phi^-| + p3 |\psi^+><\psi^+| + p4 |\psi^-><\psi^-|")
    b_1 = density_mat_from_state_vec(normalize_state_vec([1, 0, 0, 1]))
    b_2 = density_mat_from_state_vec(normalize_state_vec([1, 0, 0, -1]))
    b_3 = density_mat_from_state_vec(normalize_state_vec([0, 1, 1, 0]))
    b_4 = density_mat_from_state_vec(normalize_state_vec([0, 1, -1, 0]))
    a = st.text_input("Enter p1")
    b = st.text_input("Enter p2")
    c = st.text_input("Enter p3")
    if a != '' and b != '' and c != '':
        p1, p2, p3 = float(a.replace(' ', '')), float(b.replace(' ', '')), float(c.replace(' ', ''))
        p4 = 1 - p1 - p2 - p3
        dens_mat_mix = p1 * b_1 + p2 * b_2 + p3 * b_3 + p4 * b_4
        if st.button("Do it!"):
            ans_str = f"The concurrence is {concurrence(dens_mat_mix)} and the entanglement is {ent(dens_mat_mix)}"
            st.text(ans_str)
