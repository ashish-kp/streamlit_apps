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

st.title("Generate Bloch vector from quantum state for single qubit.")
df = pd.DataFrame({
    'first column': ['Select', 'Pure', 'Mixed'],
    })

option = st.selectbox(
    'Is the quantum state pure or mixed?',
     df['first column'])
dens_mat = []
if option == 'Pure':
	alpha = st.text_input("Enter the co-efficient of the horizontal basis")
	beta = st.text_input("Enter the co-efficient of the vertical basis")
	if alpha != '' and beta != '':
		alpha, beta = complex(alpha.replace(' ', '')), complex(beta.replace(' ', ''))
		mean = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
		m = np.array([alpha, beta]) * 1 / mean
		dens_mat = density_mat_from_state_vec(m)
		# plot_bloch_vector_from_dm(dens_mat)
elif option == 'Mixed':
	st.text("Enter the four components of the density matrix in the following way : ")
	st.latex(r'''\begin{pmatrix}A & B \\ C & D \end{pmatrix}''')
	A = st.text_input("Enter A")
	B = st.text_input("Enter B")
	C = st.text_input("Enter C")
	D = st.text_input("Enter D")	
	if A != '' and B != '' and C != '' and D != '':
		dens_mat = np.array([[complex(A.replace(' ', '')), complex(B.replace(' ', ''))], [complex(C.replace(' ', '')), complex(D.replace(' ', ''))]])
# 		st.text(dens_mat)
		# plot_bloch_vector_from_dm(dens_mat)
else:
	pass
dens_mat = np.array(dens_mat)
if len(dens_mat.shape) == 2:
    if st.button('Generate Bloch Sphere'):
	_, x, y, z = stokes_vec_from_dens_mat(dens_mat)
	st.write([x, y, z])
        plot_bloch_vector_from_dm(dens_mat)
