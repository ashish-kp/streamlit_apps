import streamlit as st
import numpy as np
import re
import sqtdiat.qops as sq
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy as sc

# N_all = [1359,61,693,687,63,1695,1004,806,791,854,69,922,713,991,932,80]
N_all = [1359, 61, 693, 886, 63, 1695, 1004, 894, 791, 854, 69, 809, 645, 971, 902, 76]
# N_all = [1000, 0, 500, 500, 0, 0, 0, 0, 500, 0, 250, 250, 500, 0, 250, 250]
# N_all = [np.random.randint(10, 1000) for x in range(16)]
Singles = [[27930, 29536], [28100, 38687], [28749, 35097], [28469, 34761], [38557, 30369], [38563, 39733],[38603, 34578],[38504, 35088],
		   [33702, 28628],[33598, 39067],[33669, 34027],[33517, 33683],[32113, 28109],[32198, 38994],[32314, 33960],[32256, 33829]]

def normalize_state_vec(state_vec):
	if type(state_vec) != np.array:
		state_vec = np.array(state_vec)
	norm_fact = 0
	for x in state_vec:
		norm_fact += x * np.conj(x)
	return 1 / np.sqrt(np.real(norm_fact)) * state_vec

def sqrt_dens_mat(dens_mat):
    # if nxn_valid_quantum(dens_mat):
    eig_vals, eig_vecs = np.linalg.eig(dens_mat)
    eig_vals = np.real(eig_vals)
    iden = np.eye(len(eig_vals))
    for i in range(len(eig_vals)):
        if np.abs(eig_vals[i]) < 10e-4:
            val = 0
        else:
            val = np.sqrt(eig_vals[i])
        iden[i] = val * iden[i]
#     print(iden)
    return eig_vecs.T @ iden @ np.linalg.inv(eig_vecs.T)

def fidelity(rho1, rho2):
    if type(rho1) != np.array:
        rho1 = np.array(rho1)
    if type(rho2) != np.array:
        rho2 = np.array(rho2)
    if rho1.shape == rho2.shape:
        if nxn_valid_quantum(rho1) and nxn_valid_quantum(rho2):
            sqrt_rho1 = sc.linalg.sqrtm(rho1)
            val = sqrt_rho1 @ rho2 @ sqrt_rho1
            if np.imag(trace(val)) < 10e-4:
                return np.round(np.real((trace(sc.linalg.sqrtm(val)))**2), 7)
    else:
        raise ValueError(f"Given density matrices are not of same dimension {rho1.shape}, {rho2.shape}")

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


def rho_from_t_2(t_vec):
	t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16 = t_vec
	T_d = np.array([[t1, 0, 0, 0], [t5 + 1j * t6, t2, 0, 0], [t7 + 1j * t8, t9 + 1j * t10, t3, 0], [t11 + 1j * t12, t13 + 1j * t14, t15 + 1j * t16, t4]])
	return np.conj(T_d).T @ T_d

def project(state_vec):
	return density_mat_from_state_vec(normalize_state_vec(state_vec))

def make_basis():
	basis_vecs_1 = np.array([np.array([1, 0]), np.array([0, 1]), np.array([1, 1]), np.array([1, 1j])])
	basis_vecs = []
	for i in range(4):
		for j in range(4):
			basis_vecs.append(project(np.kron(basis_vecs_1[i], basis_vecs_1[j])))
	return basis_vecs

def cons2(t_vec):
	tr_val = 0
	for x in t_vec:
		tr_val += x**2
	return tr_val - 1

def concurrence(dens_mat):
	if type(dens_mat) != np.array:
		dens_mat = np.array(dens_mat)
	if nxn_valid_quantum(dens_mat) and dens_mat.shape[0] == 4 and dens_mat.shape[1] == 4:
		eta = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])
		fin = dens_mat @ eta @ dens_mat.T @ eta 
		eig_val_arr = np.linalg.eigvals(fin)
		r_v = sorted(eig_val_arr, reverse = True)
		val = np.sqrt(r_v[0])
		for x in range(1, 4):
			val -= np.sqrt(r_v[x])
		return max(0. , np.real(np.round(val, 4))), fin, eig_val_arr

def two_qubit_tomography(N_all, x0 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], Singles = [[0, 0] for x in range(16)], delt = 20 * 10**-9, T_tot = 1):
	N_tot = (N_all[0] + N_all[1] + N_all[4] + N_all[5])
	N_all = np.array(N_all) / N_tot
	cons = [{'type': 'eq', 'fun': cons2}]
	E = [1, 1, 1, 1]
	E_all = []
	for x in E:
		for y in E:
			E_all.append(x * y)
	def n_acc_generate(Singles, delt, T_tot):
		n_acc = []
		for x in Singles:
			n_acc.append((x[0] * x[1] * delt) / (T_tot * N_tot))
		return n_acc
	n_acc = n_acc_generate(Singles, delt, T_tot)
	def objective_func_2(t_vec):
		rho = rho_from_t_2(t_vec)
		basis_outers = make_basis()
		L = 0
		for i in range(16):
			L += (E_all[i] * np.trace(basis_outers[i] @ rho) - N_all[i] - n_acc[i])**2 / (2 * np.trace(basis_outers[i] @ rho))
		return np.real(L)
	res = minimize(objective_func_2, x0, method='SLSQP', constraints=cons)

	return rho_from_t_2(res.x)
st.set_page_config(page_title = "Two Qubit Tomography(Beta)")
st.title("Two Qubit Tomography (Beta)")

st.text("""

	This interface can be used to obtain 
	the density matrix of a two qubit quantum system 
	upto arbitrary precision. Still requires a few corrections. This is but a prototype.""")

# st.latex(r"""\rho""")

st.text("Reference")
st.write("http://research.physics.illinois.edu/QI/Photonics/tomography-files/amo_tomo_chapter.pdf")

col1, col2, col3, col4 = st.columns(4)
with col1:
	a = st.text_input("HH", 1000)
	b = st.text_input("HV", 1000)
	c = st.text_input("HD", 1000)
	d = st.text_input("HL", 1000)
with col2:
	e = st.text_input("VH", 1000)
	f = st.text_input("VV", 1000)
	g = st.text_input("VD", 1000)
	h = st.text_input("VL", 1000)
with col3:
	j = st.text_input("DH", 1000)
	k = st.text_input("DV", 1000)
	l = st.text_input("DD", 1000)
	m = st.text_input("DL", 1000)
with col4:
	n = st.text_input("LH", 1000)
	o = st.text_input("LV", 1000)
	p = st.text_input("LD", 1000)
	q = st.text_input("LL", 1000)

acc_counts = st.checkbox("Accidental Counts Compensation")

if acc_counts:
	st.text("Enter Single Detector counts.")
	col5, col6 = st.columns(2)
	with col5:
		SH1 = st.number_input("HH 1")
		SH2 = st.number_input("HV 1")
		SH3 = st.number_input("HD 1")
		SH4 = st.number_input("HL 1")
		SH5 = st.number_input("VH 1")
		SH6 = st.number_input("VV 1")
		SH7 = st.number_input("VD 1")
		SH8 = st.number_input("VL 1")
		SH9 = st.number_input("DH 1")
		SH10 = st.number_input("DV 1")
		SH11 = st.number_input("DD 1")
		SH12 = st.number_input("DL 1")
		SH13 = st.number_input("LH 1")
		SH14 = st.number_input("LV 1")
		SH15 = st.number_input("LD 1")
		SH16 = st.number_input("LL 1")
	with col6:
		SH1_ = st.number_input("HH 2")
		SH2_ = st.number_input("HV 2")
		SH3_ = st.number_input("HD 2")
		SH4_ = st.number_input("HL 2")
		SH5_ = st.number_input("VH 2")
		SH6_ = st.number_input("VV 2")
		SH7_ = st.number_input("VD 2")
		SH8_ = st.number_input("VL 2")
		SH9_ = st.number_input("DH 2")
		SH10_ = st.number_input("DV 2")
		SH11_ = st.number_input("DD 2")
		SH12_ = st.number_input("DL 2")
		SH13_ = st.number_input("LH 2")
		SH14_ = st.number_input("LV 2")
		SH15_ = st.number_input("LD 2")
		SH16_ = st.number_input("LL 2")       
	Singles = [[SH1, SH1_], [SH2, SH2_], [SH3, SH3_], [SH4, SH4_], [SH5, SH5_], [SH6, SH6_], [SH7, SH7_], [SH8, SH8_], [SH9, SH9_], [SH10, SH10_], [SH11, SH11_], [SH12, SH12_], [SH13, SH13_], [SH14, SH14_], [SH15, SH15_], [SH16, SH16_]]


if a != '' and b != '' and c != '' and d != '' and e != '' and f != '' and g != '' and h != '' and j != '' and k != '' and l != '' and m != '' and n != '' and o != '' and p != '' and q != '':

	N_all_0 = [a, b, c, d, e, f, g, h, j, k, l, m, n, o, p, q]
	N_all = []
	for x in N_all_0:
		N_all.append(float(x))
	dens_mat_2 = two_qubit_tomography(N_all)

	fig = plt.figure(figsize=(8, 3))
	ax1 = fig.add_subplot(121, projection='3d')
	ax2 = fig.add_subplot(122, projection='3d')

	# fake data
	_x = np.arange(4)
	_y = np.arange(4)
	_xx, _yy = np.meshgrid(_x, _y)
	x, y = _xx.ravel(), _yy.ravel()

	dx = dy = 0.5

	z_data = np.real(dens_mat_2)
	z_data_1 = np.imag(dens_mat_2)
	dz = z_data.ravel()
	dz1 = z_data_1.ravel()
	ax1.bar3d(x, y, 0, dx, dy, dz, shade=True)
	ax1.set_zlim(-1, 1)
	ax1.set_title('Real values')

	ax2.bar3d(x, y, 0, dx, dy, dz1, shade=True)
	ax2.set_title('Imaginary values')
	ax2.set_zlim(-1, 1)
	if st.button("Do it"):
		st.write(np.round(dens_mat_2, 3))
		conc = f"Concurrence = {concurrence(dens_mat_2)[0]}"
		st.write(conc)
		st.pyplot(fig)
	if st.checkbox("Fidelity"):
		# coll1, coll2, coll3, coll4 = st.columns(4)
		# with coll1:
		fa = st.text_input("HH component", 1)
		# with coll2:
		fb = st.text_input("HV component", 0)
		# with coll3:
		fc = st.text_input("VH component", 0)
		# with coll4:
		fd = st.text_input("VV component", 1)	
		if fa != '' and fb != '' and fc != '' and fd != '':
			fid_st = np.array([complex(fa.replace(' ', '')), complex(fb.replace(' ', '')), complex(fc.replace(' ', '')), complex(fd.replace(' ', ''))])
			fid_den_mat = density_mat_from_state_vec(normalize_state_vec(fid_st))
			st.write(f"Fidelity with the given state is {fidelity(dens_mat_2, fid_den_mat)}")
