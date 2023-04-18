import streamlit as st
from qiskit import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
sim = Aer.get_backend('aer_simulator')

st.title("One Qubit Discrete Wigner Distribution Tomography")

st.write(r"""The counts are measured for 4 projective measurements in the $H, V, D$ and $L$ bases.

The probabilities are obtained from the counts, which in turn are used to form the ideal Wigner Distribution in the following way:
$$\begin{bmatrix}w_{10} & w_{11} \\ w_{00} & w_{01}\end{bmatrix} $$ is nothing but ,$$ \frac{1}{2} \begin{bmatrix} P_H + P_A + P_R - 1 & P_V + P_A + P_L - 1 \\ P_H + P_D + P_L - 1 & P_V + P_D + P_R - 1\end{bmatrix}$$
	""")

st.write(""" An optimization algorithm is used to obtain the correct Wigner Distribution from the obtained measurements.
	We strive to begin with a random Wigner Distribution, and move towards the correct Wigner Distribution in an iterative process
	, by minimizing a certain "objective function".

	""")

st.write(r"""Perhaps, the most elegant part of the process would be that, the constraint used is the following:

$$s_x^2 + s_y^2 + s_z^2 \leq s_0^2$$, which along with the objective function, leads to the satisfaction of the four properties for a valid density matrix.

1. $Tr(\rho) = 1$

2. $Tr(\rho^2) \leq 1$
3. $\rho = \rho^{\dagger}$
4. $Eig(\rho) \geq 0$
	""")

st.write("""The following is the proof for the fact that the constraint successfully satisfues the fourth property. """)

st.write(r"""$$\begin{bmatrix} s_0 + s_z & s_x - i s_y \\ s_0 - s_z & s_x + is_y \end{bmatrix} = \begin{bmatrix} a - \lambda & b \\ b* & d - \lambda \end{bmatrix} = (a-\lambda)(d-\lambda) - |b|^2$$

$$\lambda^2 - \lambda(a + d) - |b|^2 + ad = 0$$
Solving this, we get:
$$\lambda=\dfrac{(a + d) \pm \sqrt{(a-d)^2 + 4|b|^2}}{2}$$
$$ ad > |b|^2 \rightarrow s_0^2 - s_z^2 > s_x^2 + s_y^2 \rightarrow s_0^2 \geq s_x^2 + s_y^2 + s_z^2. $$""")

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
    if np.trace(sqr_mat) < 0.999 or np.trace(sqr_mat) > 1.0001: flag = False; raise ValueError("Trace is not equal to 1")
    if np.trace(np.dot(sqr_mat, sqr_mat)) > 1.0001: flag = False; raise ValueError("Trace of rho squared is greater than 1")
    return flag

def random_counts(phi_ = []):
    "Creates a random pure state and measures the counts in the H, V, D and L basis"
    alpha = np.random.random()
    if np.sum(phi_) == 0:
        phi = sq.norm_state_vec([alpha, np.sqrt(1 - alpha**2)])
    else:
        phi = phi_
    meas_data = []
    for i in range(3):
        dwf_tom = QuantumCircuit(1)
        dwf_tom.initialize(phi, 0)
        if i == 1:
            dwf_tom.h(0)
        if i == 2:
            dwf_tom.p(-np.pi / 2, 0)
            dwf_tom.h(0)
            dwf_tom.p(np.pi / 2, 0)
        dwf_tom.measure_all()
        meas = sim.run(assemble(dwf_tom)).result().get_counts()
        if '0' not in meas.keys():
            meas['0'] = 0
        elif '1' not in meas.keys():
            meas['1'] = 0
        meas_data.append(meas)
    return meas_data[0]['0'], meas_data[0]['1'], meas_data[1]['0'], meas_data[2]['0']


def wig_to_dens(arr, check = False):
    w00, w01, w10, w11 = arr[1][0], arr[1][1], arr[0][0], arr[0][1]
    alpha = -w00 + w10
    beta = w01 - w11
    a = (-1 + 1j) * alpha / 2 + (1 + 1j) * beta / 2
    b = (-1 - 1j) * alpha / 2 + (1 - 1j) * beta / 2
    if check == True:
        return np.array([[w00 + w10, a], [b , w01 + w11]]), sq.nxn_valid_quantum(np.array([[w00 + w10, a], [b , w01 + w11]]))
    else:
        return np.array([[w00 + w10, a], [b , w01 + w11]])

def num_2_wig(nh, nv, nd, nl):
    nt = nh + nv
    ph = nh / nt
    pv = nv / nt
    pd = nd / nt
    pl = nl / nt
    pa = (nt - nd) / nt
    pr = (nt - nl) / nt
    return np.array([[ph + pa + pr - 1, pv + pa + pl - 1], [ph + pd + pl - 1, pv + pd + pr - 1]]) / 2

nh, nv, nd, nl = st.text_input("H", "1000"), st.text_input("V", "0"), st.text_input("D", "500"), st.text_input("L", "500")

if nh != '' and nv != '' and nd != '' and nl != '':
	nh, nv, nd, nl = float(nh), float(nv), float(nd), float(nl)
	nt = nh + nv
	ph = nh / nt
	pv = nv / nt
	pd = nd / nt
	pl = nl / nt
	pa = (nt - nd) / nt
	pr = (nt - nl) / nt
	wig_dis_old = num_2_wig(nh, nv, nd, nl)
	# w00, w01, w10, w11 = wig_dis[1][0], wig_dis[1][1], wig_dis[0][0], wig_dis[0][1]
	# W = [w00, w01, w10, w11]
	def obj_fun(W):
	    w00, w01, w10, w11 = W
	    w00_ = (0.5 * (ph + pd + pl - 1)) - w00
	    w01_ = (0.5 * (pv + pd + pr - 1)) - w01
	    w10_ = (0.5 * (ph + pa + pr - 1)) - w10
	    w11_ = (0.5 * (pv + pa + pl - 1)) - w11
	#     print(w00_, w01_, w10_, w11_)
	    return w00_**2 + w01_**2 + w10_**2 + w11_**2
	    
	def const(W):
	    w00, w01, w10, w11 = W
	    return 1 - ((w00 + w01 - w10 - w11)**2 + (w00 - w01 - w10 + w11)**2 + (w00 - w01 + w10 - w11)**2)

	def const2(W):
	    w00, w01, w10, w11 = W
	    return w00 + w01 + w10 + w11 - 1
	    
	consts = [{'type' : 'ineq', 'fun' : const}]
	W = [1, 1, 1, 1]
	sol = minimize(obj_fun, W, constraints = consts, method = 'SLSQP').x
	wig_dis = [[sol[2], sol[3]], [sol[0], sol[1]]]
	st.write("The experimental data generated the following Wigner Distribution :")
	st.write(np.round(num_2_wig(nh, nv, nd, nl), 3))
	st.write("The Wigner Distribution after optimization:")
	st.write(np.round(wig_dis, 5))
	fig = plt.figure()
	ax = fig.add_subplot(122, projection='3d')
	ax1= fig.add_subplot(121, projection='3d')
	x_data = np.array([0,1])
	y_data = np.array([0,1])
	z_data =wig_dis_old
	z_data2=np.array(wig_dis)
	dx = dy = 0.5  # width of each bar in x and y direction
	dz = z_data.ravel()  # height of each bar
	dz1=z_data2.ravel()
	x, y = np.meshgrid(x_data, y_data)
	x, y, z = x.ravel(), y.ravel(), 0

	# Plot 3D bars
	ax.bar3d(x, y, z, dx, dy, dz)
	ax1.bar3d(x, y, z, dx, dy, dz1)
	ax.set_xlabel('Z Basis')
	ax.set_ylabel('X Basis')
	ax.set_zlabel('DWF')
	ax.set_zlim(0,1)


	ax.set_xlabel('Z Basis')
	ax.set_ylabel('X Basis')
	ax.set_zlabel('DWF')
	# ax1.set_zlim(0,1)
	st.pyplot(fig)
# np.round(wig_to_dens(wig_dis, False), 3), np.linalg.eigvals(wig_to_dens(wig_dis, False))
