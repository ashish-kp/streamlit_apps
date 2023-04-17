import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
	st.write(f"{wig_dis}")
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
	ax1.set_zlim(0,1)
	st.pyplot(fig)
# np.round(wig_to_dens(wig_dis, False), 3), np.linalg.eigvals(wig_to_dens(wig_dis, False))
