import sympy as sm
import numpy as np
import streamlit as st
from scipy.special import genlaguerre, laguerre, factorial
from scipy.optimize import curve_fit
from scipy.linalg import sqrtm
from scipy.special import comb
from scipy.interpolate import interp1d
import sqtdiat.qops as sq
from qutip import fidelity, Qobj, wigner
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def den_2_prob(rho, xv = np.linspace(-10, 10, 1000), theta = 0):
    prob_ = np.zeros_like(xv, dtype = 'complex128')
    if type(rho) != np.array: rho = np.array(rho)
    for n in range(rho.shape[0]):
        for m in range(n + 1):
            if np.abs(rho[n, m]) > 1e-7: 
                elem = rho[n, m]
                prob_ += elem * quad(n, m, xv, theta)
                if m != n:
                    prob_ += elem.conjugate() * quad(n, m, xv, theta)
    return np.abs(prob_)

q, theta = sm.symbols('q theta')
n = sm.symbols('n', integer = True)
m = sm.symbols('m', integer = True)

n_xtheta = sm.exp(-q**2 / 2) * sm.hermite(n, q) / sm.sqrt(2**n * sm.factorial(n)) * (1 / sm.pi) ** (1 / 4) * sm.exp(1j * (n * theta))
n_xtheta_conj = sm.exp(-q**2 / 2) * sm.hermite(m, q) / sm.sqrt(2**m * sm.factorial(m)) * (1 / sm.pi) ** (1 / 4) * sm.exp(-1j * (m * theta - sm.pi / 2))
nxt = sm.lambdify((n, q, theta), n_xtheta)

def quad(n, m, q, theta):
    return nxt(n, q, theta) * nxt(m, q, theta).conjugate()

def get_data(rho, pts = 10, angles = 360, xv = np.linspace(-10, 10, 1000)):
    data = np.zeros((pts * angles))
    thetas = np.linspace(0, 2 * np.pi, angles)
    for angle, th in zip(thetas, range(angles)):
        prob = den_2_prob(rho, xv, theta = angle)
        data[th * pts: (th + 1) * pts] = np.random.choice(a = xv, p = prob / np.sum(prob), size = (pts))
    return data

def R_(data, theta_data, den = np.eye(10)):
    R_mat = np.zeros_like(den, dtype = 'complex128')
    N, M = np.mgrid[:den.shape[0], :den.shape[0]]
    for th, q in zip(theta_data, data):
        op = quad(N, M, q, th)
        R_mat += op / np.trace(op @ den)
    return R_mat

def perform_MLE(data, theta_data, dims, iters = 20):
    rho = np.eye(dims, dtype='complex128')
    R = R_(data, theta_data, rho)
    progress_bar = st.progress(0)  # Initialize the progress bar

    for item in range(iters):
        rho = R @ rho @ R
        rho /= np.trace(rho)
        R = R_(data, theta_data, rho)

        # Update the progress bar value
        progress_bar.progress((item + 1) / iters)

    # Close the progress bar at the end
    progress_bar.empty()

    return rho.T

# Streamlit app
st.title('Quantum MLE Estimation')

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data_array = data.to_numpy()
    dims = st.number_input("Enter value for dims:", min_value=2, max_value=30, value=10, step=1)


    # Display the output of perform_MLE
    st.write("Performing MLE estimation...")
    theta_array = np.linspace(0, 2 * np.pi, len(data_array))
    result = perform_MLE(theta_data = theta_array, data = data_array, dims=dims)
    st.write("MLE Result:")
    st.write(result)

    # Visualization of Wigner distribution
    visualize_wigner = st.radio("Visualization of Wigner distribution?", ("Yes", "No"))

    if visualize_wigner == "Yes":
        xvec = np.linspace(-5, 5, 100)
        yvec = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(xvec, yvec)
        W = wigner(Qobj(result), xvec, yvec)



        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('B', fontsize=12)
        ax.set_ylabel('E', fontsize=12)
        ax.set_zlabel('W(E,B)', fontsize=12)
        ax.plot_surface(X, Y, W, cmap="viridis", lw=0, rstride=1, cstride=1)

        # Plot projections of the contours for each dimension.
        ax.contourf(X, Y, W, zdir='z', offset=-1, cmap='viridis')
        ax.view_init(elev=20, azim=45)
        ax.set(zlim=(-1, 1))
        ax.set_title('Wigner Distribution')

        st.pyplot(fig)