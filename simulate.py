import streamlit as st
import math
from scipy.special import genlaguerre, laguerre, factorial
import gmpy2
import re
import sqtdiat.qops as sq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import plotly.graph_objs as go
import plotly.express as px
from scipy.interpolate import interp1d
from scipy.signal import convolve
from skimage import transform
import time
import pandas as pd
import base64
from functools import partial

st.set_page_config(page_title="My App", initial_sidebar_state="expanded")

plt.style.use("ggplot")

ctx = gmpy2.context()
ctx.precision = 10000

st.title(r"""$\text{Balanced Homodyne Detection}\\ \text{of Quantum States}$""")
rho_opt = st.radio("Type of state", ("State Vector", "Mixed State", "Coherent State", "Cat State", "Vacuum Squeezed State"))

show_density = True
# have to add squeezed state here, readabout the number state representation of squeezed state

def str_to_arr(inp_):
    """
    Converts a string input of comma-separated complex numbers into a numpy array.

    Args:
    inp_ (str): Input string containing comma-separated complex numbers.

    Returns:
    numpy.ndarray: Array of complex numbers extracted from the input string.
    """
    cleaned_string = re.sub(r'[^0-9j,\+\-/*\.]', '', inp_)
    return np.array([complex(x) for x in cleaned_string.split(',')])

def wig_coherent(alpha, xmin, xmax, pmin, pmax, res = 200, g = np.sqrt(2), return_vecs = False):
    """
    Generates a normalized Wigner distribution for a coherent state in phase space.

    Args:
    alpha (complex): Complex number defining the coherent state.
    xmin (float): Minimum value for the x-axis.
    xmax (float): Maximum value for the x-axis.
    pmin (float): Minimum value for the momentum (p) axis.
    pmax (float): Maximum value for the momentum (p) axis.
    res (int, optional): Resolution for the grid. Defaults to 200.
    g (float, optional): Parameter for state representation. Defaults to sqrt(2).
    return_vecs (bool, optional): Whether to return coordinate vectors. Defaults to False.

    Returns:
    numpy.ndarray or tuple: Normalized Wigner distribution for the coherent state or 
                            tuple with distribution and coordinate vectors (if return_vecs=True).
    """
    if xmax - xmin < 3 * np.abs(alpha) or pmax - pmin < 3 * np.abs(alpha):
        st.write(f"Unchangeable dimensions {2 * np.abs(alpha)}")
        xmin, xmax, pmin, pmax = -2 * np.abs(alpha), 2 * np.abs(alpha), -2 * np.abs(alpha), 2 * np.abs(alpha)
    xvec = np.linspace(xmin, xmax, res)
    pvec = np.linspace(pmin, pmax, res)
    X, P = np.meshgrid(xvec, pvec)
    wig = np.exp((-(X - g * np.real(alpha))**2 - (P + g * np.imag(alpha))**2))
    norm_wig = wig / np.sum(wig)
    if return_vecs == True:
        return norm_wig, xvec, pvec
    return norm_wig
    
def wig_vac_squeezed(r, theta, res = 200, return_axes = False):
    """
    Generates the Wigner distribution for a vacuum squeezed state in phase space.

    Args:
    r (float): Squeezing parameter.
    theta (float): Phase angle in degrees.
    res (int, optional): Resolution for the grid. Defaults to 200.
    return_axes (bool, optional): Whether to return coordinate axes. Defaults to False.

    Returns:
    numpy.ndarray or tuple: Wigner distribution for the vacuum squeezed state or 
                            tuple with distribution and coordinate axes (if return_axes=True).
    """
    xv = np.linspace(-10, 10, res)
    X, P = np.meshgrid(xv, xv)
    th = np.deg2rad(theta)
    wig = np.exp(-2 * ((X * np.cos(th) + P * np.sin(th))**2 * np.exp(-2 * (r)) + (-X * np.sin(th) + P * np.cos(th))**2 * np.exp(2 * r))) * 2 / np.pi
    if return_axes == True:
        return wig, xv, xv
    return wig

def wig_loss(wig_dis, eta, xvec, pvec):
    """
    Applies loss to a given Wigner distribution.

    Args:
    wig_dis (numpy.ndarray): Input Wigner distribution.
    eta (float): Loss parameter.
    xvec (numpy.ndarray): x-axis values.
    pvec (numpy.ndarray): p-axis values.

    Returns:
    numpy.ndarray: Wigner distribution after applying loss.
    """
    X, P = np.meshgrid(xvec, pvec)
    s = eta / (1 - eta)
    s_arr = np.exp(-s * (X**2 + P**2))
    s_arr /= np.sum(s_arr)
    return convolve(wig_dis, s_arr, mode = 'same')

def fact1(n):
    return float(gmpy2.sqrt(gmpy2.fac(n)))

def coh(n, alpha):
    base = np.exp(-np.abs(alpha)**2)
    coh_arr = np.zeros((n), dtype=object)
    for i in range(n):
        coh_arr[i] = alpha**i / gmpy2.sqrt(gmpy2.factorial(i))
    return np.array(np.array(coh_arr, dtype=object) * base, dtype = "complex128")

def coh2(n, alpha):
    base = np.exp(-np.abs(alpha)**2)
    coh_arr = np.zeros((n), dtype=object)
    for i in range(n):
        coh_arr[i] = gmpy2.div(alpha**i, gmpy2.sqrt(fact1(i)))
    return np.array(coh_arr, dtype = object) * base

def rho_input(inp_, type = "state_vec"):
    types = ['state_vec', 'mixed_state', 'coherent', 'cat_states', 'vac_squeezed_states']
    if type not in types:
        raise ValueError("Please enter allowed type")
    else:
        if type == 'state_vec':
            if inp_.shape[0] < 2:
                inp_ = np.array([1, 0])
            inp_ = sq.norm_state_vec(inp_)
            return np.outer(inp_, np.conj(inp_))
        elif type == 'mixed_state':
            # inp_ = 
            pass
        elif type == 'coherent':
            # coh_size = st.radio("")
            N, alpha = inp_
            coh_state = np.array(coh(N, alpha), dtype = 'complex128')
            return np.outer(coh_state, np.conj(coh_state))
        elif type == 'cat_states':
            N, alphas, c_is = inp_
            norm_c_is = sq.norm_state_vec(c_is)
            cat_st = np.zeros((N), dtype = 'complex128')
            for c_i, alpha in zip(norm_c_is, alphas): cat_st += c_i * np.array(coh(N, alpha), dtype = 'complex128')
            return np.outer(cat_st, np.conj(cat_st))
        # elif type == 'coherent':

# def rad(img, )

def wigner_laguerre(rho, x_min = -10, x_max = 10, p_min = -10, p_max = 10, res = 200, return_axes = False, fixed = False):
    """A very basic code for obtaining the wigner distribution from the density matrix. 
    Refs:
    1. Ulf leonhardt - Measuring the Quantum States of Light. Chapter 5, Section 5.2.6 pg.nos 128-129
    2. "Numerical study of Wigner negativity in one-dimensional steady-state resonance fluorescence"
    arXiv: 1909.02395v1 Appendix A
    """
    if type(rho) != np.array:
        rho = np.array(rho)
    if rho.shape[0] == rho.shape[1]:
        x_vec = np.linspace(x_min, x_max, res)
        p_vec = np.linspace(p_min, p_max, res)
        X, P = np.meshgrid(x_vec, p_vec)
        A = X + 1j * P
        B = np.abs(A)**2
        W = np.zeros((res, res))
        for n in range(rho.shape[0]):
            if np.abs(rho[n, n]) > 0:
                W += np.real(rho[n, n] * (-1) ** n * genlaguerre(n, 0)(2 * B))
            for m in range(0, n):
                if np.abs(rho[m, n]) > 0:
                    W += 2 * np.real(rho[m, n] * (-1) ** m * np.sqrt(2 ** (n - m) * factorial(m) / factorial(n)) * genlaguerre(m, n - m)(2 * B) * A ** (n - m))
        W = W * np.exp(-B)  / np.pi
        if return_axes == True: return W / np.sum(W), x_vec, p_vec 
        return W / np.sum(W)
    else:
        raise ValueError("Dim. mismatch between rows and columns")

def sim_homodyne_data(wg, xv, theta_steps = 180, ADC_bits = 8, pts = 100, need_elec_noise = True, elec_var = 0.3, data_res = 10):
    """
    Simulates homodyne data with optional electronic noise and detector losses.

    Args:
    wg (numpy.ndarray): Wigner distribution.
    xv (numpy.ndarray): x-axis values.
    theta_steps (int): Number of quadratures to be measured in one whole period.
    ADC_bits (int): ADC used for sampling.
    pts (int): Number of measurements per quadrature.
    need_elec_noise (bool): Flag to include electronic noise.
    elec_var (float): Relative Variance of Electronic Noise w.r.t Vacuum Noise.
    data_res (int): Spacing between the discrete values obtained.

    Returns:
    numpy.ndarray: Simulated homodyne data.
    """
    # ADC_bits = 8
    thetas = np.linspace(0, 359, theta_steps)
    # pts = 100
    mask = np.array([1 if x % data_res == 0 else 0 for x in range(2**ADC_bits * data_res)])
    all_data = np.zeros((thetas.shape[0] * pts))
    # elec_var = 0.4
    for i, t in enumerate(thetas):
        f = interp1d(xv, transform.rotate(wg, t).sum(0))
        discrete_p = f(np.linspace(xv[0], xv[-1], 2**ADC_bits * data_res)) * mask
        discrete_p = discrete_p / np.sum(np.abs(discrete_p))
        # if np.sum(np.abs(discrete_p)) != 1: print(np.sum(np.abs(discrete_p)))
        data = np.random.choice(np.linspace(xv[0], xv[-1], 2**ADC_bits * data_res), p = np.abs(discrete_p), size = (pts))
        all_data[i * pts: (i + 1) * pts] = data
    if need_elec_noise == True:
        elec_f = np.exp(-(xv)**2 / elec_var)
        elec_fun = interp1d(xv, elec_f)
        elec_p = elec_fun(np.linspace(xv[0], xv[-1], 2**ADC_bits * data_res)) * mask
        elec_p /= np.sum(elec_p)
        elec_noise = np.random.choice(np.linspace(xv[0], xv[-1], 2**ADC_bits * data_res), p = np.abs(elec_p), size = (thetas.shape[0] * pts))
        return all_data + elec_noise
    return all_data

def download_csv(df):
    """
    Downloads simulated data as a CSV file.

    Args:
    df (pandas.DataFrame): Dataframe containing the simulated data.

    Returns:
    str: HTML formatted link to download the CSV file.
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="sim_data.csv">Simulated data in csv format</a>'
    return href

def perform_interpolation(arr1, arr2, m=100, x_min = -5, x_max = 5, padding=1.0, kind='cubic'):
    # Find non-zero elements and their indices
    non_zero_indices = np.nonzero(arr1)
    filtered_arr1 = arr1[non_zero_indices]
    filtered_arr2 = arr2[non_zero_indices]

    # Create x_new array for interpolation
    x_new = np.linspace(padding * x_min, padding * x_max, m)
    
    # Perform interpolation based on the specified kind (default: cubic)
    interp_func = interp1d(filtered_arr2, filtered_arr1, kind=kind, bounds_error = False, fill_value = 0)
    
    # Generate interpolated data for the x_new array
    interpolated_data = interp_func(x_new)
    
    return x_new, interpolated_data

def meas_data_2_hist(sim_data, theta, data_points, dat_min, dat_max, bins, m = 360):
    # pts = 360
    # padding = 1.2
    # dat_min, dat_max = np.min(sim_data), np.max(sim_data)
    full = np.zeros((m, theta))
    for i in range(theta):
        a, b = np.histogram(sim_data[i * data_points: (i + 1) * data_points], bins = bins)
        _, c = perform_interpolation(a, b, m = m, x_min = dat_min, x_max = dat_max)
        full[:, i] = np.abs(c)[::-1]
    return full

def filter(size):
    """Copied this filter from skimage.transform.iradon - Source code page"""
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                        np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    f_ = np.fft.ifft(f)
    return np.real(f_[:, np.newaxis] / np.max(f_))

def irad(hist_2d, xvec = np.linspace(-5, 5, 200), filter_name = "ramp", thetas = 360):
    """
    This function is a copied version of the skimage.transform.iradon,
    where I have removed certain parts of the original, and replaced variable names.
    please refer to the original function, to learn more about it.
    """
    # The parameter thetas is redundant, but I'm using it as of now.
    ## This is not a genralized function, I want it to work specifically for wigner distributions. 
    ## For generalized operations, please consider using skimage.transform.iradon
    # # Using mgrid instead of meshgrid as, here, we are only concerned with the 
    # # size of the array, and not the points it represents.
    # # Creating 2d arrays whose indices will be filled with interpolation
    # # as the np.pad function is slower, but would seem like a simpler and 
    # # more intuitive approach.
    # # # Using functools.partial insteal of np.interp1d (Much faster, need to see why EXACTLY)
    # # # final_img += interp1d(x = x_arr, y = filtered_img[:, theta], kind = "cubic", bounds_error = False, 
    # # # fill_value = 0)(p * np.cos(np.deg2rad(theta)) - x * np.sin(np.deg2rad(theta)))
    
    all_thetas = np.linspace(0, 360, thetas)
    if hist_2d.shape[0] != hist_2d.shape[1]:
        return ValueError("Please enter a square matrix of size n x n, where n is a multiple of 180.")
    if filter_name == "ramp":
        filt = filter(hist_2d.shape[0])
    filtered_img = np.real(np.fft.ifft(np.fft.fft(hist_2d, axis = 0) * filt, axis = 0)[:hist_2d.shape[1], :])
    final_img = np.zeros((hist_2d.shape))
    x, p = np.mgrid[:hist_2d.shape[0], :hist_2d.shape[0]] - hist_2d.shape[0] // 2
    x_arr = np.arange(hist_2d.shape[0]) - hist_2d.shape[0] // 2
    for col, theta in zip(range(hist_2d.shape[0]), all_thetas):
        final_img += partial(np.interp, xp = x_arr, fp = filtered_img[:, col], left = 0, right = 0)(-x * np.sin(np.deg2rad(theta)) - p * np.cos(np.deg2rad(theta)))
    return final_img / (2 * np.pi * thetas)

fin_inp = 0
no_inp = False
sim_data = np.array([0])

if rho_opt == "State Vector":
    inp_ = st.text_input("Unnormalized probability amplitudes. Eg: 1, 0, 1j, -2")
    if inp_ != "":
        inp_ = str_to_arr(inp_)
        fin_inp = rho_input(inp_, type = "state_vec")

elif rho_opt == "Mixed State":
    inp0_ = st.number_input("Enter the number of states you wish to mix. Minimum : 2, Maximum : 10", min_value = 2, max_value = 10, value = 2)
    st.latex(r"""\text{Enter the statevectors. Please make sure every state vector}\\
    \text{has the same dimension.}""")
    state_1 = st.text_input("Enter the unnormalized statevector here. Eg: 1, 0, 1j", value = "1, 0")
    st_1 = str_to_arr(state_1)
    # st.write(st_1)
    state_vec = np.zeros((inp0_, st_1.shape[0]))
    state_vec[0] = st_1
    for i in range(inp0_ - 1):
        state_n = st.text_input("Enter the statevector with same dimension as above.", value = "0, 1", key = f"{i}")
        st_n = str_to_arr(state_n)
        state_vec[i + 1] = st_n
        if st_1.shape[0] != st_n.shape[0]:
            raise ValueError("Please make sure all the statevectors have the same dimensions.")
    inp2_ = st.text_input(f"Enter the {inp0_} classical probabilities (unnormalized) for these states.")
    if inp2_ != "":
        class_probs = np.real(str_to_arr(inp2_))
        if len(class_probs) != inp0_:
            st.write(class_probs, inp0_)
            st.text("Mismatch in no. of statevectors and no. of given classical probabilities.")
        else:
            class_probs /= np.sum(class_probs)
            fin_inp = np.zeros((st_1.shape[0], st_1.shape[0]))
            for (p, state) in zip(class_probs, state_vec):
                fin_inp += p * np.outer(state, np.conj(state))
            # st.write(fin_inp)
        
elif rho_opt == "Coherent State":
    st.latex(r'|\alpha \rangle = e^{-|\alpha|^2} \sum_{n=0}^{\infty} \dfrac{\alpha^n}{\sqrt{n!}} |n\rangle')
    st.latex(r"""\text{Coherent states with N values less than 200 allow us to }\\
    \text{view the density matrices, but take some time to compute element by element.}""")
    st.latex(r"""\text{For higher } \alpha \text{ values, select the option "Direct Wigner Distribution"}\\
    \text{where, the Wigner Distribution is directly obtained from the function:}""")
    st.latex(r"W_{\alpha}(x, p)=\dfrac{1}{\pi}e^{-(x-\sqrt{2}Re(\alpha))^2 - (p - \sqrt{2}Im(\alpha))^2}")
    alpha_rng = st.radio("View density matrix?", ("Yes", "No - Direct Wigner Distribution"))
    if alpha_rng == "Yes":
        inp1_ = st.number_input("Dimension of the state vector N, maximum value is 200", min_value = 10, max_value = 200, step = 1, value = 20)
        inp2_ = st.text_input(r"$alpha$ value (complex), at maximum be less than or equal to $\sqrt{\dfrac{N}{2}}$")
        if inp1_ != "" and inp2_ != "":
            alpha = str_to_arr(inp2_)[0]
            inp_ = (inp1_, alpha)
            if alpha > np.sqrt(inp1_ / 2):
                raise ValueError("Alpha Value should be less than sqrt(N / 2)")
            fin_inp = rho_input(inp_, type = "coherent")
    elif alpha_rng == "No - Direct Wigner Distribution":
        inp_ = st.text_input(r"$\alpha$ value $\in \mathbb{C}$, should be less than 100")
        if inp_ != "":
            alpha = str_to_arr(inp_)[0]
            show_density = False
            no_inp = True
        else:
            no_inp = False

elif rho_opt == "Cat State":
    st.latex(r'|cat\rangle = \sum_{i} c_i |\alpha\rangle')
    inp0_ = st.number_input("Dimension of the state vector, maximum value is 200", min_value = 10, max_value = 200, step = 1, value = 20)
    inp1_ = st.number_input("How many coherent states do you wish to superpose?, maximum value is 10", min_value = 1, max_value = 10, value = 2)
    if inp1_ <= 10:
        inp2_ = st.text_input(r"Enter the complex $\alpha$ values, ideally should be less than or equal to $\sqrt{\dfrac{N}{2}}$", value = "1,-1")
        inp3_ = st.text_input(r"Enter the probability (unnormalized) amplitudes (ratios) of each cat state", value = "1, 1")
    else:
        "Please enter a value less than or equal to 10."
    if inp1_ != "":
        alphas = str_to_arr(inp2_)
        c_is = str_to_arr(inp3_)
        if len(alphas) == inp1_ and len(c_is) == inp1_:
            if len(alphas) == len(c_is):
                for x in alphas: 
                    if x > np.sqrt(inp0_ / 2): 
                        raise ValueError("One of the given alpha values is greater than sqrt(N / 2)")
                fin_inp = rho_input((inp0_, alphas, c_is), type = "cat_states")
            else:
                st.write(f"The number of alphas is {len(alphas)} and prob. amplitudes is {len(c_is)}")
        else:
            st.write(f"Mismatch between the entered dimensions {inp1_} and the given alpha values {len(alphas)} or prob. amplitude values {len(c_is)}")

elif rho_opt == "Vacuum Squeezed State":
    st.latex(r"""\text{Vacuum Squeezed States are obtained here, using the Wigner Distribution} \\
    W_{|0(r, \phi)\rangle}(x,p) = \dfrac{2}{\pi} e^{2((xcos\phi + psin\phi)e^{2r} + (-xsin\phi+pcos\phi)e^{-2r})}""")
    inp0_ = st.slider("Enter r value- squeezing parameter", min_value = 0.0, max_value = 2.0, value = 0.5)
    inp1_ = st.slider(r"Enter phase angle $\phi$", min_value = 0, max_value = 360, value = 0)
    r, theta = inp0_, inp1_
    show_density = False

if type(fin_inp) == np.ndarray and show_density == True:
    fig = plt.figure(figsize=(12, 6))
    if fin_inp.shape[0] > 30:
        fig, ax = plt.subplots(1, 2, figsize = (6, 6), dpi = 100)
        im1 = ax[0].imshow(np.real(fin_inp), cmap = "PiYG")
        ax[0].set_title(f'Real part')
        ax[0].axis("off")
        plt.colorbar(im1, ax=ax[0], shrink = 0.2)
        im2 = ax[1].imshow(np.imag(fin_inp), cmap = "PiYG")
        ax[1].set_title(f'Imaginary part')
        ax[1].axis("off")
        plt.colorbar(im2, ax=ax[1], shrink = 0.2)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust the top margin
        plt.figtext(0.5, 0.75, f"{fin_inp.shape[0]} x {fin_inp.shape[0]} Density Matrix of the \n superposed number state", ha='center', va='top', fontsize=14)  # Add the super title
        st.pyplot(fig)
    else:
        data = fin_inp
        ax_real = fig.add_subplot(121, projection='3d')
        xpos, ypos = np.meshgrid(np.arange(fin_inp.shape[0]), np.arange(fin_inp.shape[1]), indexing="ij")
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros_like(xpos)
        real_values = data.real.flatten()
        norm_real = Normalize(vmin=real_values.min(), vmax=real_values.max())
        colors_real = plt.cm.PiYG(norm_real(real_values))
        ax_real.bar3d(xpos, ypos, zpos, 0.5, 0.5, real_values, color=colors_real, alpha=0.4)
        ax_real.set_title('Real Part')
        ax_real.set_xlabel('X')
        ax_real.set_ylabel('Y')
        ax_real.set_zlabel('Real Value')
        ax_real.set_zlim((1.2 * np.min(fin_inp), 1.2 * np.max(fin_inp)))
        sm_real = ScalarMappable(cmap='PiYG', norm=norm_real)
        sm_real.set_array([])  # Dummy array to allow using a colorbar
        cbar_real = fig.colorbar(sm_real, ax=ax_real, pad=0.05)
        cbar_real.set_label('Real Value Color Map')
        ax_imag = fig.add_subplot(122, projection='3d')
        imag_values = data.imag.flatten()
        norm_imag = Normalize(vmin=imag_values.min(), vmax=imag_values.max())
        colors_imag = plt.cm.PiYG(norm_imag(imag_values))
        ax_imag.bar3d(xpos, ypos, zpos, 0.5, 0.5, imag_values, color = colors_imag, alpha = 0.4)
        ax_imag.set_title('Imaginary Part')
        ax_imag.set_xlabel('X')
        ax_imag.set_ylabel('Y')
        ax_imag.set_zlabel('Imaginary Value')
        ax_imag.set_zlim((1.2 * np.min(fin_inp), 1.2 * np.max(fin_inp)))
        sm_imag = ScalarMappable(cmap='PiYG', norm=norm_imag)
        sm_imag.set_array([])  # Dummy array to allow using a colorbar
        cbar_imag = fig.colorbar(sm_imag, ax=ax_imag, pad=0.1)
        cbar_imag.set_label('Imaginary Value Color Map')
        plt.figtext(0.5, 0.95, f"{fin_inp.shape[0]} x {fin_inp.shape[0]} Density Matrix of the Coherent State", ha='center', va='top', fontsize=14)  # Add the super title
        st.pyplot(fig)

if type(fin_inp) == np.ndarray or show_density == False:
    st.write(r"### $\text{Wigner Distribution}$")
    st.latex(r"""
    
    \text{Striations or artefacts in the Wigner Distribution might}\\
    \text{be due to premature truncation of the density matrix. Try increasing the} \\
    \text{dimensions of the density matrix N, if the artefacts persist for non-negative} \\
    \text{Wigner Distributions.}
    
    """)
    col1, col2, col3 = st.columns(3) 
    with col1:
        x_min = st.number_input("Minimum x value", min_value = -20, max_value = 20, value = -10)
        p_min = st.number_input("Minimum p value", min_value = -20, max_value = 20, value = -10)
    with col2:
        x_max = st.number_input("Maximum x value", min_value = -20, max_value = 20, value = 10)
        p_max = st.number_input("Maximum p value", min_value = -20, max_value = 20, value = 10)
    with col3:
        res = st.number_input("Resolution , maximum is 400", min_value = 100, max_value = 1000, value = 200)
        # p_res = st.number_input("Resolution in p-axis, maximum is 400", min_value = 100, max_value = 400, value = 200)
    if show_density == False:
        if rho_opt == "Coherent State" and no_inp:
            if alpha < 100:
                wig_dist, xv, pv = wig_coherent(alpha, x_min, x_max, p_min, p_max, res, return_vecs = True)
            else:
                raise ValueError("alpha value should be less than 100.")
        elif rho_opt == "Vacuum Squeezed State":
            wig_dist, xv, pv = wig_vac_squeezed(r, theta, res = res, return_axes = True)
    else:
        wig_start_time = time.time()
        wig_dist, xv, pv = wigner_laguerre(fin_inp, x_min, x_max, p_min, p_max, res, return_axes = True)
        wig_end_time = time.time()
        st.text(f"Time taken to calculate the wigner distribution {np.round(wig_end_time - wig_start_time, 5)} seconds")
    # 2D plot 
    
    wig_plot_opt_2 = st.radio("Visualize after loss", ["2D Plot", "3D Plot"])
    if wig_plot_opt_2 == "2D Plot":
        fig = px.imshow(wig_dist, x = xv, y = pv)
        st.plotly_chart(fig)
    
    # 3D plot
    elif wig_plot_opt_2 == "3D Plot":
        fig = go.Figure(data=[go.Surface(z=wig_dist, x=xv, y=pv)])
        fig.update_layout(title='Wigner Distribution', autosize=False, width=800, height=600, scene = dict(
            xaxis=dict(title='P'),
            yaxis=dict(title='X')))
        fig.update_traces(colorscale='turbo')
        st.plotly_chart(fig)
    
    st.write(r"### $\text{Simulated Data - Fixed Arbitrary Gain}$") 
    st.write(r"#### $\text{Losses due to Responsivity of Detector and Electronic Noise}$")
    st.latex(r"""\text{The homodyne data, for the above entered density matrix}\\
    \text{simulated, taking into consideration some experimental impediments.}\\
    \text{assuming responsivity of the detector is non-ideal.}""")
    phases = st.number_input("No. of quadratures to be measured in one whole period.", min_value = 10, max_value = 360, value = 360)
    pts = st.number_input("No. of measurements per quadrature.", min_value = 10, max_value = 1000, value = 300)
    ADC_bits = st.number_input("ADC used for sampling", min_value = 4, max_value = 16, value = 8)
    st.latex(r"\text{Optional - Select the spacing between the discrete values obtained.}")
    data_res = st.number_input("Varies from 3 to 15", min_value = 3, max_value = 15, value = 8)

    detect_eff = st.radio("Add losses due to non-ideal quantum efficiency?", ("Yes", "No"))
    if detect_eff == "Yes":
        q_eff = st.slider("Quantum efficiency of detector", min_value = 0.4, max_value = 0.99, value = 0.85)
        wig_dist = wig_loss(wig_dist, eta = q_eff, xvec = xv, pvec = pv)
        st.write(r"$\text{Wigner Distribution after detector loss.}$")
        wig_plot_opt = st.radio("Visualization", ["2D Plot", "3D Plot"])
        if wig_plot_opt == "2D Plot":
            fig = px.imshow(wig_dist, x = xv, y = pv)
            st.plotly_chart(fig)
        
        # 3D plot
        elif wig_plot_opt == "3D Plot":
            fig = go.Figure(data=[go.Surface(z=wig_dist, x=xv, y=pv)])
            fig.update_layout(title='Wigner Distribution', autosize=False, width=800, height=600, scene = dict(
                xaxis=dict(title='P'),
                yaxis=dict(title='X')))
            fig.update_traces(colorscale='turbo')
            st.plotly_chart(fig)
    
    elec_noise_add = st.radio("Add electronic noise?", ("Yes", "No"))
    if elec_noise_add == "Yes":
        need_elec_noise = True
        elec_noise_var = st.slider("Relative Variance of Electronic Noise w.r.t Vacuum Noise. Range - 0.05 to 0.7", 
                                    min_value = 0.05, max_value = 0.8, value = 0.4)
    elif elec_noise_add == "No":
        need_elec_noise = False
        elec_noise_var = 0
    point_size = st.slider("Point size in graph", min_value = 0.01, max_value = 1.0, value = 0.1)
    # st.write(point_size)
    # if st.button("Simulate"):
    sim_data = sim_homodyne_data(wig_dist, xv, ADC_bits = ADC_bits, theta_steps = phases, need_elec_noise = need_elec_noise, pts = pts, elec_var = elec_noise_var)   
    phase_dat = np.repeat(np.linspace(0, 360, phases), pts)
    # st.write(phase_dat.shape[0], sim_data.shape[0])
    fig, ax = plt.subplots()
    ax.scatter(np.linspace(0, 360, phase_dat.shape[0]), sim_data, s = point_size)
    ax.set_xlabel('Arbitrary time')
    ax.set_ylabel('Arbitrary BHD Voltage Output')
    ax.set_title('Simulated Homodyne Data')
    st.pyplot(fig)

    st.latex(r"""\text{Below link can be used to download the above}\\
    \text{simulated data as a csv file.}""")

    df = pd.DataFrame({'Phase': phase_dat, 'Simulated Homodyne Data': sim_data})
    st.markdown(download_csv(df), unsafe_allow_html=True)

    st.title(r"$\text{Wigner State Reconstruction}$")
    st.latex(r"""\text{With the quadrature data obtained from BHD}\\
    \text{limited by our detector losses, electronic noise}\\
    \text{and the discreteness of the ADC, we shall}
    \\ \text{attempt to reconstruct the Wigner State}
    \\ \text{and subsequently, the density matrix.}""")

    if sim_data.shape[0] > 1:
        # padding = st.slider("Padding ratio) to be added on the ends of the data", min_value = 1.0, max_value = 2.0, value = 1.2)
        # st.write("Padding 1 is no padding, 1.1 is 10% padding on both sides and so on.")
        bins = st.slider("No. of bins to be used. Increase to capture smaller changes in data.", min_value = 4, max_value = 50, value = 10)
        fig, ax = plt.subplots()
        hist_2d = meas_data_2_hist(sim_data, theta = phases, data_points = pts, dat_min = xv[0], dat_max = xv[-1], bins = bins, m = phases)
        # st.write(hist_2d)
        ax.imshow(hist_2d)
        st.pyplot(fig)

        # wig_plot_opt_3 = st.radio("Sinogram", ["2D Plot", "3D Plot"])
        # if wig_plot_opt_3 == "2D Plot":
        #     fig = px.imshow(hist_2d, x = np.arange(phases), y = np.linspace(10 * xv[0], 10 * xv[-1], phases))
        #     st.plotly_chart(fig)
    
        # # 3D plot
        # elif wig_plot_opt_3 == "3D Plot":
        #     fig = go.Figure(data=[go.Surface(z = hist_2d, x = xv, y = pv)])
        #     fig.update_layout(title='Sinogram', autosize = False, width = 800, height = 600, scene = dict(
        #         xaxis=dict(title='Distribution'), 
        #         yaxis=dict(title='Phases')))
        #     fig.update_traces(colorscale='turbo')
        #     st.plotly_chart(fig)

        fig, axs = plt.subplots(1, 2, figsize = (12, 5))
        axs[0].imshow(irad(hist_2d, thetas = phases))
        axs[0].axis('off')
        axs[1].imshow(wig_dist)
        axs[1].axis('off')

        st.pyplot(fig)
