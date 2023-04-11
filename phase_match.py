import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize 
import streamlit as st

def sellmeier_bbo(lmbda, ray):
    if ray == 'o':
        return np.sqrt(2.7405 + (0.0184 / (lmbda**2 - 0.0179)) - (0.0155 * lmbda**2))
    elif ray == 'e':
        return np.sqrt(2.3753 + (0.01224 / (lmbda**2 - 0.01667) - 0.01516 * lmbda**2))
def ref_angle(n0, n90, angle):
    return np.sqrt((n0**2 * n90**2) / (n90**2 * np.cos(angle)**2 + n0**2 * np.sin(angle)**2))
ls = 0.810
lp = 0.405
theta_s = 3 * np.pi / 180

# phase_matching_angle(nop, nEp, nos, 3 * np.pi / 180)

st.title("Phase Matching Angle for BBO Crystal")
st.text("Non-collinear Degenerate Spontaneous Parametric Down Conversion")

lp = st.text_input("Enter wavelength of pump in nm", 405)
ls = st.text_input("Enter wavelength of signal or idler in nm", 810)
theta_s = st.text_input("Enter angle of seperation", 3)
theta_s = float(theta_s) * np.pi / 180
lp, ls = float(lp) / 1000, float(ls) / 1000

nop = sellmeier_bbo(lp, 'o')
nEp = sellmeier_bbo(lp, 'e')
nos = sellmeier_bbo(ls, 'o')
nEs = sellmeier_bbo(ls, 'e')

def type_1(theta):
    return np.abs(ref_angle(nop, nEp, theta) - (nos * np.cos(theta_s)))

def type_2(theta):
    return np.abs(2 * ref_angle(nop, nEp, theta) - (np.cos(theta_s) * (ref_angle(nos, nEs, theta) + nos)))

if lp != '' and ls != '' and theta_s != '' and st.button("Calculate"):
	st.write(f"Type-I = {(minimize(type_1, 1).x * 180 / np.pi)[0]}")
	st.write(f"Type-II = {(minimize(type_2, 1).x * 180 / np.pi)[0]}")
