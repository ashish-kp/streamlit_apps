import streamlit as st
import pennylane as qml
import numpy as np

# Define the gates and helper functions
X, Y, Z = np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])
R = lambda gate, theta : np.linalg.matrix_power(1j * gate * theta / 2)
CU = lambda gate: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, gate[0, 0], gate[0, 1]], [0, 0, gate[1, 0], gate[1, 1]]])
P = lambda phase : np.array([[1, 0], [0, np.exp(1j * phase)]])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
I = np.eye(2)
state_vec = np.array([1, 0, 0, 0])

dev_ipe = qml.device('default.qubit', wires = 2)

@qml.qnode(dev_ipe)
def ipe_circ(params):
    U, basis_st, phase_shift, power = params
    qml.Hadamard(wires = [0])
    qml.QubitStateVector(basis_st, wires = [1])
    qml.ControlledQubitUnitary(np.linalg.matrix_power(U, power), [0], [1])
    qml.PhaseShift(2 * np.pi * phase_shift, wires = [0])
    qml.Hadamard(wires = [0])
    return qml.probs(wires = [0])

# Streamlit app
st.title('Iterative Quantum Phase Estimation')

# User inputs
t = st.number_input('Parameter t', min_value=0.0, max_value=1.0, value=0.5, step=0.0001)
num_steps = st.number_input('Number of steps', min_value=1, max_value=20, value=8)

if st.button('Run IPE'):
    U_gate = P(2 * np.pi * t)
    basis_state = np.array([0, 1])
    phase_shift = 0.0

    acc_phase = 0
    cl_phs = []

    if ipe_circ([U_gate, basis_state, 0, 2**(num_steps - 1)])[1] > 0.5:
        acc_phase += 1 / 2**(num_steps + 1)
        cl_phs.append(1)
    else:
        cl_phs.append(0)

    phase_shift = -acc_phase * 2 ** (num_steps - 2)

    for i in range(num_steps - 2, -1, -1):
        if ipe_circ([U_gate, basis_state, phase_shift, 2**i])[1] > 0.5:
            acc_phase += 1 / 2**(i + 1)
            cl_phs.append(1)
        else:
            cl_phs.append(0)
        phase_shift = -acc_phase * 2 ** (i - 1)

    st.write(f"Accumulated Phase: {acc_phase}")
    st.write(f"Classical Phases: {cl_phs[::-1]}")
    st.write(f"Original Parameter t: {t}")

# Run the app using: streamlit run ipe_streamlit_app.py
