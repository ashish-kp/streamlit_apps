import streamlit as st
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
import pyperclip

def generate_qcircuit_latex_only(circuit):
    full_latex_code = circuit_drawer(circuit, output='latex_source')
    start_index = full_latex_code.find(r'\begin{document}') + len(r'\begin{document}') + 1
    end_index = full_latex_code.find(r'\end{document}')
    circuit_only_code = full_latex_code[start_index:end_index].strip()
    return circuit_only_code

st.title("Qiskit Circuit to LaTeX Converter")
st.write("Please make sure the following packages are available in your latex file.")

st.write('\\usepackage[braket, qm]{qcircuit}')
st.write('\\usepackage{graphicx}')

qiskit_code = st.text_area("Paste your Qiskit code here (defining a 'qc' variable):", 
"""
# Example Qiskit code:
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all() 
""")  # Default code defining 'qc'


if qiskit_code:
    # Assume the user has defined a 'qc' variable for the QuantumCircuit
    try:
        # Get the 'qc' variable from the user's code
        # This assumes the user has defined 'qc' in their pasted code
        # If not, it will use the default 'qc' defined above
        # locals().update(exec(qiskit_code)) 
        exec(qiskit_code, globals())
        # qc = globals().get('qc')  
        qc = next((obj for name, obj in globals().items() if isinstance(obj, QuantumCircuit)), None)


        if qc is not None and isinstance(qc, QuantumCircuit):
            latex_code = generate_qcircuit_latex_only(qc)
            st.code(latex_code, language="latex")

            # if st.button("Copy to Clipboard"):
            #     pyperclip.copy(latex_code)
            #     st.success("LaTeX code copied to clipboard!")
        else:
            st.error("Please define a 'qc' variable for your QuantumCircuit in the pasted code.")
    except Exception as e:
        st.error(f"Error processing Qiskit code: {e}")
