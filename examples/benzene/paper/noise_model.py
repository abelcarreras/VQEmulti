import numpy as np
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error


def get_noise_model(n_qubits, multiple=1.0):
    # T1 and T2 values for qubits 0-3
    # T1s = np.random.normal(50e3, 10e3, n_qubits) * multiple  # Sampled from normal distribution mean 50 microsec
    # T2s = np.random.normal(70e3, 10e3, n_qubits) * multiple  # Sampled from normal distribution mean 50 microsec
    T1s = np.random.normal(100e3, 10e3, n_qubits) * multiple  # Sampled from normal distribution mean 50 microsec
    T2s = np.random.normal(150e3, 10e3, n_qubits) * multiple  # Sampled from normal distribution mean 50 microsec

    # Truncate random T2s <= T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(n_qubits)])

    # T1s = [10e1]*n_qubits
    # T2s = [2*10e1]*n_qubits

    # Instruction times (in nanoseconds)
    time_u1 = 0    # virtual gate
    time_u2 = 50   # (single X90 pulse)
    time_u3 = 100  # (two X90 pulses)
    time_cx = 300
    time_reset = 1000    # 1 microsecond
    time_measure = 1000  # 1 microsecond

    # QuantumError objects
    errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                    for t1, t2 in zip(T1s, T2s)]
    errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                      for t1, t2 in zip(T1s, T2s)]
    errors_u1  = [thermal_relaxation_error(t1, t2, time_u1)
                  for t1, t2 in zip(T1s, T2s)]
    errors_u2  = [thermal_relaxation_error(t1, t2, time_u2)
                  for t1, t2 in zip(T1s, T2s)]
    errors_u3  = [thermal_relaxation_error(t1, t2, time_u3)
                  for t1, t2 in zip(T1s, T2s)]
    errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
                 thermal_relaxation_error(t1b, t2b, time_cx))
                  for t1a, t2a in zip(T1s, T2s)]
                   for t1b, t2b in zip(T1s, T2s)]

    # Add errors to noise model
    noise_thermal = NoiseModel()
    for j in range(n_qubits):
        noise_thermal.add_quantum_error(errors_reset[j], "reset", [j])
        noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
        noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
        noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
        noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
        for k in range(n_qubits):
            noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])

    return noise_thermal