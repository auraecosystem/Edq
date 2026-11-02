import math
import random

# --- optional quantum simulator (qiskit). fallback to software RNG ---
use_qiskit = False
try:
    from qiskit import QuantumCircuit, Aer, execute
    use_qiskit = True
except ImportError: # Changed from generic Exception to specific ImportError
    use_qiskit = False
    print("Qiskit not found. Falling back to software RNG for quantum entropy.")

def quantum_entropy_scalar(bits=2):
    """
    Returns a float ~ (0.0, 1.0] derived from quantum measurement if qiskit is available.
    Otherwise returns a pseudo-random float from Python's RNG.
    """
    if use_qiskit:
        # small Hadamard circuit -> measure -> convert counts to a scalar
        qc = QuantumCircuit(bits, bits)
        for i in range(bits):
            qc.h(i)
        qc.measure(range(bits), range(bits))
        backend = Aer.get_backend("qasm_simulator")
        result = execute(qc, backend, shots=256).result()
        counts = result.get_counts()
        # compute a normalized entropy-like scalar
        total = sum(counts.values())
        probs = [v/total for v in counts.values()]
        entropy = -sum(p * math.log(max(p,1e-12), 2) for p in probs)
        # normalize entropy to (0.5, 1.0] roughly. Scale to be between 0 and 1.
        # Max entropy for 'bits' is log2(2^bits) = bits.
        return (entropy / bits)

    else:
        # fallback: deterministic but random-seeded scalar
        return random.random() # returns float in [0.0, 1.0)

# Example usage
if __name__ == '__main__':
    print("Generating quantum-inspired entropy scalars:")
    for _ in range(5):
        scalar = quantum_entropy_scalar(bits=4)
        print(f"Scalar: {scalar:.4f}")
    
    if not use_qiskit:
        print("\nNote: Qiskit was not available, scalars generated using Python's random module.")

