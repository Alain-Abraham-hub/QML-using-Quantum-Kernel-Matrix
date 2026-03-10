import pandas as pd
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import z_feature_map
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
import matplotlib.pyplot as plt
import seaborn as sns

# IBM Quantum Setup
print("IBM Quantum Computer Kernel Matrix Computation")
print("="*50)

# Paste your IBM API key here
IBM_API_KEY = "0MQKNQRKJlCWccwk0FBueDTPsgWQI8O3bguPWeWVl4ll" # use your own API key from ibm here do not use this one

# Initialize IBM Runtime Service
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=IBM_API_KEY)

# Find active quantum computer with least queue
def find_best_backend():
    """Find the active operational quantum computer with the least queue"""
    print("\nSearching for available quantum computers...")
    backends = service.backends(simulator=False, operational=True)

    if not backends:
        raise ValueError("No operational quantum computers available. Please try again later.")

    print(f"Found {len(backends)} operational quantum computers:")

    min_queue = float('inf')
    best_backend = None

    for backend in backends:
        try:
            status = backend.status()
            queue_depth = status.pending_jobs
        except Exception:
            queue_depth = float('inf')
        print(f"  • {backend.name}: pending jobs = {queue_depth}")

        if queue_depth < min_queue:
            min_queue = queue_depth
            best_backend = backend

    print(f"\n✓ Selected: {best_backend.name} (pending jobs: {min_queue})")
    return best_backend

# Get the best available backend
backend = find_best_backend()

# Load preprocessed data
print("\n" + "="*50)
print("Loading preprocessed data...")
df = pd.read_csv('preprocessed data.csv')

# Use a subset of samples for feasible quantum computation
MAX_SAMPLES = 50
if len(df) > MAX_SAMPLES:
    print(f"Subsampling to {MAX_SAMPLES} samples for feasible computation...")
    df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)

# Extract features (PC1-PC5)
features = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
X = df[features].values
y = df['diagnosis'].values

print(f"Data shape: {X.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")

# Set up quantum parameters
n_qubits = X.shape[1]  # 5 qubits for 5 features
reps = 2  # repetitions in the feature map

print(f"\nQuantum Circuit Configuration:")
print(f"Number of qubits: {n_qubits}")
print(f"Feature map repetitions: {reps}")

# Create feature map
feature_map = z_feature_map(feature_dimension=n_qubits, reps=reps)
print(f"Feature map circuit depth: {feature_map.decompose().depth()}")

# Build a single parameterized overlap circuit template
# Two parameter vectors: one for x_i, one for x_j
x_params  = ParameterVector('x',  n_qubits)
xp_params = ParameterVector('xp', n_qubits)

fm_x  = z_feature_map(feature_dimension=n_qubits, reps=reps)
fm_xp = z_feature_map(feature_dimension=n_qubits, reps=reps)

# Bind the two feature maps to their respective ParameterVectors
fm_x  = fm_x.assign_parameters( dict(zip(fm_x.parameters,  x_params)))
fm_xp = fm_xp.assign_parameters(dict(zip(fm_xp.parameters, xp_params)))

# Overlap circuit: U(xp) then U†(x), then measure
overlap_template = QuantumCircuit(n_qubits)
overlap_template.compose(fm_xp, inplace=True)
overlap_template.compose(fm_x.inverse(), inplace=True)
c_reg = ClassicalRegister(n_qubits, 'c')
overlap_template.add_register(c_reg)
overlap_template.measure(range(n_qubits), range(n_qubits))

# Transpile the template ONCE for the target hardware
print("\nTranspiling circuit template for the target hardware (once)...")
transpiled_template = transpile(overlap_template, backend=backend, optimization_level=1)
print(f"✓ Transpiled circuit depth: {transpiled_template.depth()}")

# Compute kernel matrix
print("\nComputing quantum kernel matrix on IBM Quantum Computer...")
print(f"Backend: {backend.name}")
n_samples = X.shape[0]
kernel_matrix = np.zeros((n_samples, n_samples))

shots = 1024
all_bitstring_counts = {}  # Accumulate bitstring frequencies across all jobs
n_elements = n_samples * (n_samples + 1) // 2

print(f"Running {shots} shots per kernel element")
print(f"Total kernel elements to compute: {n_elements}")

# ─────────────────────────────────────────────────────────────────────────────
# BATCH ALL CIRCUITS INTO A SINGLE JOB (much faster!)
# ─────────────────────────────────────────────────────────────────────────────
print("\nPreparing all circuits for batch submission...")
all_circuits = []
index_map = []  # Track (i, j) for each circuit

for i in range(n_samples):
    for j in range(i, n_samples):
        param_values = {**{x_params[k]:  X[i][k] for k in range(n_qubits)},
                        **{xp_params[k]: X[j][k] for k in range(n_qubits)}}
        bound_circuit = transpiled_template.assign_parameters(param_values)
        all_circuits.append(bound_circuit)
        index_map.append((i, j))

print(f"✓ Prepared {len(all_circuits)} circuits")
print(f"Submitting batch job to {backend.name}...")

# Run all circuits in ONE job
sampler = Sampler(mode=backend)
job = sampler.run(all_circuits, shots=shots)
print(f"Job submitted. Job ID: {job.job_id()}")
print("Waiting for results (this may take a few minutes)...")

result = job.result()
print("✓ Results received!")

# Process results
print("\nProcessing results...")
for idx, (i, j) in enumerate(index_map):
    counts = result[idx].data.c.get_counts()
    total = sum(counts.values())

    # Accumulate bitstring frequencies
    for bitstring, raw_count in counts.items():
        all_bitstring_counts[bitstring] = all_bitstring_counts.get(bitstring, 0) + raw_count / total

    # Kernel element: probability of measuring all-zeros state
    zero_state = '0' * n_qubits
    prob_zero = counts.get(zero_state, 0) / total
    kernel_matrix[i, j] = prob_zero
    kernel_matrix[j, i] = prob_zero

print("\n✓ Quantum kernel matrix computed on real quantum hardware")

# Save kernel matrix
pd.DataFrame(kernel_matrix).to_csv('quantum_kernel_matrix.csv', index=False)

# ── Plot 1: Bitstring Frequency ───────────────────────────────────────────────
sorted_bitstrings = sorted(all_bitstring_counts.items(), key=lambda x: x[1], reverse=True)
top_n = 20  # Show top 20 bitstrings
labels, freqs = zip(*sorted_bitstrings[:top_n])

fig, axes = plt.subplots(1, 2, figsize=(20, 7))

axes[0].bar(range(len(labels)), freqs, color='steelblue', edgecolor='black', alpha=0.85)
axes[0].set_xticks(range(len(labels)))
axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
axes[0].set_xlabel('Bitstring', fontsize=12)
axes[0].set_ylabel('Cumulative Probability', fontsize=12)
axes[0].set_title(f'Top {top_n} Bitstring Frequencies\n(Accumulated over all kernel evaluations)', fontsize=13)
axes[0].grid(axis='y', alpha=0.3)

# ── Plot 2: Kernel Matrix Heatmap ─────────────────────────────────────────────
sns.heatmap(kernel_matrix, cmap='viridis', ax=axes[1],
            cbar_kws={'label': 'Kernel Value'},
            xticklabels=False, yticklabels=False)
axes[1].set_title('Quantum Kernel Matrix (ZFeatureMap)\n5 Qubits, 2 Repetitions', fontsize=13)
axes[1].set_xlabel('Sample Index', fontsize=12)
axes[1].set_ylabel('Sample Index', fontsize=12)

plt.tight_layout()
plt.savefig('quantum_results.png', dpi=150)
plt.show()
print("✓ Plots saved to 'quantum_results.png'")
