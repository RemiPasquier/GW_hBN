import json
import numpy as np
import matplotlib.pyplot as plt

# Load the JSON file
with open("results-asr.gw.json") as f:
    data = json.load(f)

# Navigate into bandstructure
bs = data["kwargs"]["data"]["bandstructure"]

# Extract the ndarray
shape, dtype, flat = bs["e_int_skn"]["__ndarray__"]
energies = np.array(flat, dtype=dtype).reshape(shape)

print("Shape:", energies.shape)  # (nspin, nk, nbands)

# Example: take spin=0
bands = energies[0, :, :]  # shape (nk, nbands)

# Build a fake k-distance axis for plotting
nk = bands.shape[0]
k_axis = np.linspace(0, 1, nk)

# Plot
for ib in range(bands.shape[1]):
    plt.plot(k_axis, bands[:, ib], lw=0.8, color="black")

plt.xlabel("k-path (normalized)")
plt.ylabel("Energy (eV)")
plt.title("G₀W₀@PBE bandstructure")
plt.show()
