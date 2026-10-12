
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load LTspice inverter DC-sweep data
# ---------------------------------------------------------
df = pd.read_csv("inverter_vtc.txt", sep=r"\s+")

vin = df["Vin"].to_numpy(dtype=float)
vout = df["V(vout)"].to_numpy(dtype=float)

# Sort by input voltage and remove duplicate input values
order = np.argsort(vin)
vin, vout = vin[order], vout[order]
vin, unique_idx = np.unique(vin, return_index=True)
vout = vout[unique_idx]

if len(vin) < 10 or not np.all(np.diff(vin) > 0):
    raise ValueError("Invalid VTC data: check inverter_vtc.txt.")

VDD = float(max(vin.max(), vout.max()))

# f(x): original inverter VTC
def f(x):
    return np.interp(x, vin, vout)

# g(x): reflected VTC, represented as y = f^(-1)(x)
# vout is decreasing, so reverse it for interpolation.
vout_ascending = vout[::-1]
vin_for_inverse = vin[::-1]

def g(x):
    return np.interp(x, vout_ascending, vin_for_inverse)

# ---------------------------------------------------------
# 2. Find the approximate intersection of the two curves
# ---------------------------------------------------------
x_curve = np.linspace(0, VDD, 5001)
difference = f(x_curve) - g(x_curve)
cross_idx = np.argmin(np.abs(difference))
xc = x_curve[cross_idx]

print(f"Approximate butterfly intersection: {xc:.4f} V")

# ---------------------------------------------------------
# 3. Search for the largest axis-aligned square in each lobe
# ---------------------------------------------------------
# A square is tested against the monotonic curve boundaries.
# Left lobe:  f(x0 + side) - g(x0) >= side
# Right lobe: g(x0 + side) - f(x0) >= side
#
# The search uses a finite grid, so the result is an estimate.

def search_lobe(lobe, n_x=1801, n_side=1801):
    if lobe == "left":
        x_starts = np.linspace(0, xc, n_x)
    else:
        x_starts = np.linspace(xc, VDD, n_x)

    best = None
    best_side = 0.0

    for x0 in x_starts:
        max_side = VDD - x0
        if max_side <= best_side:
            continue

        sides = np.linspace(0, max_side, n_side)
        x_end = x0 + sides

        if lobe == "left":
            clearance = f(x_end) - g(x0)
        else:
            clearance = g(x_end) - f(x0)

        feasible = sides <= clearance + 1e-12
        feasible[0] = False

        if np.any(feasible):
            side = sides[np.where(feasible)[0][-1]]

            if side > best_side:
                if lobe == "left":
                    y0 = g(x0)
                else:
                    y0 = f(x0)

                best_side = float(side)
                best = (float(x0), float(y0), best_side)

    return best

left_square = search_lobe("left")
right_square = search_lobe("right")

candidates = [
    s for s in (left_square, right_square)
    if s is not None
]

if not candidates:
    raise RuntimeError(
        "No square found. Check the VTC data and curve construction."
    )

x0, y0, snm = max(candidates, key=lambda square: square[2])
selected_lobe = (
    "left" if left_square is not None
    and np.isclose(snm, left_square[2]) else "right"
)

print(f"Estimated SNM: {snm * 1000:.2f} mV")
print(f"Selected lobe: {selected_lobe}")
print(f"Square lower-left corner: ({x0:.4f}, {y0:.4f}) V")
print(f"Square side length: {snm:.4f} V")

# ---------------------------------------------------------
# 4. Plot the butterfly and the selected square
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(vin, vout, linewidth=2, label="Inverter VTC")
ax.plot(vout, vin, linewidth=2, label="Reflected VTC")
ax.plot(
    [0, VDD], [0, VDD], "--",
    linewidth=1.5, label="Vout = Vin"
)

# Draw the square entirely in the selected lobe
square_x = [x0, x0 + snm, x0 + snm, x0, x0]
square_y = [y0, y0, y0 + snm, y0 + snm, y0]
ax.plot(
    square_x, square_y, color="red",
    linewidth=2.5, label=f"Estimated SNM: {snm*1000:.1f} mV"
)

ax.set_xlim(0, VDD)
ax.set_ylim(0, VDD)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Voltage (V)")
ax.set_ylabel("Voltage (V)")
ax.set_title("SRAM Butterfly Curve and Estimated SNM")
ax.grid(True, alpha=0.6)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig("sram_butterfly_snm.png", dpi=300)
plt.show()
