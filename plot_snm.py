
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the LTspice-exported inverter transfer curve
df = pd.read_csv("inverter_vtc.txt", sep=r"\s+")

vin = df["Vin"].to_numpy()
vout = df["V(vout)"].to_numpy()

# Reflect the VTC across the line Vout = Vin
# Original: (Vin, Vout)
# Reflected: (Vout, Vin)
plt.figure(figsize=(7, 7))
plt.plot(vin, vout, label="Inverter VTC")
plt.plot(vout, vin, label="Reflected VTC")
plt.plot([0, 1.8], [0, 1.8], "--", label="Vout = Vin")

plt.xlabel("Voltage (V)")
plt.ylabel("Voltage (V)")
plt.title("6T SRAM Hold-SNM: Butterfly Construction")
plt.xlim(0, 1.8)
plt.ylim(0, 1.8)
plt.grid(True)
plt.legend()
plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.savefig("butterfly_plot.png", dpi=200)
plt.show()
