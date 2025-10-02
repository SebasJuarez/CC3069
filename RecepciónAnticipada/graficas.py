import csv
import math
import matplotlib.pyplot as plt

sizes, avg_us, min_us, max_us, mbps = [], [], [], [], []
with open("recv_anticipada_summary.csv", newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        sizes.append(int(row["size_bytes"]))
        avg_us.append(float(row["avg_us"]))
        min_us.append(float(row["min_us"]))
        max_us.append(float(row["max_us"]))
        mbps.append(float(row["avg_MBps"]))

# Latencia vs tamaño
plt.figure()
plt.plot(sizes, avg_us, marker="o", label="Promedio")
plt.plot(sizes, min_us, marker="x", label="Mínimo")
plt.plot(sizes, max_us, marker="^", label="Máximo")
plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("Tamaño del mensaje (bytes)")
plt.ylabel("Latencia de recepción (μs)")
plt.title("Recepción anticipada: Latencia vs Tamaño")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("recv_anticipada_latency.png", dpi=150)
plt.show()

# Rendimiento vs tamaño
plt.figure()
plt.plot(sizes, mbps, marker="o")
plt.xscale("log", base=2)
plt.xlabel("Tamaño del mensaje (bytes)")
plt.ylabel("Rendimiento promedio (MB/s)")
plt.title("Recepción anticipada: Rendimiento vs Tamaño")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("recv_anticipada_throughput.png", dpi=150)
plt.show()
