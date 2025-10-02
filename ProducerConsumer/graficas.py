import csv
import matplotlib.pyplot as plt

B, T, TH = [], [], []
with open("chunks_summary.csv", newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        B.append(int(row["chunk_bytes"]))
        T.append(float(row["total_seconds"]))
        TH.append(float(row["MBps"]))

# Tiempo total vs tamaño de chunk
plt.figure()
plt.plot(B, T, marker="o")
plt.xscale("log", base=2)
plt.xlabel("Tamaño de chunk (bytes)")
plt.ylabel("Tiempo total (s)")
plt.title("Pipeline por chunks: Tiempo total vs tamaño de chunk")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("chunks_time.png", dpi=150)
plt.show()

plt.figure()
plt.plot(B, TH, marker="o")
plt.xscale("log", base=2)
plt.xlabel("Tamaño de chunk (bytes)")
plt.ylabel("Throughput efectivo (MB/s)")
plt.title("Pipeline por chunks: Throughput vs tamaño de chunk")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("chunks_throughput.png", dpi=150)
plt.show()
