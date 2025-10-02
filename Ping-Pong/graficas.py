import csv
import matplotlib.pyplot as plt

xs, ys_us = [], []
with open("pingpong_times.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        xs.append(int(row["iteration"]))
        ys_us.append(float(row["roundtrip_seconds"]))

plt.figure()
plt.plot(xs, ys_us, marker="o")
plt.title("Ping-Pong RTT por iteración")
plt.xlabel("Iteración")
plt.ylabel("Round-trip (segundos)")
plt.grid(True)
plt.tight_layout()
plt.savefig("pingpong_rtt.png", dpi=150)
plt.show()
