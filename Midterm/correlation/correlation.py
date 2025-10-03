import math
import matplotlib.pyplot as plt

# Extracted coordinates from the graph
POINTS = [
    (-5, -5),
    (-5, 2),
    (-3, -1),
    (-1, 1),
    (1, -2),
    (3, 1),
    (5, -3),
    (7, -2),
]

def mean(vals):
    return sum(vals) / len(vals)

def sum_sq(vals, mu):
    return sum((v - mu) ** 2 for v in vals)

def sum_cov(xs, ys, mx, my):
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys))

def pearson_r(xs, ys):
    mx, my = mean(xs), mean(ys)
    return sum_cov(xs, ys, mx, my) / math.sqrt(sum_sq(xs, mx) * sum_sq(ys, my))

if __name__ == "__main__":
    xs = [p[0] for p in POINTS]
    ys = [p[1] for p in POINTS]

    r = pearson_r(xs, ys)
    print(f"Pearson r = {r:.6f}")

    # Best fit line using least squares
    m = r * (math.sqrt(sum_sq(ys, mean(ys)) / sum_sq(xs, mean(xs))))
    b = mean(ys) - m * mean(xs)
    print(f"Best-fit line: y = {m:.4f}x + {b:.4f}")

    # Scatter plot
    plt.scatter(xs, ys, color="blue", label="Data points")
    plt.plot(xs, [m * x + b for x in xs], color="red", label="Best-fit line")
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title(f"Scatter Plot with Pearson r = {r:.3f}")
    plt.legend()
    plt.grid(True)
    plt.savefig("correlation_scatter.png", dpi=200)
    plt.show()
