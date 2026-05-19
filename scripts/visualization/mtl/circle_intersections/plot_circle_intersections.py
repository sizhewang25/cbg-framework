"""3D visualizations explaining the spherical two-circle intersection.

Renders three PNGs into `outputs/`:

1. boundary_eq.png   — why a point p lies on cap i's boundary iff p · x_i = cos(r_i)
2. two_caps.png      — two spherical caps generically cross at exactly two points
3. construction.png  — how `a, b, x0, n, ±t·n` from circle_intersections() are built

Run as a script:
    python -m scripts.visualization.mtl.circle_intersections.plot_circle_intersections
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.proj3d import proj_transform

from scripts.framework.geometry import geo_to_cartesian  # noqa: F401  (kept for parity)

OUT_DIR = Path(__file__).parent / "outputs"


# ---------------------------------------------------------------------------
# 3D arrow (matplotlib's quiver looks thin/anemic in 3D)
# ---------------------------------------------------------------------------
class Arrow3D(FancyArrowPatch):
    def __init__(self, start, end, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts = (start, end)

    def do_3d_projection(self, renderer=None):
        xs, ys, zs = zip(*self._verts)
        xs, ys, zs = proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)


def add_arrow(ax, start, end, color="black", lw=2.5, label=None,
              label_offset=1.08, fontsize=14):
    ax.add_artist(Arrow3D(start, end, mutation_scale=18, lw=lw,
                          arrowstyle="-|>", color=color))
    if label:
        t = np.asarray(end) * label_offset
        ax.text(t[0], t[1], t[2], label, color=color,
                fontsize=fontsize, fontweight="bold")


# ---------------------------------------------------------------------------
# Sphere / cap helpers
# ---------------------------------------------------------------------------
def draw_sphere(ax, alpha=0.06, n=40):
    u = np.linspace(0, 2 * np.pi, 2 * n)
    v = np.linspace(0, np.pi, n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="lightblue", alpha=alpha,
                    edgecolor="gray", linewidth=0.15)


def _orthobasis(center):
    """Two orthonormal vectors perpendicular to a unit vector `center`."""
    tmp = np.array([0.0, 0.0, 1.0]) if abs(center[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(center, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(center, e1)
    return e1, e2


def cap_boundary(center, radius, n=200):
    e1, e2 = _orthobasis(center)
    t = np.linspace(0, 2 * np.pi, n)
    return (
        np.cos(radius) * center[None, :]
        + np.sin(radius) * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2)
    )


def cap_surface(center, radius, n_r=18, n_t=80):
    e1, e2 = _orthobasis(center)
    rs = np.linspace(0, radius, n_r)
    ts = np.linspace(0, 2 * np.pi, n_t)
    R, T = np.meshgrid(rs, ts)
    cR, sR = np.cos(R), np.sin(R)
    cT, sT = np.cos(T), np.sin(T)
    X = cR * center[0] + sR * (cT * e1[0] + sT * e2[0])
    Y = cR * center[1] + sR * (cT * e1[1] + sT * e2[1])
    Z = cR * center[2] + sR * (cT * e1[2] + sT * e2[2])
    return X, Y, Z


def _style(ax, title):
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])
    ax.set_zlim([-1.05, 1.05])
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, pad=8)


# ---------------------------------------------------------------------------
# Closed-form spherical two-circle intersection (mirrors circle_intersections)
# ---------------------------------------------------------------------------
def two_circle_crossings(x1, x2, r1, r2):
    """Return (p_plus, p_minus, x0, n_vec, t) for two unit-sphere caps."""
    q = np.dot(x1, x2)
    denom = 1 - q ** 2
    a = (np.cos(r1) - np.cos(r2) * q) / denom
    b = (np.cos(r2) - np.cos(r1) * q) / denom
    x0 = a * x1 + b * x2
    n_vec = np.cross(x1, x2)
    t = np.sqrt((1 - np.dot(x0, x0)) / np.dot(n_vec, n_vec))
    return x0 + t * n_vec, x0 - t * n_vec, x0, n_vec, t


# ---------------------------------------------------------------------------
# Figure 1 — boundary equation
# ---------------------------------------------------------------------------
def plot_boundary_equation(out_path: Path):
    fig = plt.figure(figsize=(8.5, 8))
    ax = fig.add_subplot(111, projection="3d")
    draw_sphere(ax)

    x_i = np.array([0.4, 0.55, 0.75]); x_i /= np.linalg.norm(x_i)
    r_i = 0.55

    X, Y, Z = cap_surface(x_i, r_i)
    ax.plot_surface(X, Y, Z, color="orange", alpha=0.30, edgecolor="none")
    bd = cap_boundary(x_i, r_i)
    ax.plot(bd[:, 0], bd[:, 1], bd[:, 2], color="darkorange", lw=2.5)

    p = bd[160]
    add_arrow(ax, [0, 0, 0], x_i, color="crimson", label=" x_i", label_offset=1.12)
    add_arrow(ax, [0, 0, 0], p,    color="navy",   label=" p",   label_offset=1.12)

    # arc at the origin showing the angle r_i
    arc = np.array([(1 - s) * x_i + s * p for s in np.linspace(0, 1, 60)])
    arc = 0.35 * arc / np.linalg.norm(arc, axis=1)[:, None]
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="purple", lw=2.5)
    mid = arc[30] * 1.35
    ax.text(mid[0], mid[1], mid[2], "r_i", color="purple", fontsize=15, fontweight="bold")

    ax.scatter([0], [0], [0], color="black", s=20)
    ax.text(0.05, 0.05, -0.05, "O", fontsize=11)

    _style(
        ax,
        "Boundary equation:  p · x_i = cos(r_i)\n"
        "Both p and x_i are unit vectors from origin O.\n"
        "Angle between them (purple) = cap's angular radius r_i  ⇔  p lies on the boundary.",
    )
    ax.view_init(elev=12, azim=-55)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2 — two caps cross at two points
# ---------------------------------------------------------------------------
def plot_two_caps(out_path: Path):
    fig = plt.figure(figsize=(8.5, 8))
    ax = fig.add_subplot(111, projection="3d")
    draw_sphere(ax)

    x1 = np.array([0.85, 0.15, 0.55]); x1 /= np.linalg.norm(x1)
    x2 = np.array([0.25, 0.80, 0.55]); x2 /= np.linalg.norm(x2)
    r1 = r2 = 0.55

    X, Y, Z = cap_surface(x1, r1); ax.plot_surface(X, Y, Z, color="tomato", alpha=0.30, edgecolor="none")
    X, Y, Z = cap_surface(x2, r2); ax.plot_surface(X, Y, Z, color="royalblue", alpha=0.30, edgecolor="none")

    b1 = cap_boundary(x1, r1); ax.plot(b1[:, 0], b1[:, 1], b1[:, 2], color="darkred",  lw=2.5)
    b2 = cap_boundary(x2, r2); ax.plot(b2[:, 0], b2[:, 1], b2[:, 2], color="darkblue", lw=2.5)

    p_plus, p_minus, *_ = two_circle_crossings(x1, x2, r1, r2)

    add_arrow(ax, [0, 0, 0], x1, color="crimson",   label=" x1")
    add_arrow(ax, [0, 0, 0], x2, color="royalblue", label=" x2")

    for pt, name, off in [(p_plus,  "p+", [0.07, 0.07,  0.07]),
                          (p_minus, "p−", [0.07, -0.05, -0.10])]:
        ax.scatter(*pt, color="black", s=60, zorder=10)
        ax.text(pt[0] + off[0], pt[1] + off[1], pt[2] + off[2], name,
                fontsize=13, fontweight="bold")

    _style(
        ax,
        "Two spherical caps generically cross at exactly two points  p+  and  p−\n"
        "(boundary circles drawn in red and blue).",
    )
    ax.view_init(elev=18, azim=-50)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3 — the construction
# ---------------------------------------------------------------------------
def plot_construction(out_path: Path):
    fig = plt.figure(figsize=(9.5, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    draw_sphere(ax, alpha=0.04)

    x1 = np.array([0.85, 0.15, 0.55]); x1 /= np.linalg.norm(x1)
    x2 = np.array([0.25, 0.80, 0.55]); x2 /= np.linalg.norm(x2)
    r1 = r2 = 0.55

    p_plus, p_minus, x0, n_vec, _t = two_circle_crossings(x1, x2, r1, r2)

    # orthonormal basis of the (x1, x2, O) plane
    u_vec = x1
    w_vec = x2 - np.dot(x1, x2) * x1
    w_vec /= np.linalg.norm(w_vec)

    S = 1.15
    corners = np.array([
        +S * u_vec + S * w_vec,
        -S * u_vec + S * w_vec,
        -S * u_vec - S * w_vec,
        +S * u_vec - S * w_vec,
    ])
    plane = Poly3DCollection([corners], alpha=0.18, facecolor="khaki",
                              edgecolor="goldenrod", linewidth=1)
    ax.add_collection3d(plane)

    # thin boundary circles for context
    b1 = cap_boundary(x1, r1); ax.plot(b1[:, 0], b1[:, 1], b1[:, 2], color="darkred",  lw=1.5, alpha=0.7)
    b2 = cap_boundary(x2, r2); ax.plot(b2[:, 0], b2[:, 1], b2[:, 2], color="darkblue", lw=1.5, alpha=0.7)

    add_arrow(ax, [0, 0, 0], x1, color="crimson",   label=" x1")
    add_arrow(ax, [0, 0, 0], x2, color="royalblue", label=" x2")

    n_hat = n_vec / np.linalg.norm(n_vec)
    add_arrow(ax, [0, 0, 0],  1.05 * n_hat, color="darkgreen", label=" n = x1×x2")
    add_arrow(ax, [0, 0, 0], -1.05 * n_hat, color="darkgreen")

    ax.scatter(*x0, color="black", s=70, zorder=10)
    ax.text(x0[0] + 0.05, x0[1] + 0.05, x0[2] + 0.05, "x0",
            fontsize=13, fontweight="bold")

    ax.plot([x0[0], p_plus[0]],  [x0[1], p_plus[1]],  [x0[2], p_plus[2]],
            "--", color="darkgreen", lw=2)
    ax.plot([x0[0], p_minus[0]], [x0[1], p_minus[1]], [x0[2], p_minus[2]],
            "--", color="darkgreen", lw=2)

    mid_p = 0.5 * (x0 + p_plus)
    mid_m = 0.5 * (x0 + p_minus)
    ax.text(mid_p[0] + 0.04, mid_p[1] + 0.04, mid_p[2] + 0.04, "+t·n",
            color="darkgreen", fontsize=12, fontweight="bold")
    ax.text(mid_m[0] + 0.04, mid_m[1] - 0.06, mid_m[2] - 0.04, "−t·n",
            color="darkgreen", fontsize=12, fontweight="bold")

    for pt, name, off in [(p_plus,  "p+", [0.06, 0.06,  0.08]),
                          (p_minus, "p−", [0.06, -0.05, -0.10])]:
        ax.scatter(*pt, color="black", s=70, zorder=10)
        ax.text(pt[0] + off[0], pt[1] + off[1], pt[2] + off[2], name,
                fontsize=13, fontweight="bold")

    ax.scatter([0], [0], [0], color="black", s=25)
    ax.text(0.04, 0.04, -0.06, "O", fontsize=11)

    _style(
        ax,
        "Construction:  x0  lives in the yellow plane through x1, x2, O  (x0 = a·x1 + b·x2)\n"
        "n = x1×x2  is the green axis perpendicular to that plane.\n"
        "p± = x0 ± t·n  pops out of the plane to land on the unit sphere.",
    )
    ax.view_init(elev=20, azim=-55)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_boundary_equation(OUT_DIR / "boundary_eq.png")
    plot_two_caps(OUT_DIR / "two_caps.png")
    plot_construction(OUT_DIR / "construction.png")
    print("Wrote:")
    for f in sorted(OUT_DIR.iterdir()):
        print(" ", f)


if __name__ == "__main__":
    main()
