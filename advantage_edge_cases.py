# plot_eq11_vs_spo_vs_ppo.py
import numpy as np
import matplotlib.pyplot as plt


def eq11_linear_penalty(r, A, eps):
    """
    Schematic Eq.(11)-style surrogate:
        L(r) = r*A - (|A|/(2*eps))*(r-1)
    (Linear penalty in (r-1).)
    """
    return r * A - (abs(A) / (2.0 * eps)) * (r - 1.0)


def spo_quadratic_penalty(r, A, eps):
    """
    SPO surrogate (as in arXiv:2401.16025, Eq. ~16 style):
        L(r) = r*A - (|A|/(2*eps))*(r-1)^2
    (Quadratic penalty in (r-1).)
    """
    return r * A - (abs(A) / (2.0 * eps)) * (r - 1.0) ** 2


def ppo_clipped_surrogate(r, A, eps):
    """
    PPO clipped surrogate:
        L(r) = min(r*A, clip(r,1-eps,1+eps)*A)
    Implemented sign-correctly via the 'min' as written.
    """
    r_clip = np.clip(r, 1.0 - eps, 1.0 + eps)
    return np.minimum(r * A, r_clip * A)


def plot_one(A, eps=0.2, r_min=0.0, r_max=2.0, num=2000, save=None):
    r = np.linspace(r_min, r_max, num)

    y_eq11 = eq11_linear_penalty(r, A, eps)
    y_spo  = spo_quadratic_penalty(r, A, eps)
    y_ppo  = ppo_clipped_surrogate(r, A, eps)

    plt.figure()
    plt.plot(r, y_eq11, label="Eq(11): linear penalty")
    plt.plot(r, y_spo,  label="SPO: quadratic penalty", linestyle="-.")
    plt.plot(r, y_ppo,  label="PPO: clipped surrogate", linestyle="--")

    # Mark key ratios
    plt.axvline(1.0, color="gray", linewidth=1, label="r=1")
    plt.axvline(1.0 - eps, color="k", linewidth=2, linestyle=":", label="r=1-ε")
    plt.axvline(1.0 + eps, color="k", linewidth=2, linestyle=":", label="r=1+ε")
    plt.axvspan(1.0 - eps, 1.0 + eps, color="0.25", alpha=0.2)

    plt.title(f"Surrogate vs ratio r for A={A:+.1f}, ε={eps}")
    plt.xlabel(r"ratio $r=\pi_\theta(a|s)/\pi_{\mathrm{ref}}(a|s)$")
    plt.ylabel("surrogate value (up to constant scaling)")
    plt.grid(False)
    plt.legend(loc="lower left", fontsize=9)

    if save is not None:
        plt.savefig(save, bbox_inches="tight", dpi=300)


def main():
    eps = 0.2
    # Two figures: one for positive advantage, one for negative advantage
    plot_one(A=+1.0, eps=eps, save="edgecase_A_pos.png")
    plot_one(A=-1.0, eps=eps, save="edgecase_A_neg.png")
    # plt.show()


if __name__ == "__main__":
    main()