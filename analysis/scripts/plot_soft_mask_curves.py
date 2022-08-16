import matplotlib.pyplot as plt
import numpy as np
import torch

from state_space_dynamics.torch_udnn import soft_mask


def main():
    plt.style.use("paper")
    fig = plt.figure(figsize=(16, 10))
    ax = plt.gca()

    xmax = 0.5
    mask_threshold = 0.08
    error = np.linspace(0, xmax, 100)
    for global_step in [1, 10, 100, 1000, 10_000]:
        error_torch = torch.from_numpy(error)
        weights = soft_mask(global_step, mask_threshold, error_torch).numpy()
        ax.plot(error, weights, label=f'global_step={global_step}')
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("prediction error")
    ax.set_ylabel("weight")
    plt.legend()
    ax.set_title(f"Soft Masking ($\delta$={mask_threshold})")
    plt.savefig("results/soft_mask_curves.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
