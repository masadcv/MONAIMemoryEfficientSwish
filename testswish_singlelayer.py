import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from monai.networks.blocks.activation import MemoryEfficientSwish, Swish

device = "cuda" if torch.cuda.is_available() else "cpu"

shapes_to_test = [128 * (x + 1) for x in range(8)]


def SwishTest(num_runs=1, mem_eff_swish=False):
    shapes_tested = []
    memory_used_in_mb = []

    if mem_eff_swish:
        layer = MemoryEfficientSwish()
    else:
        layer = Swish()

    for image_size in shapes_to_test:
        mem = 0.0
        for r in range(num_runs):
            input_image = torch.rand((1, 3, image_size, image_size), requires_grad=True).to(device)
            pred = layer(input_image)

            mem += torch.cuda.memory_allocated() / 1024 ** 2

            pred.sum().backward()

        memory_used_in_mb.append(mem / float(num_runs))
        shapes_tested.append(image_size)

    return shapes_tested, memory_used_in_mb


if __name__ == "__main__":
    sizes_tested_eff, memory_used_eff = SwishTest(mem_eff_swish=True)
    sizes_tested_neff, memory_used_neff = SwishTest(mem_eff_swish=False)

    plt.plot(sizes_tested_eff, memory_used_eff, "g-o", label="Memory Efficient Swish")
    plt.plot(sizes_tested_neff, memory_used_neff, "r-o", label="Swish")
    plt.xticks(ticks=sizes_tested_eff, rotation=45)
    plt.xlabel("Layer size")
    plt.ylabel("Allocated memory (MB)")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig("figures/figure_singlelayer.png")
