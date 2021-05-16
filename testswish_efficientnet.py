import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from monai.networks.nets import EfficientNetBN, get_efficientnet_image_size

device = "cuda" if torch.cuda.is_available() else "cpu"

models_to_test = ["efficientnet-b%d" % b for b in range(8)]


def EfficientNetSwishTest(num_runs=1, mem_eff_swish=False):
    models_tested = []
    memory_used_in_mb = []

    loss = torch.nn.CrossEntropyLoss()

    for model_name in models_to_test:

        image_size = get_efficientnet_image_size(model_name)
        net = EfficientNetBN(model_name=model_name, pretrained=False).to(device)
        net.set_swish(mem_eff_swish)
        optim = torch.optim.SGD(net.parameters(), lr=0.1)
        mem = 0.0
        for r in range(num_runs):
            input_image = torch.rand((1, 3, image_size, image_size)).to(device)
            gt = torch.randint(0, 1000, [1]).to(device)
            pred = net(input_image)

            mem += torch.cuda.memory_allocated() / 1024 ** 2

            l = loss(pred, gt)
            l.backward()
            optim.step()
            optim.zero_grad()

        memory_used_in_mb.append(mem / float(num_runs))
        model_name = "E" + model_name[1:]
        model_name = model_name[:-2] + "B" + model_name[-1]
        models_tested.append(model_name)

    return models_tested, memory_used_in_mb


if __name__ == "__main__":
    models_tested_eff, memory_used_eff = EfficientNetSwishTest(mem_eff_swish=True)
    models_tested_neff, memory_used_neff = EfficientNetSwishTest(mem_eff_swish=False)

    plt.plot(models_tested_eff, memory_used_eff, "g-o", label="Memory Efficient Swish")
    plt.plot(models_tested_neff, memory_used_neff, "r-o", label="Swish")
    plt.xticks(rotation=45)
    plt.xlabel("Models")
    plt.ylabel("Allocated memory (MB)")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig("figures/figure_efficientnets.png")
