from transformers.models.lsh.modeling_lsh import LSHLinear, STR2HASH, STR2SAMPLING
from transformers.models.lsh.hashing.node import LSHLinearStrided
import torch
import time
from argparse import ArgumentParser
import numpy as np


def main():
    inp = torch.rand((16, 512, 768))
    if args.module == "torch-linear":
        layer = torch.nn.Linear(768, 3072, True)
    elif args.module == "lsh-linear":
        layer = LSHLinear(768, 3072, 30, 7, STR2HASH["simhash"], STR2SAMPLING["vanilla"], {"sampling_num_target_neurons": 128})
    elif args.module == "lsh-linear-strided":
        layer = LSHLinearStrided(768, 3072, 30, 7, STR2HASH["simhash"], STR2SAMPLING["vanilla"], {"sampling_num_target_neurons": 128})
    else:
        raise NotImplementedError

    if args.device == "cuda":
        layer = layer.cuda()
        inp = inp.cuda()

    perf_counter = []
    for _ in range(50):
        start = time.perf_counter()
        layer(inp)
        end = time.perf_counter()
        perf_counter.append(end - start)

    file_name = f"{args.module}_{args.device}_runtime.txt"
    with open(file_name, mode="a") as f:
        f.write(f"{np.mean(perf_counter).item()} {np.std(perf_counter).item()}\n")


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--module", choices=["torch-linear", "lsh-linear", "lsh-linear-strided"], type=str)
    arg_parser.add_argument("--device", choices=["cpu", "cuda"], type=str, default="cpu")

    args = arg_parser.parse_args()
    main()
