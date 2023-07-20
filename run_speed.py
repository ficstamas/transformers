from subprocess import run
import time

modules = ["torch-linear", "lsh-linear", "lsh-linear-strided"]
trials = 50

k = 0
for mod in modules:
    for i in range(trials):
        print(f"[{k}/{len(modules)*trials}] Starting `{mod}` run #{i}")
        t0 = time.perf_counter()
        run(["python", "node_speed.py", "--module", mod, "--device", "cpu"])
        print(f"[{k}/{len(modules)*trials}] Ending `{mod}` run #{i}: {time.perf_counter() - t0}")
        k += 1
