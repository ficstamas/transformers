import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

data = {
    "device": [],
    "type": [],
    "perf_counter": []
}

for file in os.listdir("runtime/"):
    if not file.endswith(".txt"):
        continue

    path = os.path.join("runtime/", file)
    args = file.split("_")
    with open(path, mode="r") as f:
        for line in f.readlines():
            val = line.rstrip('\n')
            if len(val) == 0:
                continue
            data["type"].append(args[0])
            data["device"].append(args[1])
            data["perf_counter"].append(float(val.split(" ")[0]))

df = pd.DataFrame(data=data)

sns.displot(data=df, x="perf_counter", kde=True, hue="type", col="device", bins=10)
plt.show()