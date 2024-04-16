import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("csvs/all.csv")
ngpus = sorted(list(set(df["ngpus"])))
print(df)
df = df.groupby(["stage", "batch", "ngpus", "strategy"]).mean()
print(df)
ax = sns.pointplot(
    data=df,
    x="ngpus",
    y="ntokens_per_second",
    hue="strategy",
    markersize=5,
    native_scale=True
    )
plt.xscale('log')
ax.set(xlabel='GPUs', ylabel='Tokens /s')
ax.yaxis.labelpad = 0.5
ax.set_xticks(ngpus) # <--- set the ticks first
print(ngpus)
ax.set_xticklabels(ngpus)
ax.figure.savefig("all.png", dpi=420)