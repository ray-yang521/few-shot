import os
import pprint

import pandas as pd
from matplotlib import pyplot as plt, ticker
from matplotlib.ticker import MultipleLocator

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def generate_start_end_xlsx(save_dir, df: pd.DataFrame, start, end):
    ds = df[start:end]
    save_path = os.path.join(save_dir, f"{start}-{end}.xlsx")
    ds.to_excel(save_path, index=False)


path = ""
df = pd.read_excel(path)[0:612]
df = df.dropna()

save_dir = ""

df = df.sort_values(by="runoff", ascending=False)
y = df["runoff"]

x = [each / len(y) for each in range(len(y))]
result = list(zip(x, y))

for i in range(len(result)):
    pprint.pprint((i, result[i]))

fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
xmajorLocator = MultipleLocator(0.1)
ax.xaxis.set_major_locator(xmajorLocator)
ax.set(ylabel='Runoff (m³/s)', xlabel='Cumulative percentage')
plt.axvline(x=0.05, ls=":", c="green")
plt.axvline(x=0.2, ls=":", c="green")
plt.axvline(x=0.5, ls=":", c="green")
plt.axvline(x=0.8, ls=":", c="green")
plt.axvline(x=0.95, ls=":", c="green")
ax.plot(x, y)
fig.show()

scaler = 6826.13
classifier_list = [(each[1] / scaler) for each in result]
classifier = []
for i in range(len(classifier_list)):
    print(i, classifier_list[i])

classifier.append(classifier_list[31])
classifier.append(classifier_list[123])
classifier.append(classifier_list[306])
classifier.append(classifier_list[490])
classifier.append(classifier_list[582])

classifier.reverse()
print(classifier)
