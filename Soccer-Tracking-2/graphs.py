from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('results.csv')

print(data.columns.values)

# For some reason YOLO includes all these spaces in column names. Fix:
data = data.rename({'               epoch': 'Epoch', '      train/box_loss': 'Train Box Loss', '        val/box_loss': 'Val Box Loss', '     metrics/mAP_0.5': 'mAP 50'}, axis='columns')

print(data.columns.values)

fig = data.plot.line(x = 'Epoch', y = ['Train Box Loss', 'Val Box Loss'], )

plt.show()

fig = data.plot.line(x = 'Epoch', y = 'mAP 50')

plt.show()



