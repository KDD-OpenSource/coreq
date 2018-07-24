import pandas as pd
import BlockCorr
datafile = "StarLightCurves_TRAIN"
dataset = pd.read_csv(datafile, sep=",", header=None)
dataset.drop(0, axis=1, inplace=True)
for alpha in [0.1,0.5,0.9]:
    print("alpha = %.2f" % alpha)
    labels = BlockCorr.Cluster(dataset, alpha, 1, 0)
    print(labels)
