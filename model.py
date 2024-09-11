from collections import namedtuple
from featureBasedGameState import FeatureBasedGameState
import pickle
import constants
from pathlib import Path

ModelEntry = namedtuple('ModelEntry', "nWins nSimulations avgReward")

class Model(object):
    def __init__(self):
        self.data = {}

    def updateEntry(self, fbgs, actionTaken, nWins, nSimulations, avgReward):
        # type: (FeatureBasedGameState, str, int, float, int, float) -> None
        self.data[(fbgs, actionTaken)] = ModelEntry(nWins=nWins,
                                                    nSimulations=nSimulations, avgReward=avgReward)

    def writeModelToFile(self, file):
        with open(file, 'w') as f:
            for key, value in self.data.items():
                f.write(str(key) + ": " + str(value) + "\n")
        self.saveModel(constants.OUTPUT_MODEL)

    def saveModel(self, outputModelFilePath):
        filename = outputModelFilePath
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)
    
    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key]=value

    def __contains__(self, item):
        return item in self.data


def getModel(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    model = Model()
    model.data = data
    return model

# This global Model is used to store the statistics of all the simulations
# Note that if you are using an existing model (loaded from the .pkl file) the stats will be combined
# global commonModel

# commonModel = Model()

def fetch_model(is_training=False):
    if is_training or not Path("models/model-latest.pkl").exists():
        return Model()
    else:
        return getModel("models/model-latest.pkl")