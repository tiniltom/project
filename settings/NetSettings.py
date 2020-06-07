
from src.net.G2D19_P2OF_ResHB_1LSTM import *


def GetNetwork(inputImage_, batchSize_, unrolledSize_, isTraining_, trainingStep_):
	print("\n Using Network: ", Net.__module__, "\n")
	return Net(inputImage_, batchSize_, unrolledSize_, isTraining_, trainingStep_)
