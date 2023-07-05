from mlnumpy import PROJECT_DIR
from mlnumpy.datasets.mnist import MNIST
from mlnumpy.metrics.fscore import F1Score
from mlnumpy.models.sequence import SequenceModel

data = MNIST()

model = SequenceModel.load(PROJECT_DIR / "experiments/mnist_conv_net.pkl")

if __name__ == "__main__":
    model.eval()
    features, targets = data.test_data()
    predictions = model(features)

    print(F1Score(num_classes=data.num_classes())(targets, predictions))
