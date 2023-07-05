from mlnumpy import PROJECT_DIR
from mlnumpy.activations.relu import ReLU
from mlnumpy.callbacks.logs import MetricLogCallback
from mlnumpy.callbacks.saver import BestModelSaverCallback
from mlnumpy.datasets.mnist import MNIST
from mlnumpy.initializers.xavier import XavierInitializer
from mlnumpy.layers.convolution2d import Convolution2D
from mlnumpy.layers.dense import Dense
from mlnumpy.layers.flatten import Flatten
from mlnumpy.layers.pooling2d import MaxPooling2D
from mlnumpy.losses.sce import SoftmaxCrossEntropy
from mlnumpy.metrics.accuracy import Accuracy
from mlnumpy.metrics.fscore import F1Score
from mlnumpy.metrics.precision import Precision
from mlnumpy.metrics.recall import Recall
from mlnumpy.models.sequence import SequenceModel
from mlnumpy.optimizers.adam import Adam

data = MNIST()

model = SequenceModel(
    layers=[
        Convolution2D(
            filters=8,
            kernel=(3, 3),
            activation=ReLU(),
            kernel_initializer=XavierInitializer(),
        ),
        MaxPooling2D(
            pool_size=(2, 2),
            stride=(2, 2),
        ),
        Flatten(),
        Dense(
            size=50,
            activation=ReLU(),
            initializer=XavierInitializer(),
        ),
        Dense(
            size=10,
            initializer=XavierInitializer(),
        ),
    ],
    optimizer=Adam(),
    loss=SoftmaxCrossEntropy(),
    metrics={
        "accuracy": Accuracy(),
        "precision": Precision(),
        "recall": Recall(),
        "f1_score": F1Score(),
    },
    callbacks=[
        MetricLogCallback(),
        BestModelSaverCallback(
            path=PROJECT_DIR / "experiments/mnist_conv_net.pkl",
            metric="f1_score",
        ),
    ],
    batch_size=64,
    max_epochs=5,
)

if __name__ == "__main__":
    model.fit(data)
