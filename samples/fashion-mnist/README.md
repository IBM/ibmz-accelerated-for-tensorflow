# Fashion MNIST Sample

The code sample in this directory
[loads](https://www.tensorflow.org/versions/r2.12/api_docs/python/tf/keras/datasets/fashion_mnist/load_data)
the
[Fashion MNIST data set](https://www.tensorflow.org/datasets/catalog/fashion_mnist)
and trains a model. A second script performs inference on the model with the
test data set and displays the results.

The [tensorflow README file](../../README.md) contains general information on
downloading and running the samples.

These samples will download the MNIST data set from the Internet.

## Running the Sample

Note that you will run these commands from inside the IBM Z Accelerated for
TensorFlow container.

First, train and save the model to disk with the `fashion_mnist_training.py`
script. This will download the fashion MNIST data set and create a model in the
current directory.

Training will take some time. The epoch number in the output will indicate
progress.

```bash
python fashion_mnist_training.py
```

Once the model has been trained, run the `fashion_mnist.py` script to run
inference against the model.

```bash
python fashion_mnist.py
```

The script will report a prediction for some sample images.
