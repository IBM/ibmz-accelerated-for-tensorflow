# Fashion MNIST Sample

The code sample in this directory trains a model on the
[Fashion MNIST data set](https://www.tensorflow.org/datasets/catalog/fashion_mnist)
and runs inference on the trained model. It runs directly in the base
container — no `prerequisites.sh` is needed.

The [tensorflow README file](../../README.md) contains general information on
downloading and running the samples.

The Fashion MNIST data set is downloaded automatically when the training script
runs.

> If you are using rootless podman, see the
> [Running with podman](../README.md#running-with-podman) section in the
> top-level samples README before proceeding.

## Running the Sample

Run these commands from the **host** machine. Replace `X.X.X` with the
current version of the container image.

Create a workspace directory and start an interactive container shell:

```bash
mkdir -p workspace

docker run -it --rm \
    -v "$(pwd)":/sample:ro,z \
    -v "$(pwd)/workspace":/workspace:z \
    -w /workspace \
    icr.io/ibmz/ibmz-accelerated-for-tensorflow:X.X.X bash
```

From inside the container, train and save the model with the
`fashion_mnist_training.py` script. Training will take some time.

```bash
python /sample/fashion_mnist_training.py
```

This saves the trained model as `model.keras` in `/workspace`. Once training
is complete, run the `fashion_mnist.py` script to run inference against the
model.

```bash
python /sample/fashion_mnist.py
```

The script will report the test accuracy.

## Known Issues

There are no known open issues with this sample.
