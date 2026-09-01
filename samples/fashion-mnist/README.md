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

Start an interactive container shell with a named volume for the workspace:

```bash
docker run -it --rm \
    -v "$(pwd)":/scripts:ro,z \
    -v fashion-mnist-workspace:/workspace \
    -w /workspace \
    icr.io/ibmz/ibmz-accelerated-for-tensorflow:X.X.X bash
```

From inside the container, train and save the model with the
`fashion_mnist_training.py` script. Training will take some time.

```bash
python /scripts/fashion_mnist_training.py
```

You can specify the number of epochs with `--epochs` (default: `10`) and
the batch size with `--batch-size` (default: `64`):

```bash
python /scripts/fashion_mnist_training.py --epochs 5 --batch-size 32
```

This saves the trained model as `model.keras` in `/workspace`. Once training
is complete, run the `fashion_mnist.py` script to run inference against the
model.

```bash
python /scripts/fashion_mnist.py
```

The script will report the test accuracy.

## Cleanup

When you are finished with the sample, remove the stopped container and the
workspace volume:

```bash
docker container prune -f
docker volume rm fashion-mnist-workspace
```

If you are using rootless Podman, verify no processes are left behind:

```bash
top -u $(whoami)
```

## Known Issues

There are no known open issues with this sample.
