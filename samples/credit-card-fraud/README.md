# Credit Card Fraud Sample

The code sample in this directory uses the
[Credit Card Fraud data set](https://github.com/IBM/TabFormer/tree/main/data/credit_card)
and trains a model. A second script performs inference on the model with the
test data set and displays the results.

The [tensorflow README file](../../README.md) contains general information on
downloading and running the samples.

These samples require first downloading the data set from the Internet and
extracting the archive.

## Running the Sample

Note that you will run these commands from inside the IBM Z Accelerated for
TensorFlow container.

First, train and save the model to disk with the `credit_card_fraud_training.py`
script. Training will take some time. The epoch number in the output will
indicate progress.

```bash
python credit_card_fraud_training.py
```

To train a GRU model instead:

```bash
python credit_card_fraud_training.py --rnn-type gru
```

Once the model has been trained, run the `credit_card_fraud.py` script to run
inference against the model.

```bash
python credit_card_fraud.py
```

The script will report the test accuracy. To run inference with the GRU model:

```bash
python credit_card_fraud.py --rnn-type gru
```
