<!-- markdownlint-disable  MD033 -->

# Using the IBM Z Accelerated for TensorFlow Container Image

# Table of contents

- [Overview](#overview)
- [Downloading the IBM Z Accelerated for TensorFlow Container Image](#container)
- [Container Image Contents](#contents)
- [TensorFlow Usage](#tensorflow)
- [A Look into the Acceleration](#acceleration)
- [Security and Deployment Guidelines](#security-and-deployment-guidelines)
- [Execution on the Integrated Accelerator for AI and on CPU](#execution-paths)
- [Model Validation](#model-validation)
- [Using the Code Samples](#code-samples)
- [Frequently Asked Questions](#faq)
- [Technical Support](#contact)
- [Versioning Policy and Release Cadence](#versioning)
- [Licenses](#licenses)

# Overview <a id="overview"></a>

[TensorFlow](https://www.tensorflow.org/) is an open source machine learning
framework. It has a comprehensive set of tools that enable model development,
training, and inference. It also features a rich, robust ecosystem.

On IBM® z16™ and later (running Linux on IBM Z or IBM® z/OS® Container
Extensions (IBM zCX)), TensorFlow core Graph Execution will leverage new
inference acceleration capabilities that transparently target the IBM Integrated
Accelerator for AI through the
[IBM z Deep Neural Network](https://github.com/IBM/zDNN) (zDNN) library. The IBM
zDNN library contains a set of primitives that support Deep Neural Networks.
These primitives transparently target the IBM Integrated Accelerator for AI on
IBM z16 and later. No changes to the original model are needed to take advantage
of the new inference acceleration capabilities.

_Note. When using IBM Z Accelerated for TensorFlow on either an IBM z15® or an
IBM z14®, TensorFlow will transparently target the CPU with no changes to the
model._

# Downloading the IBM Z Accelerated for TensorFlow Container Image <a id="container"></a>

Downloading the IBM Z Accelerated for TensorFlow container image requires
credentials for the IBM Z and LinuxONE Container Registry,
[icr.io](https://icr.io).

Documentation on obtaining credentials to `icr.io` is located
[here](https://ibm.github.io/ibm-z-oss-hub/main/main.html).

---

Once credentials to `icr.io` are obtained and have been used to login to the
registry, you may pull (download) the IBM Z Accelerated for TensorFlow container
image with the following code block:

```bash
# Replace X.X.X with the desired version to pull
docker pull icr.io/ibmz/ibmz-accelerated-for-tensorflow:X.X.X
```

In the `docker pull` command illustrated above, the version specified above is
`X.X.X`. This is based on the version available in the
[IBM Z and LinuxONE Container Registry](https://ibm.github.io/ibm-z-oss-hub/containers/ibmz-accelerated-for-tensorflow.html).
Release notes about a particular version can be found in this GitHub Repository
under releases
[here](https://github.com/IBM/ibmz-accelerated-for-tensorflow/releases).

---

To remove the IBM Z Accelerated for TensorFlow container image, please follow
the commands in the code block:

```bash
# Find the Image ID from the image listing
docker images

# Remove the image
docker rmi <IMAGE ID>
```

---

\*_Note. This documentation will refer to image/containerization commands in
terms of Docker. If you are utilizing Podman, please replace `docker` with
`podman` when using our example code snippets._

# Container Image Contents <a id="contents"></a>

To view a brief overview of the operating system version, software versions and
content installed in the container, as well as any release notes for each
released container image version, please visit the `releases` section of this
GitHub Repository, or you can click
[here](https://github.com/IBM/ibmz-accelerated-for-tensorflow/releases).

# TensorFlow Usage <a id="tensorflow"></a>

For documentation on how to train and run inferences on models with TensorFlow
please visit the official
[Open Source TensorFlow documentation](https://www.tensorflow.org/?hl=en).

For brief examples on how to train and run inferences on models with TensorFlow
please visit our [samples section](#code-samples).

# A Look into the Acceleration <a id="acceleration"></a>

The acceleration is enabled through Custom Ops, Kernels, and a Graph Optimizer
that get registered within TensorFlow.

- The registered Ops are custom versions of built-in TensorFlow Ops.

  - They will only support float and half data types.

- The registered Kernels will perform the computation for the Custom Ops.

  - There will be one-or-more Kernels registered for each Custom Op registered,
    depending on the data type(s) of the input(s) and/or output(s).
  - Many Kernels will call one-or-more zDNN functions to perform computation
    using the accelerator.
  - Some Kernels will run custom logic to perform non-computational procedures
    such as transposing or broadcasting.

- The registered Graph Optimizer will check a TensorFlow Graph for Ops with
  valid input(s) and/or output(s) and remap them to the Custom Ops.
  - Only Ops with valid input(s) and/or output(s) will be remapped.
  - Some Ops with valid input(s) and/or output(s) may still not be remapped if
    their overhead is likely to outweigh any cost savings.

## Tensor

Kernels will receive input(s) and/or output(s) in the form of `Tensor` objects.

- TensorFlow's internal `Tensor` objects manage the shape, data type, and a
  pointer to the data buffer
- More info can be found [here](https://www.tensorflow.org/guide/tensor)

## Graph Mode Requirement

Custom Kernels will only be used when the Graph Optimizer is utilized. This
happens whenever TensorFlow is operating within a
[tf.function](https://www.tensorflow.org/api_docs/python/tf/function).

- TensorFlow's built-in Keras module, used for creating a
  [Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model), will use a
  `tf.function` within many the Model's internal functions, include the
  [predict](https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict)
  function.
- A normal function can be used as a `tf.function` by:
  - Passing it to a `tf.function` call, or
  - Adding a `@tf.function` decorator above the function.
- More information can be found
  [here](https://www.tensorflow.org/guide/intro_to_graphs).

## Eigen Fallback

During the Graph Optimizer pass, input(s) and/or output(s) are checked to ensure
they are the correct shape and data type.

- This is done before computation and is designed to work with variable batch
  sizes.
- This can result in shapes for input(s) and/or output(s) being
  partially-defined, denoting undefined dimensions with negative numbers.
  - This means it is not always possible to determine if the input(s) and/or
    output(s) will be a valid shape for all batch sizes.

Due to this, all Custom Ops will check the shape of the input(s) and/or
output(s) before performing computation.

- If all shapes are valid, the custom logic is used.
- If any shape is invalid, the default Eigen logic is used.

## NNPA Instruction Set Requirement

Before the Graph Optimizer is registered, a call to `zdnn_is_nnpa_installed` is
made to ensure the NNPA instruction set for the accelerator is installed.

- If this call returns false, the Graph Optimizer is not registered and runtime
  should proceed the same way TensorFlow would without the acceleration
  benefits.

## Environment Variables for Logging <a id="env-variables"></a>

Certain environment variables can be set before execution to enable/disable
features or logs.

- `ZDNN_ENABLE_PRECHECK`: true

  - If set to true, zDNN will print logging information before running any
    computational operation.
  - Example: `export ZDNN_ENABLE_PRECHECK=true`.
    - Enable zDNN logging.

- `TF_CPP_MIN_LOG_LEVEL`: integer

  - If set to any number >= 0, logging at that level and higher (up until
    `TF_CPP_MAX_VLOG_LEVEL`) will be enabled.
  - If `TF_CPP_MAX_VLOG_LEVEL` is not set, only logs exactly at
    `TF_CPP_MIN_LOG_LEVEL` will be enabled.
  - Example: `export TF_CPP_MIN_LOG_LEVEL=0`.
    - Logs at level 0 will be enabled.

- `TF_CPP_MAX_VLOG_LEVEL`: integer

  - If set to any number >= 0, logging at that level and lower (down until
    `TF_CPP_MIN_LOG_LEVEL`) will be enabled.
  - Requires `TF_CPP_MIN_LOG_LEVEL` is set to a number >= 0.
  - Example: `export TF_CPP_MIN_LOG_LEVEL=0` `export TF_CPP_MAX_VLOG_LEVEL=1`.
    - Logs at levels 0 and 1 will be enabled.

- `TF_CPP_VMODULE`: 'file=level' | 'file1=level1,file2=level2',
  - Enables logging at level 'level' and lower (down until
    `TF_CPP_MIN_LOG_LEVEL`) for any file named 'file'.
    - Extensions for file name are ignored so 'file.h', 'file.cc', and
      'file.cpp' would all have logging enabled.
  - Requires `TF_CPP_MIN_LOG_LEVEL` is set to a number >= 0.
  - Example:
    `export TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE='remapper=2,cwise_ops=1'`.
    - Logs at level 0 will be enabled.
    - Logs for files named `remapper.*` will be enabled at levels 0, 1, and 2.
    - Logs for files named `cwise_ops.*` will be enabled at levels 0 and 1.

# Security and Deployment Guidelines <a id="security-and-deployment-guidelines"></a>

- For security and deployment best practices, please visit the common AI Toolkit
  documentation found
  [here](https://github.com/IBM/ai-toolkit-for-z-and-linuxone).

# Execution on the Integrated Accelerator for AI and on CPU <a id="execution-paths"></a>

## Execution Paths

IBM Z Accelerated for TensorFlow container image follows IBM's train anywhere
and deploy on IBM Z strategy.

By default, when using the IBM Z Accelerated for TensorFlow container image on
an IBM z16 and later system, TensorFlow core will transparently target the
Integrated Accelerator for AI for a number of compute-intensive operations
during inferencing with no changes to the model.

When using IBM Z Accelerated for TensorFlow on either an IBM z15 or an IBM z14,
TensorFlow will transparently target the CPU with no changes to the model

To modify the default execution path, you may change the environment variable,
`NNPA_DEVICES`, before the application calls any TensorFlow API:

- `NNPA_DEVICES`: 0 | false
  - If set to '0' or 'false', the Graph Optimizer will not be registered and
    runtime will proceed the same way TensorFlow would without the acceleration
    benefits. If NNPA_DEVICES is `unset` the IBM Integrated Accelerator for AI
    is targeted by default.
  - Example: `export NNPA_DEVICES=0`.
    - Graph Optimizer will not be registered.

## Eager Mode vs Graph Mode

The two primary methods of performing computations with TensorFlow are
[Eager Execution and Graph Execution](https://www.tensorflow.org/guide/intro_to_graphs).

- **Eager Execution** performs the computations immediately as they are received
  (operation by operation), those values are then returned as a result.

- **Graph Execution** encapsulates the computations within a graph (`tf.graph`).
  Each node in the graph is an operation (`tf.Operation`). Each edge that
  connects one node to another is a tensor (`tf.Tensor`) that represents the
  flow between operations. Graph Execution performs these computation at a later
  point within a TensorFlow Session. To instruct TensorFlow to run in Graph
  Mode, leverage `tf.function` either as a direct call or as a decorator. The
  `tf.function` is a Python callable that builds TensorFlow graphs from the
  Python function.

As mentioned in an early section, to take advantage of the the acceleration
capabilities, there is a **Graph Mode Requirement**. This means in order to
leverage the IBM Z Integrated Accelerated for AI, TensorFlow must be used
through `tf.function`.

## Training Workloads

Some of the samples provided in this documentation, or your own TensorFlow
applications, will train models. Generally, training should work with the IBM Z
Accelerated for TensorFlow container. However, our testing efforts have focused
more on inferencing. Problems may arise during training, so we highly advise that
you disable the IBM Integrated Accelerator for AI.

If you have any issues training models, you can disable the IBM Integrated
Accelerator for AI optimizations by setting the environment variable
`NNPA_DEVICES=0`:

```bash
# This will manually instruct TensorFlow to target the CPU for all operations.
# This might resolve training problems.
export NNPA_DEVICES=0

# Run the training script or workload.
# Some samples might not have such a script and instead download
# a model from the Internet.
python <train.py -args>

# When training is complete, unset the variable to enable
# inference optimizations.
unset NNPA_DEVICES

# Run the inference script or workload
python <inference.py -args>
```

If problems persist, this might indicate a problem with how the model is
constructed, or it might be a TensorFlow issue.

# Model Validation <a id="model-validation"></a>

Various models that were trained on x86 or IBM Z have demonstrated focused
optimizations that transparently target the IBM Integrated Accelerator for AI
for a number of compute-intensive operations during inferencing.

Models that we expect (based on internal research) to demonstrate the
optimization illustrated in this document can be found
[here](https://www.tensorflow.org/versions/r2.12/api_docs/python/tf/keras/applications).

_Note. Models that were trained outside of the TensorFlow ecosystem may throw
endianness issues._

# Using the Code Samples <a id="code-samples"></a>

Documentation for our code samples can be found [here](samples).

# Frequently Asked Questions <a id="faq"></a>

## Q: Where can I get the IBM Z Accelerated for TensorFlow container image?

Please visit this link
[here](https://ibm.github.io/ibm-z-oss-hub/containers/ibmz-accelerated-for-tensorflow.html).
Or read the section titled
[Downloading the IBM Z Accelerated for TensorFlow container image](#container).

## Q: Why are there multiple TensorFlow container images in the IBM Z and LinuxONE Container Registry? <!-- markdownlint-disable-line MD013 -->

You may have seen multiple TensorFlow container images in IBM Z and LinuxONE
Container Registry, namely
[ibmz/tensorflow](https://ibm.github.io/ibm-z-oss-hub/containers/tensorflow.html)
and
[ibmz/ibmz-accelerated-for-tensorflow](https://ibm.github.io/ibm-z-oss-hub/containers/ibmz-accelerated-for-tensorflow.html).

The `ibmz/tensorflow` container image does not have support for the IBM
Integrated Accelerator for AI. The `ibmz/tensorflow` container image only
transparently targets the CPU. It does not have any optimizations referenced in
this document.

The `ibmz/ibmz-accelerated-for-tensorflow` container image includes support for
TensorFlow core Graph Execution to transparently target the IBM Integrated
Accelerator for AI. The `ibmz/ibmz-accelerated-for-tensorflow` container image
also still allows it's users to transparently target the CPU. This container
image contains the optimizations referenced in this document.

## Q: Where can I run the IBM Z Accelerated for TensorFlow container image?

You may run the IBM Z Accelerated for TensorFlow container image on IBM Linux on
Z or IBM® z/OS® Container Extensions (IBM zCX).

_Note. The IBM Z Accelerated for TensorFlow container image will transparently
target the IBM Integrated Accelerator for AI on IBM z16 and later. However, if
using the IBM Z Accelerated for TensorFlow container image on either an IBM z15
or an IBM z14, TensorFlow will transparently target the CPU with no changes to
the model._

## Q: Can I install a newer or older version of TensorFlow in the container?

No. Installing newer or older version of TensorFlow than what is configured in
the container will not target the IBM Integrated Accelerator for AI.
Additionally, installing a newer or older version of TensorFlow, or modifying
the existing TensorFlow that is installed in the container image may have
unintended, unsupported, consequences. This is not advised.

## Q: How can I verify the IBM Integrated Accelerator for AI is being utilized?

As discuessed in [Environment Variables for Logging](#env-variables), there are
multiple methods to display logging information.

To simply verify that the IBM Integrated Accelerator for AI is being utilized,
you can set the environment variable `ZDNN_ENABLE_PRECHECK=true`, which will
display logging information before each call to the accelerator:

```bash
# This will manually instruct zDNN to display logging information before
# each call to the accelerator.
export ZDNN_ENABLE_PRECHECK=true

# Run the script.
python <script.py -args>

# When script is complete, unset the variable to disable zDNN logs.
unset ZDNN_ENABLE_PRECHECK
```

To see which operations in the Graph have been remapped to custom operations,
you can set the environment variable `TF_CPP_VMODULE='remapper=1'`, which will
display logging information before each remapping of an operation.

TensorFlow also requires `TF_CPP_MIN_LOG_LEVEL` to be set for any logs to be
generated:

```bash
# This will manually instruct TensorFlow to display logging information that
# occurs at a level >= 0.
export TF_CPP_MIN_LOG_LEVEL=0

# This will manually instruct TensorFlow to display logging information for all
# files named `remapper.*` at log level 1.
export TF_CPP_VMODULE='remapper=1'

# Run the script.
python <script.py -args>

# When script is complete, unset the variable to disable TensorFlow logs.
unset TF_CPP_MIN_LOG_LEVEL
unset TF_CPP_VMODULE
```

Finally, you can use TensorBoard for profiling. Custom operations have been built
to include additional profiling events. More information about TensorBoard can be
found [here](https://www.tensorflow.org/tensorboard/get_started).

# Technical Support <a id="contact"></a>

Information regarding technical support can be found
[here](https://github.com/IBM/ai-toolkit-for-z-and-linuxone).

# Versioning Policy and Release Cadence <a id="versioning"></a>

IBM Z Accelerated for TensorFlow will follow the
[semantic versioning guidelines](https://semver.org/) with a few deviations.
Overall, IBM Z Accelerated for TensorFlow follows a continuous release model
with a cadence of 1-2 minor releases per year. In general, bug fixes will be
applied to the next minor release and not back ported to prior major or minor
releases. Major version changes are not frequent and may include features
supporting new zSystems hardware as well as major feature changes in TensorFlow
that are not likely backward compatible. Please refer to
[TensorFlow guidelines](https://www.tensorflow.org/guide/versions) for backwards
compatibility across different versions of TensorFlow.

## IBM Z Accelerated for TensorFlow Versions

Each release version of IBM Z Accelerated for TensorFlow has the form
MAJOR.MINOR.PATCH. For example, IBM Z Accelerated for TensorFlow version 1.2.3
has MAJOR version 1, MINOR version 2, and PATCH version 3. Changes to each
number have the following meaning:

### MAJOR / VERSION

All releases with the same major version number will have API compatibility.
Major version numbers will remain stable. For instance, 1.X.Y may last 1 year or
more. It will potentially have backwards incompatible changes. Code and data
that worked with a previous major release will not necessarily work with the new
release.

### MINOR / FEATURE

Minor releases will typically contain new backward compatible features,
improvements, and bug fixes.

### PATCH / MAINTENANCE

Maintenance releases will occur more frequently and depend on specific patches
introduced (e.g. bug fixes) and their urgency. In general, these releases are
designed to patch bugs.

## Release cadence

Feature releases for IBM Z Accelerated for TensorFlow occur about every 6 months
in general. Hence, IBM Z Accelerated for TensorFlow 1.3.0 would generally be
released about 6 months after 1.2.0. Maintenance releases happen as needed in
between feature releases. Major releases do not happen according to a fixed
schedule.

# Licenses <a id="licenses"></a>

The International License Agreement for Non-Warranted Programs (ILAN) agreement
can be found
[here](https://www14.software.ibm.com/cgi-bin/weblap/lap.pl?li_formnum=L-PXXE-ZEV3SH).

The registered trademark Linux® is used pursuant to a sublicense from the Linux
Foundation, the exclusive licensee of Linus Torvalds, owner of the mark on a
worldwide basis.

TensorFlow, the TensorFlow logo and any related marks are trademarks of Google
Inc.

Docker and the Docker logo are trademarks or registered trademarks of Docker,
Inc. in the United States and/or other countries. Docker, Inc. and other parties
may also have trademark rights in other terms used herein.

IBM, the IBM logo, and ibm.com, IBM z16, IBM z15, IBM z14 are trademarks or
registered trademarks of International Business Machines Corp., registered in
many jurisdictions worldwide. Other product and service names might be
trademarks of IBM or other companies. The current list of IBM trademarks can be
found [here](https://www.ibm.com/legal/copyright-trademark).
