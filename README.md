# Sampling-free Variational Inference for Neural Networks with Multiplicative Activation Noise

This code release accompanies the paper

**Sampling-free Variational Inference for Neural Networks with Multiplicative Activation Noise** \
[Jannik Schmitt](https://www.visinf.tu-darmstadt.de/visinf/team_members/jschmitt/jschmitt.en.jsp), [Stefan Roth](https://www.visinf.tu-darmstadt.de/visinf/team_members/sroth/sroth.en.jsp).
In GCPR 2021.

Contact: jannik.schmitt[at]visinf.tu-darmstadt.de

## Requirements

This code was tested on Python 3.6.9 with PyTorch 1.4.0 and CUDA 10.2.

All requirements are summarized in ``requirements.txt``.
An easy way to install the required packages is setting up an Anaconda environment.

```
conda config --add channels pytorch
conda create -n <environment_name> --file requirements.txt
conda activate <environment_name>
```

## Data Preprocessing

### Regression on UCI Datasets

For the [UCI](https://archive.ics.uci.edu/ml/datasets.php) regression experiments, we store the data in comma-separated csv-files with a header in the first row.

### Image Classification on [ImageNet](http://image-net.org/download)

We rescale all images so that the smaller image dimension equals 256px.


## Training and Evaluation

The commands required to train and evaluate the models in the paper can be found in the ``commands.md``.


## Results

Our models achieve the following performance:

### LLH for Regression on the UCI Datasets

|      | boston | concrete | energy | kin8 | power |  wine  | yacht |
| ---- | ------ | -------- | ------ | ---- | ----- | ------ | ----- |
| SMFVI|  -3.51 |   -3.42  |  -1.11 | 1.17 | -2.88 |  -2.01 | -0.37 |
| MNVI |  -2.43 |   -3.05  |  -1.33 | 1.15 | -2.86 |  -0.96 | -0.37 |

### Image Classification on MNIST

|      Model      | Misclass. |  NLLH  |   ECE   |  AUMRC  |
| --------------- | --------- | ------ | ------- | ------- |
|    LeNet MFVI   |   0.57%   |  0.017 |  0.0021 | 8.30e-5 |
|    LeNet SMFVI  |   0.60%   |  0.017 |  0.0020 | 8.27e-5 |
|    LeNet MNVI   |   0.55%   |  0.018 |  0.0019 | 8.33e-5 |

### Image Classification on CIFAR-10

|      Model      | Misclass. |  NLLH  |   ECE   |  AUMRC  |
| --------------- | --------- | ------ | ------- | ------- |
| AllConvNet MFVI |   7.72%   |  0.348 |  0.0495 | 0.00898 |
| AllConvNet SMFVI|   8.39%   |  0.482 |  0.0586 | 0.01034 |
| AllConvNet MNVI |   7.62%   |  0.352 |  0.0492 | 0.00895 |
|  ResNet18 MFVI  |   5.63%   |  0.256 |  0.0372 | 0.00564 |
|  ResNet18 SMFVI |   5.84%   |  0.233 |  0.0304 | 0.00750 |
|  ResNet18 MNVI  |   5.60%   |  0.246 |  0.0346 | 0.00553 |

### Image Classification on CIFAR-100

|      Model      | Misclass. |  NLLH  |   ECE   |  AUMRC  |
| --------------- | --------- | ------ | ------- | ------- |
|  ResNet18 MFVI  |   26.91%  |  1.271 |  0.131  |  0.0787 |
|  ResNet18 MFVI  |   27.18%  |  1.297 |  0.136  |  0.0803 |
|  ResNet18 MNVI  |   25.30%  |  1.085 |  0.105  |  0.0740 |

### Image Classification on ImageNet

|      Model      | Misclass. |  NLLH  |   ECE   |  AUMRC  |
| --------------- | --------- | ------ | ------- | ------- |
|  ResNet18 MNVI  |   31.05%  |  1.276 |  0.0388 |  0.1092 |

## Acknowledgements

This code is based upon [Jochen Gast's](https://scholar.google.com/citations?user=tmRcFacAAAAJ&hl=en) Lightweight Probabilistic Deep Networks [implementation.](https://github.com/ezjong/lightprobnets)

## Citations

If you use our code, please cite our GCPR 2021 paper:

    @inproceedings{Schmitt:2021:SFV,
        title = {Sampling-free Variational Inference for Neural Networks with Multiplicative Activation Noise},
        author = {Jannik Schmitt and Stefan Roth},
        booktitle = {Pattern Recognition, 43rd DAGM German Conference, DAGM GCPR 2021},
        year = {2021}}