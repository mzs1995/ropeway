# Ropeway
This repository contains code to reproduce results from the paper:

Enhancing transferability of adversarial examples with gradient shift

Authors...

## Environment

- Python 3.9.7
- [PyTorch 1.10.0](https://pytorch.org/get-started/previous-versions/)
- Scipy 1.6.2
- [Pretrainedmodels 0.7.4](https://github.com/Cadene/pretrained-models.pytorch)

## Introduction

- `attack`: the implementation for various attacks.
- `storage`: the folder for datasets, labels and adversarial examples.
- `utils`: the functions for generating adversarial examples and verification.
- `generate_adv_example.py`: generating adversarial examples on the source model (Inception-v3 as default).
- `verify_on_normal_model.py`: verifying on 6 target models.

## Datasets and Models

### Datasets
You should download the dataset at [here](https://drive.google.com/open?id=1CfobY6i8BfqfWPHL31FKFDipNjqWwAhS), then put all images to `storage/imgs`. The datasets and labels are provided by [SI-NI-FGSM](https://github.com/JHL-HUST/SI-NI-FGSM). We have already provided the all labels.

### Models
The pretrained models are provided by ["pretrainedmodels"](https://github.com/Cadene/pretrained-models.pytorch) which you should install in environment.

## Generate adversarial examples
Generate adversarial examples:
```
python generate_adv_example.py
```
- Before generating adversarial examples, you should empty `/storage/adv`.
- The default attack is R-DIM. You can change attack with `python generate_adv_example.py --attack=XXX` (All attacks are `DIM`,`R-DIM`,`SIM`,`R-SIM`,`TIM`,`R-TIM`,`DI-Admix`,`R-DI-Admix`).
- The default source model is Inception-v3.
- The adversarial examples will be stored in `/storage/adv`.

## Verify
Verify the results:
```
python verify_on_normal_model.py
```
- Before running, you should make sure that `/storage/adv` is not empty and label file in `/storage/label` is ready.
- There are 6 normally trained models as default.

## Other tips

### Verify on adversarially trained models
- This part is implemented with TensorFlow. We recommend referring to the source code of [FIA](https://github.com/hcguoO0/FIA), which provides detailed information, including version and lib information.
- The adversarially trained models can be found [here](https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models).

### About the `/storage/label`
- This folder is uesed to store the labels of benign images on different models.
- Ideally, the 1000 images tested are correctly classified by all models. But in our experiments, this is not the case. For example, there are 939 images can be classified correctly by ResNet-152 and 989 images for Inception-v3. So, we provide images with names which can be classified correctly and their ground truth labels.

## Acknowledgements
The following resources are very helpful for our work:
- [FGSM](https://github.com/cleverhans-lab/cleverhans)
- [DIM](https://github.com/cihangxie/DI-2-FGSM)
- [SIM](https://github.com/JHL-HUST/SI-NI-FGSM)
- [TIM](https://github.com/dongyp13/Translation-Invariant-Attacks)
- [Admix](https://github.com/JHL-HUST/Admix)
- [FIA](https://github.com/hcguoO0/FIA)
- [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch) and [adversarially trained models](https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models)

## Citation
Still under review...
