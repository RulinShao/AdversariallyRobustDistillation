# Adversarially Robust Distillation (ARD): A PyTorch implementation

This repository contains Pytorch code for the ARD method from [paper](Insert ArXiv link here) "Adversarially Robust Distillation" by Micah Goldblum, Liam Fowl, Soheil Feizi, and Tom Goldstein.

Adversarially Robust Distillation is a method for transferring robustness from a robust teacher network to the student network during distillation.  In our experiments, small ARD student models outperform adversarially trained models with identical architecture.

## Prerequisites
* Python3
* Pytorch
* CUDA

## Run
Here is an example of how to run our program:
```
$ python main.py --teacher_path INSERT-YOUR-TEACHER-PATH
```
## Want to attack ARD?
A MobileNetV2 ARD model distilled from a [TRADES](https://arxiv.org/pdf/1901.08573.pdf) WideResNet (34-10) teacher can be found at [link](INSERT HERE).