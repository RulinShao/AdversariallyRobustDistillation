# Adversarially Robust Distillation (ARD): PyTorch implementation

This repository contains PyTorch code for the ARD method from ["Adversarially Robust Distillation"](https://arxiv.org/abs/1905.09747) by Micah Goldblum, Liam Fowl, Soheil Feizi, and Tom Goldstein.

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
A MobileNetV2 ARD model distilled from a [TRADES](https://arxiv.org/pdf/1901.08573.pdf) WideResNet (34-10) teacher on CIFAR-10 can be found [here](https://drive.google.com/drive/folders/15Od-zi6HGwQoIym3AkLGzLVPaR8oH9UR?usp=sharing).



# Note for `adv-v3-2`
## May 13

* Modified `opt.zero_grad()` to update parameters onlu according to the final loss.
* Added coeeficients for the losses.

Others are the same as `adv-v3`:

Follow the computation graph of delta in fast adversarial training:

The delta takes the previous values and is fed into the student network and then clampped to a new delta. Note the loss in this round is backward, obtaining the first gradient of the student network.

Then feed the new delta into both the student and the teacher network, calculate the new `ce_loss`, `kd_loss`, and `iga_loss` in this block.

Sum up the three loss and do the second backward, getting the added gradients of the student model and do optimization step.