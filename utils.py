from cgi import test
from unittest import TestLoader
import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from robustbench.data import load_cifar10
from autoattack import AutoAttack


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_model(model_name, model_path):
    if model_name in ['PreActResNet18', 'WideResNet']:
        model = eval(model_name)().cuda()
        model.load_state_dict(torch.load(model_path))
        return  model
    else:
        from robustbench.utils import load_model as load_teacher_model
        model = load_teacher_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10', threat_model='Linf').cuda()
        return model


class NormalizeLayer(torch.nn.Module):

    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = nn.Parameter(torch.tensor(means), requires_grad=False)
        self.sds = nn.Parameter(torch.tensor(sds), requires_grad=False)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).contiguous().cuda()
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).contiguous().cuda()
        return (input - means) / sds


def evaluate_autoattack(model, n=1000, no_pp=False, specify=None):
    x_test, y_test = load_cifar10(n_examples=n)
    x_test, y_test = x_test.cuda(), y_test.cuda()
    if not no_pp:
        model = nn.Sequential(NormalizeLayer(cifar10_mean, cifar10_std), model)
    adversary = AutoAttack(model, norm='Linf', eps=8/255)
    if specify is not None:
        adversary.attacks_to_run = specify
    x_adv = adversary.run_standard_evaluation(x_test, y_test)
    with torch.no_grad():
        output = model(x_adv)
        loss = F.cross_entropy(output, y_test)
        aa_acc = (output.max(1)[1] == y_test).sum().item() / n
    return loss, aa_acc


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_loaders(dir_, batch_size, no_pp=False):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    if no_pp:
        test_transform = transforms.Compose([
        transforms.ToTensor(),
        ])
    else:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
    num_workers = 2
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader, test_loader
        

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts):
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss/n, pgd_acc/n


def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n


if __name__ == '__main__':
    from model import *
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--data_dir', default='~/rulin/dataset', type=str)
    parser.add_argument('--no_pp', action='store_true', help='no preprocessing')
    args = parser.parse_args()

    train_loader, test_loader = get_loaders(args.data_dir, 128, args.no_pp)

    model_test = get_model(args.model_name, args.model_path)
    model_test.float()
    model_test.eval()

    test_loss = test_acc = pgd_loss = pgd_acc = aa_loss = aa_acc = - 1.0
    aa_loss, aa_acc = evaluate_autoattack(model_test, n=100, no_pp=args.no_pp, specify=['apgd-ce', 'fab'])
    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)
    
    print('Test Loss \t Test Acc \t PGD Loss \t PGD Acc \t AA Loss \t AA Acc')
    print('%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f '%(test_loss, test_acc, pgd_loss, pgd_acc, aa_loss, aa_acc))