import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
import numpy as np
import os
import random
import argparse

from autoattack import AutoAttack

from models import *


parser = argparse.ArgumentParser(description='Evaluate Robustness on CIFAR')
parser.add_argument('--dataset', default='CIFAR10', type=str, choices=['CIFAR10', 'CIFAR100'])
parser.add_argument('--model', default='MobileNetV2', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--mode', default=['auto'], help='terms be evaluates, choose from clean, pgd, auto')
# for PGD attack
parser.add_argument('--seed', default=310)
parser.add_argument('--batch_size', default=100)
parser.add_argument('--epsilons', default=[8/255])
parser.add_argument('--num_steps', default=20)
# fixed parameters
parser.add_argument('--data_dir', default='../dataset', help='path to dataset')
args = parser.parse_args()
print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True


print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
if args.dataset == 'CIFAR10':
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    num_classes = 10
elif args.dataset == 'CIFAR100':
    testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    num_classes = 100
else:
    raise AttributeError


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_model(model_name=args.model):
    print('==> Building model..' + model_name)
    if model_name == 'MobileNetV2':
        basic_net = MobileNetV2(num_classes=num_classes)
    elif model_name == 'WideResNet':
        basic_net = WideResNet(num_classes=num_classes)
    elif model_name == 'ResNet18':
        basic_net = ResNet18(num_classes=num_classes)
    else:
        raise AttributeError
    basic_net = basic_net.to(device)
    if args.model_path:
        basic_net.load_state_dict(torch.load(args.model_path)['net'])
    return basic_net.eval()


def count_parameters():
    model = load_model(args.model)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{args.model}: {count}")
    return count


def evaluate(model):
    print('==> Evaluating clean accuray..')
    criterion = nn.CrossEntropyLoss()
    val_accuracy, val_loss = 0.0, 0.0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = (outputs.argmax(axis=-1) == labels).float().mean()
            val_accuracy += acc / len(testloader)
            val_loss += loss / len(testloader)
    print(f"  Test Loss: {val_loss}, Clean Accuracy: {val_accuracy}")


def pgd_attack(model):
    robust_acc = []
    print('==> Testing with PGD...')
    for eps in args.epsilons:
        step_size = 2 * eps / args.num_steps
        adv_correct, total = 0, 0
        for i, data in enumerate(testloader, 0):
            inputs, targets = data[0].to(device), data[1].to(device)
            x = inputs.detach()
            x = x + torch.zeros_like(x).uniform_(-eps, eps)
            for i in range(args.num_steps):
                x.requires_grad_()
                with torch.enable_grad():
                    loss = F.cross_entropy(model(x), targets, size_average=False)
                grad = torch.autograd.grad(loss, [x])[0]
                x = x.detach() + step_size * torch.sign(grad.detach())
                x = torch.min(torch.max(x, inputs - eps), inputs + eps)
                x = torch.clamp(x, 0.0, 1.0)
            _, adv_predicted = model(x).max(1)
            adv_correct += adv_predicted.eq(targets).sum().item()
            total += targets.size(0)
        robust_acc.append(adv_correct / total)
        print(f"  Step {args.num_steps}, Linf norm ≤ {eps:<6}: {robust_acc[-1]:1.4f}")
    return robust_acc


def auto_attack(model):
    print('==> Testing with AutoAttack...')
    attack = AutoAttack
    robust_acc = []
    for eps in args.epsilons:
        adversary = attack(model, norm='Linf', eps=eps)
        adv_correct, total = 0, 0
        for i, data in enumerate(testloader, 0):
            inputs, targets = data[0].to(device), data[1].to(device)
            x_adv = adversary.run_standard_evaluation(inputs, targets)
            _, adv_predicted = model(x_adv).max(1)
            adv_correct += adv_predicted.eq(targets).sum().item()
            total += targets.size(0)
        robust_acc.append(adv_correct / total)
    for i in range(len(args.epsilons)):
        print(f"  AutoAttack, Linf norm ≤ {args.epsilons[i]:<6}: {robust_acc[i]:1.4f}")


if __name__ == "__main__":
    model = load_model()
    print(f"==> Testing {args.model} loaded from {args.model_path} on {args.dataset}, random seed: {args.seed}")
    if 'clean' in args.mode:
        evaluate(model)
    if 'pgd' in args.mode:
        pgd_attack(model)
    if 'auto' in args.mode:
        auto_attack(model)
    if 'count' in args.mode:
        count_parameters()