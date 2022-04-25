from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import grad
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_schedule', type=int, nargs='+', default=[100, 150], help='Decrease learning rate at these epochs.')
parser.add_argument('--lr_factor', default=0.1, type=float, help='factor by which to decrease lr')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs for training')
parser.add_argument('--model', default = 'MobileNetV2', type = str, help = 'student model name')
parser.add_argument('--teacher_model', default = 'WideResNet', type = str, help = 'teacher network model')
parser.add_argument('--teacher_path', default = '../checkpoint/trades/model_cifar_wrn.pt', type=str, help='path of teacher net being distilled')
parser.add_argument('--temp', default=1.0, type=float, help='temperature for distillation')
parser.add_argument('--val_period', default=1, type=int, help='print every __ epoch')
parser.add_argument('--save_period', default=1, type=int, help='save every __ epoch')
parser.add_argument('--alpha', default=0.5, type=float, help='weight for sum of losses')
parser.add_argument('--gamma', default=0.5, type=float, help='use gamma/bs for iga')
parser.add_argument('--dataset', default = 'CIFAR10', type=str, help='name of dataset')
parser.add_argument('--distill_method', default='kdiga', choices=['ard','kd','kdiga','kdiga_rs','kdiga_ard','normal'])
parser.add_argument('--output', default = '1017', type=str, help='output subdirectory')
parser.add_argument('--exp_note', default='kdiga_alpha0.5_gamma0.5')

# PGD attack
parser.add_argument('--epsilon', default=8/255)
parser.add_argument('--num_steps', default=10)
parser.add_argument('--step_size', default=16/255/10)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def adjust_learning_rate(optimizer, epoch, lr):
    if epoch in args.lr_schedule:
        lr *= args.lr_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
if args.dataset == 'CIFAR10':
    trainset = torchvision.datasets.CIFAR10(root='../dataset', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='../dataset', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    num_classes = 10
elif args.dataset == 'CIFAR100':
    trainset = torchvision.datasets.CIFAR100(root='../dataset', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='../dataset', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    num_classes = 100


class AttackPGD(nn.Module):
    def __init__(self, basic_net, config):
        super(AttackPGD, self).__init__()
        self.basic_net = basic_net
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']

    def forward(self, inputs, targets):
        x = inputs.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                loss = F.cross_entropy(self.basic_net(x), targets, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0.0, 1.0)
        return self.basic_net(x), x



print('==> Building model..'+args.model)
if args.model == 'MobileNetV2':
    basic_net = MobileNetV2(num_classes=num_classes)
elif args.model == 'WideResNet':
    basic_net = WideResNet(num_classes=num_classes)
elif args.model == 'ResNet18':
	basic_net = ResNet18(num_classes=num_classes)
basic_net = basic_net.to(device)
assert args.teacher_path != ''
if args.teacher_model == 'MobileNetV2':
    teacher_net = MobileNetV2(num_classes=num_classes)
elif args.teacher_model == 'WideResNet':
    teacher_net = WideResNet(num_classes=num_classes)
elif args.teacher_model == 'ResNet18':
    teacher_net = ResNet18(num_classes=num_classes)
else:
    raise AttributeError
teacher_net = teacher_net.to(device)
for param in teacher_net.parameters():
    param.requires_grad = False

config = {
    'epsilon': args.epsilon,
    'num_steps': args.num_steps,
    'step_size': args.step_size,
}
net = AttackPGD(basic_net, config)
if device == 'cuda':
    cudnn.benchmark = True

print('==> Loading teacher..')
teacher_net.load_state_dict(torch.load(args.teacher_path))
teacher_net.eval()

KL_loss = nn.KLDivLoss()
XENT_loss = nn.CrossEntropyLoss()
lr=args.lr


def train(epoch, optimizer,exp_id=''):
    train_loss = 0
    iterator = tqdm(trainloader, ncols=0, leave=False)
    if args.distill_method == 'ard':
        net.train()
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, pert_inputs = net(inputs, targets)
            teacher_outputs = teacher_net(inputs)
            basic_outputs = basic_net(inputs)
            # loss = args.alpha*args.temp*args.temp*KL_loss(F.log_softmax(outputs/args.temp, dim=1),F.softmax(teacher_outputs/args.temp, dim=1))+(1.0-args.alpha)*XENT_loss(basic_outputs, targets)
            loss = args.alpha * args.temp * args.temp * KL_loss(F.log_softmax(outputs / args.temp, dim=1),
                                                                F.softmax(teacher_net(pert_inputs) / args.temp, dim=1)) + (
                               1.0 - args.alpha) * XENT_loss(basic_outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            iterator.set_description(str(loss.item()))
    elif args.distill_method == 'kd':
        basic_net.train()
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            teacher_outputs = teacher_net(inputs)
            basic_outputs = basic_net(inputs)
            loss = args.alpha * args.temp * args.temp * KL_loss(F.log_softmax(basic_outputs / args.temp, dim=1),
                                                                F.softmax(teacher_outputs / args.temp, dim=1)) + (
                               1.0 - args.alpha) * XENT_loss(basic_outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            iterator.set_description(str(loss.item()))
    elif args.distill_method == 'kdiga' or args.distill_method == 'kdiga_rs':
        basic_net.train()
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            if 'rs' in args.distill_method:
                inputs = inputs + torch.zeros_like(inputs).uniform_(-8.0/255, 8.0/255)
            optimizer.zero_grad()

            basic_outputs = basic_net(inputs)
            teacher_outputs = teacher_net(inputs)

            x = inputs.detach()
            x.requires_grad_()
            with torch.enable_grad():
                t_loss = XENT_loss(teacher_net(x), targets)
            t_grad = torch.autograd.grad(t_loss, x)[0]
            x.grad = None
            with torch.enable_grad():
                s_loss = XENT_loss(basic_net(x), targets)
            s_grad = torch.autograd.grad(s_loss, x, create_graph=True)[0]

            gama = args.gamma / inputs.shape[0]
            loss = args.alpha * args.temp * args.temp * KL_loss(F.log_softmax(basic_outputs / args.temp, dim=1),
                                                                F.softmax(teacher_outputs / args.temp, dim=1)) + (
                    1.0 - args.alpha) * XENT_loss(basic_outputs, targets) + gama * (s_grad - t_grad).norm(2)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            iterator.set_description(str(loss.item()))
    elif args.distill_method == 'kdiga_ard':
        net.train()
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs, pert_inputs = net(inputs, targets)

            x = inputs.detach()
            x.requires_grad_()
            with torch.enable_grad():
                t_loss = XENT_loss(teacher_net(x), targets)
            t_grad = torch.autograd.grad(t_loss, x)[0]
            x.grad = None
            with torch.enable_grad():
                s_loss = XENT_loss(basic_net(x), targets)
            s_grad = torch.autograd.grad(s_loss, x, create_graph=True)[0]

            gama = args.gamma/inputs.shape[0]

            loss = (args.alpha * args.temp * args.temp) * KL_loss(F.log_softmax(outputs / args.temp, dim=1),
                                                                F.softmax(teacher_net(pert_inputs) / args.temp, dim=1)) + (
                           1.0 - args.alpha) * XENT_loss(basic_net(inputs), targets) + gama * (s_grad - t_grad.detach()).norm(2)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            iterator.set_description(str(loss.item()))
    else:
        basic_net.train()
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            basic_outputs = basic_net(inputs)
            loss = XENT_loss(basic_outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            iterator.setm_description(str(loss.item()))
        # raise AttributeError
    if (epoch+1)%args.save_period == 0:
        state = {
            'net': basic_net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir('checkpoint/'+args.dataset+'/'+exp_id+'/'):
            os.makedirs('checkpoint/'+args.dataset+'/'+exp_id+'/', )
        torch.save(state, './checkpoint/'+args.dataset+'/'+exp_id+'/epoch='+str(epoch)+'.t7')
    print('Mean Training Loss:', train_loss/len(iterator))
    return train_loss


def test():
    net.eval()
    adv_correct = 0
    natural_correct = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(testloader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            adv_outputs, pert_inputs = net(inputs, targets)
            natural_outputs = basic_net(inputs)
            _, adv_predicted = adv_outputs.max(1)
            _, natural_predicted = natural_outputs.max(1)
            natural_correct += natural_predicted.eq(targets).sum().item()
            total += targets.size(0)
            adv_correct += adv_predicted.eq(targets).sum().item()
            iterator.set_description(str(adv_predicted.eq(targets).sum().item()/targets.size(0)))
    robust_acc = 100.*adv_correct/total
    natural_acc = 100.*natural_correct/total
    print('Natural acc:', natural_acc)
    print('Robust acc:', robust_acc)
    return natural_acc, robust_acc


def main(test_student_path=None):
    if test_student_path is not None:
        basic_net.load_state_dict(torch.load(test_student_path)['net'])
        basic_net.eval()
        natural_val, robust_val = test()
        print(test_student_path)
        print(f"natural acc = {natural_val:.4f}, robust acc = {robust_val:.4f}")
        return

    prefix = f"{args.output}/{args.distill_method}_{args.teacher_model}_{args.model}_{args.dataset}_{args.epsilon}_{args.exp_note}"
    i = 1
    exp_id = prefix + f"({i})"
    while os.path.isdir(prefix + f"({i})"):
        i += 1
        exp_id = prefix + f"({i})"
    writer = SummaryWriter(log_dir="runs/"+exp_id)
    lr = args.lr
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, lr)
        train_loss = train(epoch, optimizer,exp_id=exp_id)
        writer.add_scalar('train/loss', train_loss, epoch)
        if (epoch+1)%args.val_period == 0:
            natural_val, robust_val = test()
            writer.add_scalar('val/natural', natural_val, epoch)
            writer.add_scalar('val/robust', robust_val, epoch)


def pgd_attack(basic_net, inputs, targets, epsilon, num_steps, step_size):
    x = inputs.detach()
    x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    for i in range(num_steps):
        x.requires_grad_()
        with torch.enable_grad():
            loss = F.cross_entropy(basic_net(x), targets, size_average=False)
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + step_size*torch.sign(grad.detach())
        x = torch.min(torch.max(x, inputs - epsilon), inputs + epsilon)
        x = torch.clamp(x, 0.0, 1.0)
    return basic_net(x), x


def llm_eval(data, target, model, eps):
    inner_iter = 50
    s = 0.1
    x = data.detach()
    bs = x.size()[0]
    x.requires_grad_()
    with torch.enable_grad():
        loss = F.cross_entropy(model(x), target, reduction='sum')
    x_grad = grad(loss, x)[0].clone().detach()
    x.requires_grad_(False)

    if eps > 0:
        delta = torch.zeros_like(x).uniform_(-eps, eps)
        for _ in range(inner_iter):
            delta.requires_grad_()
            with torch.enable_grad():
                llm = (F.cross_entropy(model(x+delta), target, reduction='sum')\
                       - F.cross_entropy(model(x), target, reduction='sum') \
                       - torch.sum(delta * x_grad)) / bs
            delta_grad = grad(llm, [delta])[0]
            delta = delta.detach() + s * torch.sign(delta_grad.detach())
            delta = torch.min(torch.max(delta, 0.0-x), 1.0-x)
            delta = torch.clamp(delta, -eps, eps)
    else:
        delta = torch.zeros_like(x)

    # llr(delta;x,y)=l(x+delta;y)-l(x;y)-delta^T*x_grad
    llm = (F.cross_entropy(model(x+delta), target, reduction='sum') \
          - F.cross_entropy(model(x), target, reduction='sum') \
          - torch.sum(delta * x_grad)) / bs
    return llm


if __name__ == '__main__':
    main()


    # mode = 'ga'
    # path_list = []
    # path_list.append('/home/hh239/rulin/AdversariallyRobustDistillation/checkpoint/CIFAR10/normal_WideResNet_MobileNetV2_CIFAR10_0.03137254901960784_(1)/epoch=167.t7')
    # path_list.append(
    #     '/home/hh239/rulin/AdversariallyRobustDistillation/checkpoint/CIFAR10/ard_WideResNet_MobileNetV2_CIFAR10_0.03137254901960784_alpha=0.5_temp=1(1)/epoch=199.t7')
    # path_list.append('/home/hh239/rulin/AdversariallyRobustDistillation/checkpoint/CIFAR10/kdiga_ard_WideResNet_MobileNetV2_CIFAR10_0.03137254901960784_alpha=0.5_temp=1_gamma=10(1)/epoch=131.t7')
    # path_list.append('/home/hh239/rulin/AdversariallyRobustDistillation/checkpoint/CIFAR10/kd_WideResNet_MobileNetV2_CIFAR10_0.03137254901960784_alpha=0.5_temp=1(1)/epoch=199.t7')
    # path_list.append('/home/hh239/rulin/AdversariallyRobustDistillation/checkpoint/CIFAR10/kdiga_ard_WideResNet_MobileNetV2_CIFAR10_0.03137254901960784_alpha=0.5_temp=1_gamma=10_cleang(1)/epoch=117.t7')
    # path_list.append('/home/hh239/rulin/AdversariallyRobustDistillation/checkpoint/CIFAR10/kdiga_WideResNet_MobileNetV2_CIFAR10_0.03137254901960784_alpha=0.5_temp=1_gamma=1(1)/epoch=167.t7')
    # path_list.append(
    #     '/home/hh239/rulin/AdversariallyRobustDistillation/checkpoint/CIFAR10/kdiga_WideResNet_MobileNetV2_CIFAR10_0.03137254901960784_alpha=0.5_temp=1_gamma=100(1)/epoch=101.t7')
    # path_list.append(
    #     '/home/hh239/rulin/AdversariallyRobustDistillation/checkpoint/CIFAR10/kdiga_WideResNet_MobileNetV2_CIFAR10_0.03137254901960784_alpha=0.5_temp=1_gamma=10(1)/epoch=179.t7')
    #
    # for path in path_list:
    #     basic_net.load_state_dict(torch.load(path)['net'])
    #     basic_net.eval()
    #     epsilons = [i/255 for i in range(1,9)]
    #     epsilons = [0/255, 4/255, 8/255]
    #
    #     if mode == 'pgd':
    #         num_steps = 20
    #         robust_acc = []
    #         natural_acc = -1
    #         for eps in epsilons:
    #             step_size = 2 * eps / num_steps
    #             adv_correct = 0
    #             natural_correct = 0
    #             total = 0
    #             with torch.no_grad():
    #                 iterator = tqdm(testloader, ncols=0, leave=False)
    #                 for batch_idx, (inputs, targets) in enumerate(iterator):
    #                     inputs, targets = inputs.to(device), targets.to(device)
    #                     total += targets.size(0)
    #                     if natural_acc < 0:
    #                         natural_outputs = basic_net(inputs)
    #                         _, natural_predicted = natural_outputs.max(1)
    #                         natural_correct += natural_predicted.eq(targets).sum().item()
    #                     adv_outputs, pert_inputs = pgd_attack(basic_net, inputs, targets, eps, num_steps, step_size)
    #                     _, adv_predicted = adv_outputs.max(1)
    #                     adv_correct += adv_predicted.eq(targets).sum().item()
    #                     iterator.set_description(str(adv_predicted.eq(targets).sum().item() / targets.size(0)))
    #             robust_acc.append(100. * adv_correct / total)
    #             if natural_acc < 0:
    #                 natural_acc = 100. * natural_correct / total
    #         print(path)
    #         print(f"natural acc = {natural_acc:.4f}, robust acc = {robust_acc}")
    #     elif mode == 'llm':
    #         test_batches = 10
    #         avg_llm = [0.0 for _ in range(len(epsilons))]
    #         for i, eps in enumerate(epsilons):
    #             iterator = tqdm(testloader, ncols=0, leave=False)
    #             with torch.no_grad():
    #                 for batch_idx, data in enumerate(iterator):
    #                     if batch_idx >= test_batches:
    #                         break
    #                     inputs, labels = data[0].to(device), data[1].to(device)
    #                     avg_llm[i] += float(llm_eval(inputs, labels, basic_net, eps) / test_batches)
    #         print(path)
    #         print(f"llm = {avg_llm}")
    #     elif mode == 'ce':
    #         test_batches = 10
    #         l_ce = 0.0
    #         iterator = tqdm(testloader, ncols=0, leave=False)
    #         with torch.no_grad():
    #             for batch_idx, data in enumerate(iterator):
    #                 if batch_idx >= 10:
    #                     break
    #                 inputs, labels = data[0].to(device), data[1].to(device)
    #                 l_ce += float(XENT_loss(basic_net(inputs), labels) / test_batches)
    #         print(path)
    #         print(f"Cross Entropy Loss = {l_ce}")
    #     elif mode == 'ga':
    #         test_batches = 10
    #         l_g = 0.0
    #         iterator = tqdm(testloader, ncols=0, leave=False)
    #         with torch.no_grad():
    #             for batch_idx, data in enumerate(iterator):
    #                 if batch_idx >= 10:
    #                     break
    #                 inputs, labels = data[0].to(device), data[1].to(device)
    #
    #                 basic_outputs = basic_net(inputs)
    #                 teacher_outputs = teacher_net(inputs)
    #
    #                 x = inputs.detach()
    #                 x.requires_grad_()
    #                 with torch.enable_grad():
    #                     t_loss = XENT_loss(teacher_net(x), labels)
    #                 t_grad = torch.autograd.grad(t_loss, x)[0]
    #                 x.grad = None
    #                 with torch.enable_grad():
    #                     s_loss = XENT_loss(basic_net(x), labels)
    #                 s_grad = torch.autograd.grad(s_loss, x, create_graph=True)[0]
    #
    #                 l_g += float((s_grad - t_grad).norm(2) / test_batches)
    #         print(path)
    #         print(f"Gradient Alignment Loss = {l_g}")


