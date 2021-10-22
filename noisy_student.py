from __future__ import print_function
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import grad
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models import *


parser = argparse.ArgumentParser(description='Noisy Student CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_schedule', type=int, nargs='+', default=[50, 100], help='Decrease learning rate at these epochs.')
parser.add_argument('--lr_factor', default=0.1, type=float, help='factor by which to decrease lr')
parser.add_argument('--epochs', default=150, type=int, help='number of epochs for training')
parser.add_argument('--model', default = 'MobileNetV2', type = str, help = 'student model name')
parser.add_argument('--teacher_model', default = 'WideResNet', type = str, help = 'initial teacher network model')
parser.add_argument('--teacher_path', default = '../checkpoint/trades/model_cifar_wrn.pt', type=str, help='path of teacher net being distilled')
parser.add_argument('--temp', default=1.0, type=float, help='temperature for distillation')
parser.add_argument('--val_period', default=1, type=int, help='print every __ epoch')
parser.add_argument('--save_period', default=1, type=int, help='save every __ epoch')
parser.add_argument('--alpha', default=0.5, type=float, help='weight for sum of losses')
parser.add_argument('--gamma', default=1, type=float, help='use gamma/bs for iga')
parser.add_argument('--dataset', default = 'CIFAR10', type=str, help='name of dataset')
parser.add_argument('--noisy_student_loop', default=3)
parser.add_argument('--output', default = '1022', type=str, help='output subdirectory')
parser.add_argument('--exp_note', dnefault='noisy_student_alpha0.5_gamma1')

# PGD attack
parser.add_argument('--epsilon', default=8/255)
parser.add_argument('--num_steps', default=10)
parser.add_argument('--step_size', default=16/255/10)
args = parser.parse_args()

config = {
    'epsilon': args.epsilon,
    'num_steps': args.num_steps,
    'step_size': args.step_size,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True

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


def build_student_model(model_name=args.model):
    print('==> Building model..'+args.model)
    if model_name == 'MobileNetV2':
        basic_net = MobileNetV2(num_classes=num_classes)
    elif model_name == 'WideResNet':
        basic_net = WideResNet(num_classes=num_classes)
    elif model_name == 'ResNet18':
        basic_net = ResNet18(num_classes=num_classes)
    else:
        raise AttributeError
    basic_net = basic_net.to(device)
    return basic_net

def build_teacher_model(model_name=args.teacher_model):
    assert args.teacher_path != ''
    if model_name == 'MobileNetV2':
        teacher_net = MobileNetV2(num_classes=num_classes)
    elif model_name == 'WideResNet':
        teacher_net = WideResNet(num_classes=num_classes)
    elif model_name == 'ResNet18':
        teacher_net = ResNet18(num_classes=num_classes)
    else:
        raise AttributeError
    teacher_net = teacher_net.to(device)
    for param in teacher_net.parameters():
        param.requires_grad = False
    return teacher_net


def build_mode(loop=0, exp_id=None):
    basic_net = build_student_model()
    net = AttackPGD(basic_net, config)
    print(f"==> Loading teacher from {args.teacher_path}")
    if loop == 0:
        teacher_net = build_teacher_model()
        teacher_net.load_state_dict(torch.load(args.teacher_path))
    else:
        assert exp_id is not None
        teacher_net = build_teacher_model(args.model)
        _, best_robust_path = best_paths(exp_id=exp_id)
        teacher_net.load_state_dict(torch.load(best_robust_path)['net'])
    teacher_net.eval()
    return basic_net, net, teacher_net


def train(basic_net, net, teacher_net, KL_loss, XENT_loss, epoch, optimizer, exp_id=''):
    train_loss = 0
    iterator = tqdm(trainloader, ncols=0, leave=False)
    basic_net.train()
    for batch_idx, (inputs, targets) in enumerate(iterator):
        inputs, targets = inputs.to(device), targets.to(device)
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

    if (epoch+1)%args.save_period == 0:
        save_model(basic_net, optimizer, exp_id, '/epoch='+str(epoch)+'.t7')

    print('Mean Training Loss:', train_loss/len(iterator))
    return train_loss


def test(basic_net, net):
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


def save_model(basic_net, optimizer, exp_id, name):
    state = {
        'net': basic_net.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    if not os.path.isdir('checkpoint/' + args.dataset + '/' + exp_id + '/'):
        os.makedirs('checkpoint/' + args.dataset + '/' + exp_id + '/', )
    torch.save(state, './checkpoint/' + args.dataset + '/' + exp_id + name)


def best_paths(exp_id):
    natural_best = './checkpoint/' + args.dataset + '/' + exp_id + '/best_natural.t7'
    robust_best  = './checkpoint/' + args.dataset + '/' + exp_id + '/best_robust.t7'
    return natural_best, robust_best


def evaluate(test_student_path):
    basic_net, net, teacher_net = build_mode()
    basic_net.load_state_dict(torch.load(test_student_path)['net'])
    basic_net.eval()
    natural_val, robust_val = test()
    print(test_student_path)
    print(f"natural acc = {natural_val:.4f}, robust acc = {robust_val:.4f}")


def main():
    for loop in range(args.noisy_student_loop):
        prefix = f"{args.output}/NoisyStudent_loop{loop}_{args.exp_note}"
        i = 1
        exp_id = prefix + f"({i})"
        while os.path.isdir(prefix + f"({i})"):
            i += 1
            exp_id = prefix + f"({i})"
        writer = SummaryWriter(log_dir="runs/"+exp_id+f"loop{loop}")

        basic_net, net, teacher_net = build_mode(loop=loop)

        lr = args.lr
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
        KL_loss = nn.KLDivLoss()
        XENT_loss = nn.CrossEntropyLoss()

        best_natural_val = .0
        best_robust_val = .0
        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer, epoch, lr)
            train_loss = train(basic_net, net, teacher_net, KL_loss, XENT_loss, epoch, optimizer,exp_id=exp_id)
            writer.add_scalar('train/loss', train_loss, epoch)
            if (epoch+1)%args.val_period == 0:
                natural_val, robust_val = test(basic_net, net)
                if natural_val > best_natural_val:
                    save_model(basic_net, optimizer, exp_id, '/best_natural.t7')
                if robust_val > best_robust_val:
                    save_model(basic_net, optimizer, exp_id, '/best_robust.t7')
                writer.add_scalar('val/natural', natural_val, epoch)
                writer.add_scalar('val/robust', robust_val, epoch)


if __name__ == '__main__':
    main()