import argparse
import copy
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from apex import amp
from robustbench.utils import load_model

from model.preact_resnet import PreActResNet18
from model.wide_resnet import WideResNet
from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard, evaluate_autoattack)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--data-dir', default='~/rulin/dataset', type=str)
    parser.add_argument('--alp', default=0.5, type=float)
    parser.add_argument('--temp', default=1., type=float)
    parser.add_argument('--gama', default=1000, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--rounds', default=5, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='fast_isd_output/', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"Output are saved at {args.out_dir}...")
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile) and not args.resume:
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(args.out_dir, 'output.log'))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    pgd_alpha = (2 / 255.) / std

    model = PreActResNet18().cuda()
    if args.resume:
        model.load_state_dict(torch.load(os.path.join(args.out_dir, 'model.pth')))
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    # amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    # if args.opt_level == 'O2':
    #     amp_args['master_weights'] = args.master_weights
    # model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()

    if args.delta_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # Training
    prev_robust_acc = 0.
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = len(train_loader)
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            if i == 0:
                first_batch = (X, y)
            if args.delta_init != 'previous':
                delta = torch.zeros_like(X).cuda()
            if args.delta_init == 'random':
                for j in range(len(epsilon)):
                    delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            ori_delta = delta
            delta.requires_grad = True
            output_s = model(X + delta[:X.size(0)])
            loss = F.cross_entropy(output_s, y)
            grad = torch.autograd.grad(loss, delta, create_graph=True)[0]
            delta.data = clamp(delta + alpha * torch.sign(grad.detach()), -epsilon, epsilon)
            delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            delta = delta.detach()
            output = model(X + delta[:X.size(0)])

            iga_loss = args.gama / X.shape[0] * (grad - grad_t).norm(2)
            loss = criterion(output, y) + iga_loss

            opt.zero_grad()
            loss.backward()

            opt.step()
            train_loss += loss.item() / train_n
            train_acc += (output.max(1)[1] == y).to(dtype=torch.float).mean().item() / train_n

            scheduler.step()
        if args.early_stop:
            _, robust_acc = evaluate_autoattack(model, 1000)
            if robust_acc > prev_robust_acc:
                prev_robust_acc = robust_acc
                best_state_dict = copy.deepcopy(model.state_dict())
            if robust_acc - prev_robust_acc < -0.2:
                break
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss, train_acc)
   
    for round in range(args.rounds):
        teacher_net = PreActResNet18().cuda()
        teacher_net.load_state_dict(best_state_dict)
        model = PreActResNet18().cuda()
        logger.info('Round \t Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
        for epoch in range(args.epochs):
            start_epoch_time = time.time()
            train_loss = 0
            train_acc = 0
            train_n = len(train_loader)
            for i, (X, y) in enumerate(train_loader):
                X, y = X.cuda(), y.cuda()
                if i == 0:
                    first_batch = (X, y)
                if args.delta_init != 'previous':
                    delta = torch.zeros_like(X).cuda()
                if args.delta_init == 'random':
                    for j in range(len(epsilon)):
                        delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                ori_delta = delta
                delta.requires_grad = True
                output_s = model(X + delta[:X.size(0)])
                loss = F.cross_entropy(output_s, y)
                grad = torch.autograd.grad(loss, delta, create_graph=True)[0]
                delta.data = clamp(delta + alpha * torch.sign(grad.detach()), -epsilon, epsilon)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
                delta = delta.detach()
                output = model(X + delta[:X.size(0)])

                ori_delta.requires_grad = True
                output_t = teacher_net(X + ori_delta[:X.size(0)])
                with torch.enable_grad():
                    loss_t = F.cross_entropy(output_t, y)
                grad_t = torch.autograd.grad(loss_t, ori_delta)[0]

                kl_loss = args.temp * args.temp * F.kl_div(F.log_softmax(model(X + ori_delta[:X.size(0)].detach()) / args.temp, dim=1),
                                                        F.softmax(output_t.detach() / args.temp, dim=1))
                iga_loss = args.gama / X.shape[0] * (grad - grad_t).norm(2)
                loss = criterion(output, y) + kl_loss + iga_loss

                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item() / train_n
                train_acc += (output.max(1)[1] == y).to(dtype=torch.float).mean().item() / train_n
                scheduler.step()
            if args.early_stop:
                _, robust_acc = evaluate_autoattack(model, 1000)
                if robust_acc > prev_robust_acc:
                    prev_robust_acc = robust_acc
                    best_state_dict = copy.deepcopy(model.state_dict())
                if robust_acc - prev_robust_acc < -0.2:
                    break
            epoch_time = time.time()
            lr = scheduler.get_lr()[0]
            logger.info('%d \t %d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                round, epoch, epoch_time - start_epoch_time, lr, train_loss, train_acc)
    
    train_time = time.time()
    if not args.early_stop:
        best_state_dict = model.state_dict()
    torch.save(best_state_dict, os.path.join(args.out_dir, 'model.pth'))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
    model_test = PreActResNet18().cuda()
    model_test.load_state_dict(best_state_dict)
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)
    aa_loss, aa_acc = evaluate_autoattack(model_test)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc \t AA Loss \t AA Acc')
    logger.info('%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f '%(test_loss, test_acc, pgd_loss, pgd_acc, aa_loss, aa_acc))


if __name__ == "__main__":
    main()