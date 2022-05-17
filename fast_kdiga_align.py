import argparse
import copy
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp
from robustbench.utils import load_model

# import sys
# import inspect

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)

from model.preact_resnet import PreActResNet18
from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard, evaluate_autoattack)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--out-dir', default='fast_kdiga_output/fast_kdiga_with_grad_align/debug/', type=str, help='Output directory')
    parser.add_argument('--data-dir', default='~/rulin/dataset', type=str)
    parser.add_argument('--robustbench-teacher', default='Gowal2021Improving_28_10_ddpm_100m', type=str)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--temp', default=1., type=float)
    parser.add_argument('--iga_lambda', default=1, type=int)
    parser.add_argument('--kd_lambda', default=0.5, type=float)
    parser.add_argument('--adv_lambda', default=0.5, type=float)
    parser.add_argument('--grad_align_cos_lambda', default=0.2, type=float)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
        help='Perturbation initialization method')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    parser.add_argument('--description', default='Implemented fast kdiga with gradient alignment', type=str)
    return parser.parse_args()


def reset_delta(delta):
    delta = delta.detach()
    delta.requires_grad = True
    return delta


def l2_norm_batch(v):
    norms = (v ** 2).sum([1, 2, 3]) ** 0.5
    return norms


def cos_similarity(grad1, grad2):
    grads_nnz_idx = ((grad1**2).sum([1, 2, 3])**0.5 != 0) * ((grad2**2).sum([1, 2, 3])**0.5 != 0)
    grad1, grad2 = grad1[grads_nnz_idx], grad2[grads_nnz_idx]
    grad1_norms, grad2_norms = l2_norm_batch(grad1), l2_norm_batch(grad2)
    grad1_normalized = grad1 / grad1_norms[:, None, None, None]
    grad2_normalized = grad2 / grad2_norms[:, None, None, None]
    cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
    return cos


def main():
    args = get_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
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
    model.train()

    teacher_net = load_model(model_name=args.robustbench_teacher, dataset='cifar10', threat_model='Linf').cuda()
    teacher_net.eval()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    model, opt = amp.initialize(model, opt, **amp_args)
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
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            bs = X.size(0)
            if i == 0:
                first_batch = (X, y)
            if args.delta_init != 'previous':
                delta = torch.zeros_like(X).cuda()
            if args.delta_init == 'random':
                for j in range(len(epsilon)):
                    delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta = reset_delta(delta)
            output = model(X + delta[:bs])
            loss = F.cross_entropy(output, y)
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            grad = delta.grad.detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data[:bs] = clamp(delta[:bs], lower_limit - X, upper_limit - X)
            delta1 = reset_delta(delta)
            output = model(X + delta1[:bs])
            ce_loss_adv = criterion(output, y)
            
            # KDIGA with correct teacher alignment
            grad_s_adv = torch.autograd.grad(ce_loss_adv, delta1, create_graph=True)[0]
            delta2 = reset_delta(delta)
            output_t_adv = teacher_net(X + delta2[:bs])
            t_correct = (output_t_adv.detach().max(1)[1] == y).to(dtype=torch.float)
            loss_t_adv = criterion(output_t_adv, y)
            grad_t_adv = torch.autograd.grad(loss_t_adv, delta2)[0]
            grad_t_adv = grad_t_adv.detach()
            # Align iff teacher predicts right
            t_correct_ = t_correct[:, None].repeat(1, args.num_classes)
            kd_loss = args.temp * args.temp * F.kl_div(F.log_softmax(output / args.temp, dim=1) * t_correct_,
                                                       F.softmax(output_t_adv.detach() / args.temp, dim=1) * t_correct_)
            t_correct_ = t_correct[:, None, None, None].repeat([1]+list(grad_t_adv.shape[1:]))
            grad_diff = torch.flatten((grad_s_adv - grad_t_adv)[:bs] * t_correct_, start_dim=1)
            iga_loss = torch.linalg.norm(grad_diff, ord=2, dim=1).mean()
            # Grad Align regularizer
            delta = torch.zeros_like(X, requires_grad=True)
            output = model(X + delta)
            loss = F.cross_entropy(output, y)
            with amp.scale_loss(loss, opt) as scaled_loss:
                grad1 = torch.autograd.grad(scaled_loss, delta, create_graph=True)[0]
                grad1 /= scaled_loss / loss
            grad2 = grad_s_adv.detach()
            cos = cos_similarity(grad1, grad2)
            reg = (1.0 - cos.mean())
            loss = args.adv_lambda * ce_loss_adv + args.kd_lambda *  kd_loss + args.iga_lambda * iga_loss + args.grad_align_cos_lambda * reg

            opt.zero_grad()
            with amp.scale_loss(ce_loss_adv, opt) as scaled_loss:
                scaled_loss.backward()
            opt.step()
            train_loss += ce_loss_adv.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()
        if args.early_stop:
            # Check current PGD robustness of model using random minibatch
            X, y = first_batch
            pgd_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 5, 1, opt)
            with torch.no_grad():
                output = model(clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
            robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
            if robust_acc - prev_robust_acc < -0.2:
                break
            prev_robust_acc = robust_acc
            best_state_dict = copy.deepcopy(model.state_dict())
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
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