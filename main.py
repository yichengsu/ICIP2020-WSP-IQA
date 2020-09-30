import argparse
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data as Data
import torch.utils.data.distributed
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter

from dataset import KonIQ10KDataset
from huberloss import HuberLoss
from iqanet import IQANet

parser = argparse.ArgumentParser(description='BLIND NATURAL IMAGE QUALITY PREDICTION USING '
                                 'CONVOLUTIONAL NEURAL NETWORKS AND WEIGHTED SPATIAL POOLING')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs when using Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-a', '--arch', metavar='ARCH', default='checkpoint',
                    help='path to pretrained model without postfix (default: checkpoint)')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:6328', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--th', default=0, type=int,
                    help='Use which ones in order file to train (default: 0)')
parser.add_argument('--train-size', default=8073, type=int, metavar='SIZE',
                    help='how many files are used for training (default: 8073)')
parser.add_argument('--tensorboard', action='store_true',
                    help='Use Tensorboard to record results')
parser.add_argument('--comment', default='', type=str,
                    help='Comment log_dir suffix appended to the default log_dir.')


def main():
    args = parser.parse_args()
    args.timestep = time.strftime('%Y%m%d-%H%M%S', time.localtime())

    # subfolder with images
    args.images_folder = os.path.join(args.data, '1024x768')
    assert os.path.exists(args.images_folder), 'images folder not exists'

    # subfile with mos detail
    args.mos_file = os.path.join(args.data, "koniq10k_scores_and_distributions.csv")
    assert os.path.exists(args.mos_file), 'MOS file not exists'

    # order file
    args.order_file = './orders.csv'
    assert os.path.exists(args.order_file), 'order file not exists, please run generate_order.py'

    # checkpoints
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(0, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.tensorboard:
        # Tensorboard Writer
        writer = SummaryWriter(os.path.join('runs', args.timestep + args.comment))
    else:
        writer = None

    if gpu is not None:
        print("Use GPU: {} for training".format(gpu))

    if args.multiprocessing_distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=ngpus_per_node, rank=gpu)

    # create model
    if args.pretrained:
        print("=> using pre-trained model.")
        path = 'checkpoints/{}.pth.tar'.format(args.arch)
        state_dict = torch.load(path, map_location='cpu')['state_dict']
        model = IQANet()
        model.load_state_dict(state_dict)
    else:
        print("=> creating model (gpu:{})".format(gpu))
        model = IQANet()

    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    if args.multiprocessing_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    criterion = HuberLoss(1/9).cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

    cudnn.benchmark = True

    # Data loading
    mos_df = pd.read_csv(args.mos_file)
    order = pd.read_csv(args.order_file).iloc[args.th].to_numpy()
    order_train = order[:args.train_size]
    order_test = order[args.train_size:]
    mos_df_train = mos_df.iloc[order_train]
    mos_df_test = mos_df.iloc[order_test]

    train_dataset = KonIQ10KDataset(mos_df_train, args.images_folder)

    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                   sampler=train_sampler, shuffle=(train_sampler is None),
                                   pin_memory=True, num_workers=args.workers)

    val_loader = Data.DataLoader(dataset=KonIQ10KDataset(mos_df_test, args.images_folder, False),
                                 batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, 0, writer, args)
        return

    best_res = (0, 0, 0)
    for epoch in range(args.epochs):
        if args.multiprocessing_distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, writer, args)

        # evaluate on validation set
        loss, res = validate(val_loader, model, criterion, epoch, writer, args)
        scheduler.step(loss)

        if gpu == 0 and epoch > 50:
            is_best = res[0] > best_res[0]
            if is_best:
                best_res = res

            # save model after training
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_res': best_res,
                'args': args,
            }, is_best, 'checkpoints/{}'.format(args.timestep + args.comment))

    if writer:
        writer.flush()
        writer.close()


def train(train_loader, model, criterion, optimizer, epoch, writer, args):
    batch_time = AverageMeter('Time', ':6.2f', print_sum=True)
    data_time = AverageMeter('Data', ':6.2f', print_sum=True)
    losses = AverageMeter('Loss', ':.4e')
    result = ResultMeter()

    progress = ProgressMeter(args.epochs, [batch_time, data_time, losses, result], prefix="Train")

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        result.update(output, target)
        losses.update(loss.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    progress.display(epoch, optimizer.param_groups)
    if args.tensorboard:
        to_tensorboard = {'PLCC': result.PLCC, 'SRCC': result.SRCC,
                          'RMSE': result.RMSE, 'LOSS': losses.avg}
        writer_helper(writer, args.gpu, 'train', to_tensorboard, epoch)


def validate(val_loader, model, criterion, epoch, writer, args):
    batch_time = AverageMeter('Time', ':6.2f', print_sum=True)
    data_time = AverageMeter('Data', ':6.2f', print_sum=True)
    losses = AverageMeter('Loss', ':.4e')
    result = ResultMeter()

    progress = ProgressMeter(args.epochs, [batch_time, data_time, losses, result], prefix='*Test')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            result.update(output, target)
            losses.update(loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    progress.display(epoch)
    if args.tensorboard:
        to_tensorboard = {'PLCC': result.PLCC, 'SRCC': result.SRCC,
                          'RMSE': result.RMSE, 'LOSS': losses.avg}
        writer_helper(writer, args.gpu, 'valid', to_tensorboard, epoch)

    return losses.avg, (result.PLCC, result.SRCC, result.RMSE)


def writer_helper(writer, gpu, tag, to_tensorboard, epoch):
    writer.add_scalars('gpu{}/PLCC'.format(gpu), {tag: to_tensorboard['PLCC']}, global_step=epoch)
    writer.add_scalars('gpu{}/SRCC'.format(gpu), {tag: to_tensorboard['SRCC']}, global_step=epoch)
    writer.add_scalars('gpu{}/RMSE'.format(gpu), {tag: to_tensorboard['RMSE']}, global_step=epoch)
    writer.add_scalars('gpu{}/LOSS'.format(gpu), {tag: to_tensorboard['LOSS']}, global_step=epoch)


class AverageMeter(object):
    """Computes and stores the average"""

    def __init__(self, name, fmt=':f', print_sum=False):
        self.name = name
        self.fmt = fmt
        self.print_sum = print_sum
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.print_sum:
            fmtstr = '{name} {sum' + self.fmt + '}'
        else:
            fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ResultMeter(object):
    def __init__(self):
        self.y_pred_all = torch.zeros(0, dtype=torch.float32)
        self.y_all = torch.zeros(0, dtype=torch.float32)

    def update(self, y_pred, y):
        self.y_pred_all = torch.cat((self.y_pred_all, y_pred.cpu()), dim=0)
        self.y_all = torch.cat((self.y_all, y.cpu()), dim=0)

    def __str__(self):
        y_all = distribution_to_mos(self.y_all)
        y_pred_all = distribution_to_mos(self.y_pred_all)

        self.PLCC = pearsonr(y_all, y_pred_all)[0]
        self.SRCC = spearmanr(y_all, y_pred_all)[0]
        self.RMSE = np.sqrt(mean_squared_error(y_all, y_pred_all))
        return 'PLCC=%.4f|SRCC=%.4f|RMSE=%.4f' % (self.PLCC, self.SRCC, self.RMSE)


class ProgressMeter(object):
    """Control print"""

    def __init__(self, num_epochs, meters, prefix=""):
        """Args:
            num_epochs (int): total number of all epochs
            meters (list): list of AverageMeter
            prefix (str, optional): Defaults to "".
        """
        self.epoch_fmtstr = self._get_epoch_fmtstr(num_epochs)
        self.meters = meters
        self.prefix = prefix

    def display(self, epoch, param_groups=None):
        entries = [self.prefix, time.strftime('%m-%d %H:%M:%S', time.localtime()),
                   self.epoch_fmtstr.format(epoch)]
        entries += [str(meter) for meter in self.meters]
        if param_groups is not None:
            entries += ["(lr:{})".format('/'.join(['{:.0e}'.format(p['lr'])
                                                   for p in param_groups]))]
        print(' '.join(entries))

    def _get_epoch_fmtstr(self, num_epochs):
        num_digits = len(str(num_epochs // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_epochs) + ']'


def distribution_to_mos(t):
    assert t.shape[1] == 5
    return (t[:, 0]*1 + t[:, 1]*2 + t[:, 2]*3 + t[:, 3]*4 + t[:, 4]*5).detach().numpy()


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + '.pth.tar')
    if is_best:
        shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')


if __name__ == '__main__':
    main()
