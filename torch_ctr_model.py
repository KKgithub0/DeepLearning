# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import argparse

import numpy as np
import os
import sys
import io
import time
import random

"""
for training ecr ecom ctr model
"""

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='surrogateescape')

def LoadWordEmbedding():
    words = []
    embeddings = []
    with open('./model.bin', 'r', encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split(' ')
            if len(arr) != 129:
                continue
            words.append(arr[0])
            embeddings.append([np.float32(x) for x in arr[1:]])
    embeddings = torch.from_numpy(np.asarray(embeddings, dtype='float32'))
    #embeddings = embeddings.cuda()
    return words, embeddings

def word_to_index(word1, word2):
    word1 = word1.split('|')
    word2 = word2.split('|')
    # max num of cut
    count = 10
    word_seg1 = [0] * count
    word_seg2 = [0] * count
    for i in range(count):
        if i < len(word1) and word1[i] in words:
            word_seg1[i] = np.array(words.index(word1[i])).astype('int64')
        else:
            # words_size - 2 means UNK
            word_seg1[i] = np.array(len(words) - 2).astype('int64')

        if i < len(word2) and word2[i] in words:
            word_seg2[i] = np.array(words.index(word2[i])).astype('int64')
        else:
            # words_size - 2 means UNK
            word_seg2[i] = np.array(len(words) - 2).astype('int64')

    return np.array(word_seg1).astype('int64'), np.array(word_seg2).astype('int64')

class CtrDNNModel(nn.Module):
    def __init__(self):
        super(CtrDNNModel, self).__init__()
        #self.fc1 = Linear(in_features = 282, out_features = 512)
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.embedding.weight.requires_grad = True
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        return

    def forward(self, inputs, label = None):
        # embedding, pooling
        word1 = inputs[:, 0]
        word_vec1 = torch.randn(len(word1), 128)
        word_vec1 = word_vec1.cuda(args.local_rank)
        for i in range(len(word1)):
            word = word1[i]
            emb = self.embedding(word)
            emb = sum(emb) / len(emb)
            word_vec1[i, :] = emb
        word2 = inputs[:, 1]
        word_vec2 = torch.randn(len(word2), 128)
        word_vec2 = word_vec1.cuda(args.local_rank)
        for i in range(len(word2)):
            word = word2[i]
            emb = self.embedding(word)
            emb = sum(emb) / len(emb)
            word_vec2[i, :] = emb

        # concat
        concat_embed = torch.cat([word_vec1, word_vec2], 1)

        # dnn
        output1 = torch.relu(self.fc1(concat_embed))
        output2 = torch.relu(self.fc2(output1))
        output3 = torch.relu(self.fc3(output2))
        output4 = torch.relu(self.fc4(output3))
        output5 = torch.sigmoid(self.fc5(output4))
        output_final = output5
        return output_final[:, 0]

def load_data_new(fi):
    # load data
    data = []
    labels = []
    with open(fi, 'r', encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split('\t')
            if len(arr) != 3:
                continue

            word1, word2, label = arr

            seg_id1, seg_id2 = word_to_index(word1, word2)

            label = np.array(label).astype('float32')

            data.append([seg_id1, seg_id2])
            labels.append(label)

    #return torch.from_numpy(np.array(data, dtype='int64')), torch.from_numpy(np.array(labels, dtype=np.long))
    return np.array(data, dtype='int64'), np.array(labels, dtype='float32')

class myDataset(Dataset):
    def __init__(self, data_in, label_in):
        self.len = len(data_in)
        self.data = [data_in, label_in]
    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index]
    def __len__(self):
        return self.len

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def train(train_loader, model, criterion, optimizer, epoch, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for batch_id, data in enumerate(train_loader):
        data_time.update(time.time() - end)
        x, y = data
        x = x.cuda(args.local_rank, non_blocking=True)
        y = y.cuda(args.local_rank, non_blocking=True)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        predicts = model(x)

        # loss
        loss = criterion(predicts, y)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)
        reduced_acc1 = reduce_mean(acc1, args.nprocs)
        reduced_acc5 = reduce_mean(acc5, args.nprocs)

        losses.update(reduced_loss.item(), x.size(0))
        top1.update(reduced_acc1.item(), x.size(0))
        top5.update(reduced_acc5.item(), x.size(0))

        # backward
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def validate(valid_loader, model, criterion, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_id, data in enumerate(valid_loader):
            x, y = data
            x = x.cuda(args.local_rank, non_blocking=True)
            y = y.cuda(args.local_rank, non_blocking=True)

            # forward
            predicts = model(x)

            # loss
            loss = criterion(predicts, y)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)
            reduced_acc5 = reduce_mean(acc5, args.nprocs)

            losses.update(reduced_loss.item(), x.size(0))
            top1.update(reduced_acc1.item(), x.size(0))
            top5.update(reduced_acc5.item(), xsize(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == '__main__':
    # env
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    # pretrain embed
    words, embeddings = LoadWordEmbedding()

    # model
    model = CtrDNNModel()
    # model to gpu
    model.cuda(args.local_rank)
    if torch.cuda.device_count() > 1:
        print("use {} gpus".format(torch.cuda.device_count()))
        model = DDP(model, device_ids=[args.local_rank])

    # loss func
    criterion = nn.BCELoss().cuda(args.local_rank)

    # opt func
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #dist_optim = DistributedOptimizer(optim.SGD, model.parameters(), lr=0.001,)

    cudnn.benchmark = True

    # data loader
    # training set
    data_in1, label_in1 = load_data_new('./training_set')
    train_dataset = myDataset(data_in1, label_in1)
    train_sampler = DistributedSampler(train_dataset, rank=args.local_rank)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=False, sampler=train_sampler)
    # valid set
    data_in2, label_in2 = load_data_new('./validate_set')
    valid_dataset = myDataset(data_in2, label_in2)
    valid_sampler = DistributedSampler(valid_dataset, rank=args.local_rank)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1024, shuffle=False, sampler=valid_sampler)

    if args.evaluate:
        validate(valid_loader, model, criterion, args.local_rank, args)
        exit(0)

    # train
    best_acc1 = .0
    num_epochs = 50
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        train(train_loader, model, criterion, optimizer, epoch, args.local_rank, args)

        acc1 = validate(valid_loader, model, criterion, args.local_rank, args)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if args.local_rank == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.module.state_dict(),
                    'best_acc1': best_acc1,
                }, is_best)
