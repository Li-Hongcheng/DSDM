import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from data import transform_imagenet, transform_cifar, transform_svhn, transform_mnist, transform_fashion
from data import TensorDataset, save_img
from data import ClassDataLoader, ClassMemDataLoader, MultiEpochsDataLoader
from data import MEANS, STDS
from train import define_model, train_epoch
from test import test_data, load_ckpt
from misc.augment import DiffAug
from misc import utils
from math import ceil
import random


class Synthesizer():
    """Condensed data class
    """

    def __init__(self, args, nclass, nchannel, hs, ws, device='cuda'):
        self.ipc = args.ipc
        self.nclass = nclass
        self.nchannel = nchannel
        self.size = (hs, ws)
        self.device = device

        self.data = torch.randn(size=(self.nclass * self.ipc, self.nchannel, hs, ws),
                                dtype=torch.float,
                                requires_grad=True,
                                device=self.device)
        self.data.data = torch.clamp(self.data.data / 4 + 0.5, min=0., max=1.)
        self.targets = torch.tensor([np.ones(self.ipc) * i for i in range(nclass)],
                                    dtype=torch.long,
                                    requires_grad=False,
                                    device=self.device).view(-1)
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(self.data.shape[0]):
            self.cls_idx[self.targets[i]].append(i)

        print("\nDefine synthetic data: ", self.data.shape)

        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type
        self.resize = nn.Upsample(size=self.size, mode='bilinear')
        print(f"Factor: {self.factor} ({self.decode_type})")

    def init(self, loader, init_type='noise'):
        """Condensed data initialization
        """
        if init_type == 'random':
            print("Random initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc)
                self.data.data[self.ipc * c:self.ipc *
                               (c + 1)] = img.data.to(self.device)

        elif init_type == 'mix':
            print("Mixed initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc * self.factor**2)
                img = img.data.to(self.device)

                s = self.size[0] // self.factor
                remained = self.size[0] % self.factor
                k = 0
                n = self.ipc

                h_loc = 0
                for i in range(self.factor):
                    h_r = s + 1 if i < remained else s
                    w_loc = 0
                    for j in range(self.factor):
                        w_r = s + 1 if j < remained else s
                        img_part = F.interpolate(
                            img[k * n:(k + 1) * n], size=(h_r, w_r))
                        self.data.data[n * c:n * (c + 1), :, h_loc:h_loc + h_r,
                                       w_loc:w_loc + w_r] = img_part
                        w_loc += w_r
                        k += 1
                    h_loc += h_r

        elif init_type == 'noise':
            pass

    def parameters(self):
        parameter_list = [self.data]
        return parameter_list

    def subsample(self, data, target, max_size=-1):
        if (data.shape[0] > max_size) and (max_size > 0):
            indices = np.random.permutation(data.shape[0])
            data = data[indices[:max_size]]
            target = target[indices[:max_size]]

        return data, target

    def decode_zoom(self, img, target, factor):
        """Uniform multi-formation
        """
        h = img.shape[-1]
        remained = h % factor
        if remained > 0:
            img = F.pad(img, pad=(0, factor - remained,
                        0, factor - remained), value=0.5)
        s_crop = ceil(h / factor)
        n_crop = factor**2

        cropped = []
        for i in range(factor):
            for j in range(factor):
                h_loc = i * s_crop
                w_loc = j * s_crop
                cropped.append(
                    img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
        cropped = torch.cat(cropped)
        data_dec = self.resize(cropped)
        target_dec = torch.cat([target for _ in range(n_crop)])

        return data_dec, target_dec

    def decode_zoom_multi(self, img, target, factor_max):
        """Multi-scale multi-formation
        """
        data_multi = []
        target_multi = []
        for factor in range(1, factor_max + 1):
            decoded = self.decode_zoom(img, target, factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

        return torch.cat(data_multi), torch.cat(target_multi)

    def decode_zoom_bound(self, img, target, factor_max, bound=128):
        """Uniform multi-formation with bounded number of synthetic data
        """
        bound_cur = bound - len(img)
        budget = len(img)

        data_multi = []
        target_multi = []

        idx = 0
        decoded_total = 0
        for factor in range(factor_max, 0, -1):
            decode_size = factor**2
            if factor > 1:
                n = min(bound_cur // decode_size, budget)
            else:
                n = budget

            decoded = self.decode_zoom(
                img[idx:idx + n], target[idx:idx + n], factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

            idx += n
            budget -= n
            decoded_total += n * decode_size
            bound_cur = bound - decoded_total - budget

            if budget == 0:
                break

        data_multi = torch.cat(data_multi)
        target_multi = torch.cat(target_multi)
        return data_multi, target_multi

    def decode(self, data, target, bound=128):
        """Multi-formation
        """
        if self.factor > 1:
            if self.decode_type == 'multi':
                data, target = self.decode_zoom_multi(
                    data, target, self.factor)
            elif self.decode_type == 'bound':
                data, target = self.decode_zoom_bound(
                    data, target, self.factor, bound=bound)
            else:
                data, target = self.decode_zoom(data, target, self.factor)

        return data, target

    def sample(self, c, max_size=128):
        """Sample synthetic data per class
        """
        idx_from = self.ipc * c
        idx_to = self.ipc * (c + 1)
        data = self.data[idx_from:idx_to]
        target = self.targets[idx_from:idx_to]

        data, target = self.decode(data, target, bound=max_size)
        data, target = self.subsample(data, target, max_size=max_size)
        return data, target

    def loader(self, args, augment=True):
        """Data loader for condensed data
        """
        if args.dataset == 'imagenet':
            train_transform, _ = transform_imagenet(augment=augment,
                                                    from_tensor=True,
                                                    size=0,
                                                    rrc=args.rrc,
                                                    rrc_size=self.size[0])
        elif args.dataset[:5] == 'cifar':
            train_transform, _ = transform_cifar(
                augment=augment, from_tensor=True)
        elif args.dataset == 'svhn':
            train_transform, _ = transform_svhn(
                augment=augment, from_tensor=True)
        elif args.dataset == 'mnist':
            train_transform, _ = transform_mnist(
                augment=augment, from_tensor=True)
        elif args.dataset == 'fashion':
            train_transform, _ = transform_fashion(
                augment=augment, from_tensor=True)

        data_dec = []
        target_dec = []
        for c in range(self.nclass):
            idx_from = self.ipc * c
            idx_to = self.ipc * (c + 1)
            data = self.data[idx_from:idx_to].detach()
            target = self.targets[idx_from:idx_to].detach()
            data, target = self.decode(data, target)

            data_dec.append(data)
            target_dec.append(target)

        data_dec = torch.cat(data_dec)
        target_dec = torch.cat(target_dec)

        train_dataset = TensorDataset(
            data_dec.cpu(), target_dec.cpu(), train_transform)

        print("Decode condensed data: ", data_dec.shape)
        nw = 0 if not augment else args.workers
        train_loader = MultiEpochsDataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=nw,
                                             persistent_workers=nw > 0)
        return train_loader

    def test(self, args, val_loader, logger):
        """Condensed data evaluation
        """
        loader = self.loader(args, args.augment)
        result = test_data(args, loader, val_loader,
                           test_resnet=False, logger=logger)

        return result


def load_resized_data(args):
    """Load original training data (fixed spatial size and without augmentation) for condensation
    """
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            args.data_dir, train=True, transform=transforms.ToTensor())
        normalize = transforms.Normalize(
            mean=MEANS['cifar10'], std=STDS['cifar10'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR10(
            args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(args.data_dir,
                                          train=True,
                                          transform=transforms.ToTensor())

        normalize = transforms.Normalize(
            mean=MEANS['cifar100'], std=STDS['cifar100'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR100(
            args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 100

    elif args.dataset == 'svhn':
        train_dataset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                      split='train',
                                      transform=transforms.ToTensor())
        train_dataset.targets = train_dataset.labels

        normalize = transforms.Normalize(mean=MEANS['svhn'], std=STDS['svhn'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                    split='test',
                                    transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'mnist':
        train_dataset = datasets.MNIST(
            args.data_dir, train=True, transform=transforms.ToTensor())

        normalize = transforms.Normalize(
            mean=MEANS['mnist'], std=STDS['mnist'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.MNIST(
            args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'fashion':
        train_dataset = datasets.FashionMNIST(args.data_dir,
                                              train=True,
                                              transform=transforms.ToTensor())

        normalize = transforms.Normalize(
            mean=MEANS['fashion'], std=STDS['fashion'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.FashionMNIST(
            args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10



    val_loader = MultiEpochsDataLoader(val_dataset,
                                       batch_size=args.batch_size // 2,
                                       shuffle=False,
                                       persistent_workers=True,
                                       num_workers=4)

    # width check
    assert train_dataset[0][0].shape[-1] == val_dataset[0][0].shape[-1]

    return train_dataset, val_loader


def remove_aug(augtype, remove_aug):
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)

    return "_".join(aug_list)


def diffaug(args, device='cuda'):
    """Differentiable augmentation for condensation
    """
    aug_type = args.aug_type
    normalize = utils.Normalize(
        mean=MEANS[args.dataset], std=STDS[args.dataset], device=device)
    print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    if args.mixup_net == 'cut':
        aug_type = remove_aug(aug_type, 'cutout')
    print("Augmentataion Net update: ", aug_type)
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand


def dist(x, y, method='mse'):
    """Distance objectives
    """
    if method == 'mse':
        dist_ = (x - y).pow(2).sum()
    elif method == 'l1':
        dist_ = (x - y).abs().sum()
    elif method == 'l1_mean':
        n_b = x.shape[0]
        dist_ = (x - y).abs().reshape(n_b, -1).mean(-1).sum()
    elif method == 'cos':
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        dist_ = torch.sum(1 - torch.sum(x * y, dim=-1) /
                          (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6))

    return dist_


def add_loss(loss_sum, loss):
    if loss_sum == None:
        return loss
    else:
        return loss_sum + loss


def matchloss(args, img_real, img_syn, model, h_p=None):
    loss = None
    k = img_real.shape[0]
    with torch.no_grad():
        feat_tg, _ = model.get_feature(
            img_real[:k], args.idx_from, args.idx_to)
    feat, _ = model.get_feature(img_syn, args.idx_from, args.idx_to)

    proto_loss = add_loss(loss, dist(
        feat_tg[len(feat_tg)-1].mean(0), feat[len(feat)-1].mean(0), method=args.metric))

    proto_tg = feat_tg[len(feat)-1].mean(0)
    proto_tg = proto_tg.view(proto_tg.shape[0], -1)
    proto_tg = proto_tg.reshape(-1)
    feat_tg_view = feat_tg[len(feat_tg)-1].view(feat_tg[len(feat_tg)-1].size(0), -1)   

    proto_syn = feat[len(feat)-1].mean(0)
    proto_syn = proto_syn.view(proto_syn.shape[0], -1)
    proto_syn = proto_syn.reshape(-1)
    feat_view = feat[len(feat) - 1].view(feat[len(feat) - 1].size(0), -1)

    centered_real = feat_tg_view - proto_tg
    centered_syn = feat_view - proto_syn

    cov_real = torch.matmul(centered_real.t(), centered_real) / (feat_tg_view.size(0) - 1)
    cov_syn = torch.matmul(centered_syn.t(), centered_syn) / (feat_view.size(0) - 1)
    semantic_loss = torch.mul(dist(cov_syn, cov_real, method=args.metric)/proto_syn.shape[0], args.cov_weight)
    loss = add_loss(proto_loss, semantic_loss)
    
    if h_p is not None:
        h_p_loss = torch.mul(dist(feat[len(feat)-1].mean(0), h_p, method=args.metric)/proto_syn.shape[0], args.h_p_weight)
        loss = add_loss(loss, h_p_loss)

    return loss




def condense(args, logger, device='cuda'):
    """Optimize condensed data
    """
    # Define real dataset and loader
    trainset, val_loader = load_resized_data(args)
    if args.load_memory:
        loader_real = ClassMemDataLoader(trainset, batch_size=args.batch_real)
    else:
        loader_real = ClassDataLoader(trainset,
                                      batch_size=args.batch_real,
                                      num_workers=args.workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)
    nclass = trainset.nclass
    nch, hs, ws = trainset[0][0].shape

    # Define syn dataset
    synset = Synthesizer(args, nclass, nch, hs, ws)
    synset.init(loader_real, init_type=args.init)
    save_img(os.path.join(args.save_dir, 'init.png'),
             synset.data,
             unnormalize=False,
             dataname=args.dataset)

    # Define augmentation function
    aug, aug_rand = diffaug(args)
    save_img(os.path.join(args.save_dir, f'aug.png'),
             aug(synset.sample(0, max_size=args.batch_syn_max)[0]),
             unnormalize=True,
             dataname=args.dataset)

    # if not args.test:
    #     synset.test(args, val_loader, logger, bench=False)

    # Data distillation
    optim_img = torch.optim.SGD(
        synset.parameters(), lr=args.lr_img, momentum=args.mom_img)

    ts = utils.TimeStamp(args.time)
   
    it_log = 20
    it_test = [i for i in range(0, args.niter+1, args.evaluate_iter)]

    logger(
        f"\nStart condensing with {args.match} matching for {args.niter} iteration")

    best_acc = -1

    smooth_syns = [None] * nclass 
    h_p = [None] * nclass

    for it in range(args.niter):
        j = random.randint(0, args.pretrained_model_number-1)
        model = define_model(args, nclass).to(device)
        if args.dataset == 'cifar10':
            model.load_state_dict(torch.load(
                f'./{args.save_pretrain_dir}/{args.dataset}_model_{j}.pth', map_location=device))
        elif args.dataset == 'cifar100':
            model.load_state_dict(torch.load(
                f'./{args.save_pretrain_dir}/{args.dataset}_model_{j}.pth', map_location=device))
        elif args.dataset == 'svhn':
            model.load_state_dict(torch.load(
                f'./{args.save_pretrain_dir}/{args.dataset}_model_{j}.pth', map_location=device))
        elif args.dataset == 'mnist':
            model.load_state_dict(torch.load(
                f'./{args.save_pretrain_dir}/{args.dataset}_model_{j}.pth', map_location=device))
        elif args.dataset == 'fashion':
            model.load_state_dict(torch.load(
                f'./{args.save_pretrain_dir}/{args.dataset}_model_{j}.pth', map_location=device))

        loss_total = 0
        synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)
        ts.set()

        for c in range(nclass):
            img, lab = loader_real.class_sample(c)
            img_syn, lab_syn = synset.sample(c, max_size=args.batch_syn_max)
            ts.stamp("data")
            n = img.shape[0]
            img_aug = aug(torch.cat([img, img_syn]))
            ts.stamp("aug")

            optim_img.zero_grad()
            if it > args.smooth_iter:
                loss = matchloss(args, img_aug[:n], img_aug[n:], model, h_p=h_p[c])
            else:
                loss = matchloss(args, img_aug[:n], img_aug[n:], model)
            loss_total += loss.item()
            ts.stamp("loss")
            loss.backward()
            optim_img.step()
            ts.stamp("backward")
            ts.flush()
     
        for c in range(nclass):
            img_syn, lab_syn = synset.sample(c, max_size=args.batch_syn_max)
            ts.stamp("data")
            syn_img_aug = aug(torch.cat([img_syn]))
            ts.stamp("aug")
            if (it == 0):
                with torch.no_grad():
                    feature_h, _ = model.get_feature(syn_img_aug, args.idx_from, args.idx_to)
                smooth_syns[c] = feature_h[len(feature_h)-1].mean(0)
                h_p[c] = smooth_syns[c]
            else:
                with torch.no_grad():
                    feature_h, _ = model.get_feature(syn_img_aug, args.idx_from, args.idx_to)
                smooth_syns[c] = feature_h[len(feature_h)-1].mean(0)
                h_p[c] = (1 -args.smooth_factor) * smooth_syns[c] + args.smooth_factor * h_p[c]

         # Logging
        if it % it_log == 0:
            logger(
                f"{utils.get_time()} (Iter {it:3d}) loss: {loss_total/nclass:.1f}")
        if (it + 1) in it_test:
            if not args.test:
                conv_result = synset.test(args, val_loader, logger)
                if conv_result > best_acc:
                    best_acc = conv_result
                    torch.save(
                        [synset.data.detach().cpu(), synset.targets.cpu()],
                        os.path.join(args.save_dir, f'data_best.pt'))
                    print("best img and data updated!")
                    save_img(os.path.join(args.save_dir, f'img{it+1}.png'),
                             synset.data,
                             unnormalize=False,
                             dataname=args.dataset)
                logger(
                    "->->->->->->->->->->->->-> Best Result: {:.1f}".format(best_acc))


if __name__ == '__main__':
    import shutil
    from misc.utils import Logger
    from argument import args
    import torch.backends.cudnn as cudnn
    import json

    assert args.ipc > 0

    cudnn.benchmark = True
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    cur_file = os.path.join(os.getcwd(), __file__)
    shutil.copy(cur_file, args.save_dir)

    logger = Logger(args.save_dir)
    logger(f"Save dir: {args.save_dir}")
    with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    condense(args, logger)
