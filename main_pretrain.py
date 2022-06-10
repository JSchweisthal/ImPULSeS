import os
import numpy as np
import torch
import torchvision
import argparse

import torch.nn.functional as F

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SimCLR
from simclr import SimCLR
from simclr.modules import get_resnet
from simclr.modules.transformations import TransformsSimCLR

from simclr.modules.model import load_optimizer, save_model
from utils import yaml_config_hook


###new
def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

def train(args, train_loader, model, optimizer, writer):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        _, _, out_1, out_2 = model(x_i, x_j)
        out_1, out_2 = F.normalize(out_1, dim=-1), F.normalize(out_2, dim=-1)
        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / args.temperature)
        mask = get_negative_mask(args.batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * args.batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.temperature)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        if args.debiased:
            N = args.batch_size * 2 - 2
            Ng = (-args.tau_plus * N * pos + neg.sum(dim = -1)) / (1 - args.tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / args.temperature))
        else:
            Ng = neg.sum(dim=-1)

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng) )).mean()

        loss.backward()

        optimizer.step()
        


        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        
        writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
        args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch


def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(
            args.dataset_dir,
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )

    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
        
        # just take 750 instead of 5000 of the 4 vehicle (positive) classes --> P:U ratio 3k : 30k
        if args.data_pretrain == "imbalanced" and "2class" not in args.data_pretrain:
            idxs = []
            for cls in range(10):
                idxs_cls = [i for i in range(len(train_dataset.targets)) if train_dataset.targets[i]==cls]
                if cls in [0, 1, 8, 9]:
                    if args.data_pretrain == "imbalanced":
                        idxs_cls = idxs_cls[:750]
                idxs.extend(idxs_cls)
                idxs.sort()

            train_datasubset_pu = torch.utils.data.Subset(train_dataset, idxs)

    else:
        raise NotImplementedError

    if args.data_pretrain == "all":
            train_datasubset_pu = train_dataset


    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_datasubset_pu, #train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler
    )
    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # initialize model
    model = SimCLR(encoder, args.projection_dim, n_features)
    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.tar".format(args.model_num)
        )
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)

    model = model.to(args.device)

    writer = SummaryWriter('runs/' + args.config)

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.reload:
            args.current_epoch = epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, optimizer, writer)

        if scheduler:
            scheduler.step()

        if epoch % 10 == 0:
            save_model(args, model, optimizer)
        
        writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)

        print(
            f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
        )
        args.current_epoch += 1

    ## end training
    save_model(args, model, optimizer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")

    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    config = yaml_config_hook(f"./config/{args.config}.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    main(args)