import argparse
import os
import sys
import time
import tabulate

import torch
import torch.nn.functional as F
import torchvision
import numpy as np

from swag import data, models, utils, losses
from swag.posteriors import SWAG, SAM

parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument(
    "--dir",
    type=str,
    default="TEST",
    required=False,
    help="training directory (default: None)",
)

parser.add_argument(
    "--dataset", type=str, default="CIFAR10", help="dataset name (default: CIFAR10)"
)

parser.add_argument("--sam", action="store_true")

parser.add_argument(
    "--data_path",
    type=str,
    default="swa_gaussian/data",
    required=False,
    metavar="PATH",
    help="path to datasets location (default: None)",
)
parser.add_argument(
    "--use_test",
    dest="use_test",
    action="store_true",
    help="use test dataset instead of validation (default: False)",
)
parser.add_argument("--split_classes", type=int, default=None)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size (default: 128)",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    metavar="N",
    help="number of workers (default: 4)",
)
parser.add_argument(
    "--model",
    type=str,
    default="VGG16",
    required=False,
    metavar="MODEL",
    help="model name (default: None)",
)

parser.add_argument(
    "--resume",
    type=str,
    default=None,
    metavar="CKPT",
    help="checkpoint to resume training from (default: None)",
)


parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    metavar="N",
    help="number of epochs to train (default: 200)",
)
parser.add_argument(
    "--save_freq",
    type=int,
    default=25,
    metavar="N",
    help="save frequency (default: 25)",
)
parser.add_argument(
    "--eval_freq",
    type=int,
    default=5,
    metavar="N",
    help="evaluation frequency (default: 5)",
)
parser.add_argument(
    "--lr_init",
    type=float,
    default=0.01,
    metavar="LR",
    help="initial learning rate (default: 0.01)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="SGD momentum (default: 0.9)",
)
parser.add_argument(
    "--wd", type=float, default=1e-4, help="weight decay (default: 1e-4)"
)

parser.add_argument("--swa", action="store_true", help="swa usage flag (default: off)")
parser.add_argument(
    "--swa_start",
    type=float,
    default=161,
    metavar="N",
    help="SWA start epoch number (default: 161)",
)
parser.add_argument(
    "--swa_lr", type=float, default=0.02, metavar="LR", help="SWA LR (default: 0.02)"
)
parser.add_argument(
    "--swa_c_epochs",
    type=int,
    default=1,
    metavar="N",
    help="SWA model collection frequency/cycle length in epochs (default: 1)",
)
parser.add_argument("--cov_mat", action="store_true", help="save sample covariance")
parser.add_argument(
    "--max_num_models",
    type=int,
    default=20,
    help="maximum number of SWAG models to save",
)

parser.add_argument(
    "--swa_resume",
    type=str,
    default=None,
    metavar="CKPT",
    help="checkpoint to restor SWA from (default: None)",
)
parser.add_argument(
    "--loss",
    type=str,
    default="CE",
    help="loss to use for training model (default: Cross-entropy)",
)

parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument("--no_schedule", action="store_true", help="store schedule")

parser.add_argument("--use_cswag", action="store_true")
parser.add_argument("--use_cyc", action="store_true")
parser.add_argument("--use_noise", action="store_true")
parser.add_argument("--num_cyc_samples", type=int, default=3)
parser.add_argument("--min_cyc_lr", type=float, default=0)
parser.add_argument(
    "--num_cycles", type=int, default=4, help="number of cycles for cyclical schedule"
)

parser.add_argument("--cyc_weights", action="store_true")
parser.add_argument("--samples_per_cycle", type=int, default=5)
args = parser.parse_args()
args.device = None

init_lr = args.lr_init
use_cuda = torch.cuda.is_available()

if use_cuda:
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")

print("Preparing directory %s" % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, "command.sh"), "w") as f:
    f.write(" ".join(sys.argv))
    f.write("\n")

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)

if use_cuda:
    torch.cuda.manual_seed(args.seed)

print("Using model %s" % args.model)
model_cfg = getattr(models, args.model)

print("Loading dataset %s from %s" % (args.dataset, args.data_path))
loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    model_cfg.transform_train,
    model_cfg.transform_test,
    use_validation=not args.use_test,
    split_classes=args.split_classes,
)

print("Preparing model")
print(*model_cfg.args)
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.to(args.device)


if args.cov_mat:
    args.no_cov_mat = False
else:
    args.no_cov_mat = True
if args.swa:
    if args.use_cyc:
        print("cSWAG training")
    else:
        print("SWAG training")
    swag_model = SWAG(
        model_cfg.base,
        no_cov_mat=args.no_cov_mat,
        max_num_models=args.max_num_models,
        *model_cfg.args,
        num_classes=num_classes,
        **model_cfg.kwargs,
    )
    swag_model.to(args.device)
else:
    print("SGD training")


def schedule(epoch):
    print("updating lr")
    t = (epoch) / (args.swa_start if args.swa else args.epochs)
    lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01

    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor


# learning rate scheduler for cyclical stepsizes
def schedule_cyclical(epoch, batch_idx):
    datasize = 50000  # THIS IS ONLY CORRECT FOR CIFAR
    num_batch = datasize // args.batch_size
    rcounter = epoch * (datasize // args.batch_size) + batch_idx - 1
    T = args.epochs * num_batch  # total number iterations
    M = args.num_cycles  # number of cycles

    cos_inner = np.pi * (rcounter % (T // M))
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    lr = 0.5 * cos_out * init_lr
    if lr < args.min_cyc_lr:
        return args.min_cyc_lr
    return lr


def calc_weight(num_steps):
    res = []
    total_iter = num_steps  # how many steps are going to be included in the mean and cov calculations v
    for k_iter in range(total_iter):
        inner = (np.pi * k_iter) / total_iter
        cur_weight = np.cos(inner) + 1
        res.append(cur_weight)
    # res = res[::-1] # reversing to make start small and become larger
    res = torch.tensor(res)
    res = F.softmax(res)  # just to be safe
    return res


# use a slightly modified loss function that allows input of model
if args.loss == "CE":
    criterion = losses.cross_entropy
    # criterion = F.cross_entropy
elif args.loss == "adv_CE":
    criterion = losses.adversarial_cross_entropy


if args.sam:
    optimizer = SAM(
        model.parameters(),
        torch.optim.SGD,
        lr=args.lr_init,
        momentum=args.momentum,
        adaptive=False,
    )
else:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr_init,
        momentum=args.momentum,
        weight_decay=args.wd,
    )

start_epoch = 0
if args.resume is not None:
    print("Resume training from %s" % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

if args.swa and args.swa_resume is not None:
    checkpoint = torch.load(args.swa_resume)
    swag_model = SWAG(
        model_cfg.base,
        no_cov_mat=args.no_cov_mat,
        max_num_models=args.max_num_models,
        loading=True,
        *model_cfg.args,
        num_classes=num_classes,
        **model_cfg.kwargs,
    )
    swag_model.to(args.device)
    swag_model.load_state_dict(checkpoint["state_dict"])

columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time", "mem_usage"]
if args.swa:
    columns = columns[:-2] + ["swa_te_loss", "swa_te_acc"] + columns[-2:]
    swag_res = {"loss": None, "accuracy": None}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    state_dict=model.state_dict(),
    optimizer=optimizer.state_dict(),
)

sgd_ens_preds = None
sgd_targets = None
n_ensembled = 0.0
cur_weight_idx = 0
itr_per_cycle = args.epochs // args.num_cycles
cyc_num_weights = int(itr_per_cycle // 4)
cyc_weights = calc_weight(cyc_num_weights)
print(cyc_weights)
print(len(cyc_weights))
for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()
    itr_within_cycle = epoch % (args.epochs // args.num_cycles)
    lr_func = None
    if not args.no_schedule and not args.use_cyc and not args.use_cswag:
        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)

    # use noise for cyclical SWAG / MCMC
    use_noise = False
    if args.use_noise and itr_within_cycle + 1 > itr_per_cycle - args.num_cyc_samples:
        use_noise = True
    if args.use_cyc or args.use_cswag:
        lr_func = lambda batch_idx: schedule_cyclical(epoch, batch_idx)
        lr = lr_func(0)
    else:
        lr = args.lr_init

    if (args.swa and (epoch + 1) > args.swa_start) and args.cov_mat:
        train_res = utils.train_epoch(
            loaders["train"],
            model,
            criterion,
            optimizer,
            cuda=use_cuda,
            cyc_update_function=lr_func,
            use_sam=args.sam,
            use_noise=use_noise,
            epoch=epoch,
        )
    else:
        train_res = utils.train_epoch(
            loaders["train"],
            model,
            criterion,
            optimizer,
            cuda=use_cuda,
            cyc_update_function=lr_func,
            use_sam=args.sam,
            use_noise=use_noise,
            epoch=epoch,
        )

    if (
        epoch == 0
        or epoch % args.eval_freq == args.eval_freq - 1
        or epoch == args.epochs - 1
    ):
        test_res = utils.eval(loaders["test"], model, criterion, cuda=use_cuda)
    else:
        test_res = {"loss": None, "accuracy": None}

    if (
        args.swa
        and (epoch + 1) > args.swa_start
        and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0
        and not args.use_cyc
        and not args.use_cswag
    ):
        # sgd_preds, sgd_targets = utils.predictions(loaders["test"], model)
        sgd_res = utils.predict(loaders["test"], model)
        sgd_preds = sgd_res["predictions"]
        sgd_targets = sgd_res["targets"]
        print("updating sgd_ens")
        if sgd_ens_preds is None:
            sgd_ens_preds = sgd_preds.copy()
        else:
            # TODO: rewrite in a numerically stable way
            sgd_ens_preds = sgd_ens_preds * n_ensembled / (
                n_ensembled + 1
            ) + sgd_preds / (n_ensembled + 1)
        n_ensembled += 1
        swag_model.collect_model(model)
        if (
            epoch == 0
            or epoch % args.eval_freq == args.eval_freq - 1
            or epoch == args.epochs - 1
        ):
            swag_model.sample(0.0)
            utils.bn_update(loaders["train"], swag_model)
            swag_res = utils.eval(loaders["test"], swag_model, criterion)
        else:
            swag_res = {"loss": None, "accuracy": None}
    if args.use_cswag:
        print("Performing steps for cyclical variant")
        if itr_within_cycle >= itr_per_cycle - len(cyc_weights):
            print("collecting")

            cur_weight = cyc_weights[cur_weight_idx]
            if args.cyc_weights:
                swag_model.collect_model(model, cur_weight)
            else:
                swag_model.collect_model(model)

            cur_weight_idx += 1

        if itr_within_cycle == (args.epochs // args.num_cycles) - 1:
            cycle_num = epoch // (args.epochs // args.num_cycles)
            print(f"{cycle_num}")
            utils.save_checkpoint(
                args.dir,
                cycle_num,
                name="c_swag_cycle",
                state_dict=swag_model.state_dict(),
            )
            swag_model.wipe_clean()
            init_lr = args.lr_init
            cur_weight_idx = 0

    # if (
    #    epoch == 0
    #    or epoch % args.eval_freq == args.eval_freq - 1
    #    or epoch == args.epochs - 1
    # ):

    # TODO: making the ensemble prediction for the cyclical variant
    if args.use_cyc:
        if itr_within_cycle + 1 > itr_per_cycle - args.num_cyc_samples:
            sample_num = itr_within_cycle + 1 - itr_per_cycle + args.num_cyc_samples

            cycle_num = epoch // (args.epochs // args.num_cycles)
            print(sample_num)
            # for normal cSGMCMC, only need the base model state dict -- don't need the SWAG object
            utils.save_checkpoint(
                args.dir,
                sample_num,
                name=f"csgmcmc_cycle_{cycle_num}",
                state_dict=model.state_dict(),
            )

    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
        if args.swa:
            utils.save_checkpoint(
                args.dir, epoch + 1, name="swag", state_dict=swag_model.state_dict()
            )

    time_ep = time.time() - time_ep

    if use_cuda:
        memory_usage = torch.cuda.memory_allocated() / (1024.0**3)

    values = [
        epoch + 1,
        lr,
        train_res["loss"],
        train_res["accuracy"],
        test_res["loss"],
        test_res["accuracy"],
        time_ep,
        memory_usage,
    ]
    if args.swa:
        values = values[:-2] + [swag_res["loss"], swag_res["accuracy"]] + values[-2:]
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 40 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)

if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
    )
    if args.swa and args.epochs > args.swa_start:
        utils.save_checkpoint(
            args.dir, args.epochs, name="swag", state_dict=swag_model.state_dict()
        )

if args.swa:
    np.savez(
        os.path.join(args.dir, "sgd_ens_preds.npz"),
        predictions=sgd_ens_preds,
        targets=sgd_targets,
    )
