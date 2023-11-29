import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import tqdm
import random
from swag import data, losses, models, utils
from swag.posteriors import SWAG, KFACLaplace, cSGMCMC, cSWAG
import itertools
from torchattacks import PGD, PGDL2


parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument("--file", type=str, default=None, required=True, help="checkpoint")

parser.add_argument(
    "--dataset", type=str, default="CIFAR10", help="dataset name (default: CIFAR10)"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/scratch/datasets/",
    metavar="PATH",
    help="path to datasets location (default: None)",
)
parser.add_argument(
    "--use_test",
    dest="use_test",
    action="store_true",
    help="use test dataset instead of validation (default: False)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size (default: 128)",
)
parser.add_argument("--split_classes", type=int, default=None)
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
    metavar="MODEL",
    help="model name (default: VGG16)",
)
parser.add_argument(
    "--method",
    type=str,
    default="SWAG",
    choices=[
        "csgmcmc",
        "cSWAG",
        "SWAG",
        "KFACLaplace",
        "SGD",
        "HomoNoise",
        "Dropout",
        "SWAGDrop",
    ],
    required=True,
)
parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    required=True,
    help="path to npz results file",
)
parser.add_argument("--samples_per_cycle", type=int, default=3)
parser.add_argument("--num_cycles", type=int, default=6)
parser.add_argument("--N", type=int, default=30)
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument(
    "--cov_mat", action="store_true", help="use sample covariance for swag"
)
parser.add_argument("--use_diag", action="store_true", help="use diag cov for swag")

parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument("--adv", action="store_true", help="evaluate adversarial metrics")
parser.add_argument("--adv_mode", type=str, default="l2", help="l2 or linf")
parser.add_argument("--adv_norm", type=float, default=0.5)
parser.add_argument("--adv_steps", type=int, default=10)


def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    nll = -np.sum(np.log(ps))
    return nll


args = parser.parse_args()

sample_with_cov = args.cov_mat and not args.use_diag
eps = 1e-12
if args.cov_mat:
    args.cov_mat = True
else:
    args.cov_mat = False

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
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
    shuffle_train=False,
)
"""if args.split_classes is not None:
    num_classes /= 2
    num_classes = int(num_classes)"""

print("Preparing model")
if args.method in ["SWAG", "HomoNoise", "SWAGDrop", "cSWAG"]:
    if args.method == "cSWAG":
        print("Loading cSWAG")
        mode_models = []
        for i in range(args.num_cycles):
            model = SWAG(
                model_cfg.base,
                no_cov_mat=not args.cov_mat,
                max_num_models=20,
                *model_cfg.args,
                num_classes=num_classes,
                **model_cfg.kwargs,
            )
            sd = torch.load(f"{args.file}_cycle-{i}.pt")
            model.load_state_dict(sd["state_dict"])
            mode_models.append(model)
        model = cSWAG(
            mode_models,
            cov=sample_with_cov,
            scale=args.scale,
            bn_update_fn=lambda m: utils.bn_update(loaders["train"], m),
        )
    else:
        model = SWAG(
            model_cfg.base,
            no_cov_mat=not args.cov_mat,
            max_num_models=20,
            *model_cfg.args,
            num_classes=num_classes,
            **model_cfg.kwargs,
        )


elif args.method in ["SGD", "Dropout", "KFACLaplace", "csgmcmc"]:
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
else:
    assert False
model.cuda()


def train_dropout(m):
    if type(m) == torch.nn.modules.dropout.Dropout:
        m.train()


if args.method == "csgmcmc":
    print("Loading csgmcmc")
    mode_models = []
    for i in range(args.num_cycles):
        tmp = []
        for j in range(args.samples_per_cycle):
            model_sd = f"{args.file}_cycle_{i}-{j+1}.pt"
            base_model = model_cfg.base(
                *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs
            )
            checkpoint = torch.load(model_sd)
            base_model.load_state_dict(checkpoint["state_dict"])
            tmp.append(base_model)
        mode_models.append(tmp)
    print(mode_models)
    c_sgmcmc = cSGMCMC(mode_models)
    args.N = 1
    model = c_sgmcmc.cuda()  # make it a fair comparison
elif args.method == "cSWAG":
    pass
else:
    print("Loading model %s" % args.file)
    checkpoint = torch.load(args.file)
    model.load_state_dict(checkpoint["state_dict"])

if args.method == "KFACLaplace":
    print(len(loaders["train"].dataset))
    model = KFACLaplace(
        model, eps=5e-4, data_size=len(loaders["train"].dataset)
    )  # eps: weight_decay

    t_input, t_target = next(iter(loaders["train"]))
    t_input, t_target = (
        t_input.cuda(non_blocking=True),
        t_target.cuda(non_blocking=True),
    )

if args.method == "HomoNoise":
    std = 0.01
    for module, name in model.params:
        mean = module.__getattr__("%s_mean" % name)
        module.__getattr__("%s_sq_mean" % name).copy_(mean**2 + std**2)


predictions = np.zeros((len(loaders["test"].dataset), num_classes))
targets = np.zeros(len(loaders["test"].dataset))
print(targets.size)
csgmcmc_sample_cycle_pairs = list(
    itertools.product(
        [i for i in range(args.num_cycles)], [i for i in range(args.samples_per_cycle)]
    )
)

if args.adv and "SWAG" in args.method:
    print("Precomputing adv attacks SWAG methods")
    adv_trainloader = []
    model.update_models()
    for input, target in tqdm.tqdm(loaders["test"]):
        input = input.cuda()
        target = target.cuda()
        if args.adv_mode == "linf":
            attacker = PGD(model, eps=args.adv_norm, steps=args.adv_steps)
        else:
            attacker = PGDL2(
                model,
                eps=args.adv_norm,
                alpha=(2 * args.adv_norm) / 2,
                steps=args.adv_steps,
            )
        eval_input = attacker(input, target)
        input = input.cpu()
        eval_input = eval_input.cpu()
        target = target.cpu()
        adv_trainloader.append((eval_input, target))
for i in range(args.N):
    print("%d/%d" % (i + 1, args.N))
    if args.method == "KFACLaplace":
        ## KFAC Laplace needs one forwards pass to load the KFAC model at the beginning
        model.net.load_state_dict(model.mean_state)

        if i == 0:
            model.net.train()

            loss, _ = losses.cross_entropy(model.net, t_input, t_target)
            loss.backward(create_graph=True)
            model.step(update_params=False)

    if args.method not in ["SGD", "Dropout", "csgmcmc", "cSWAG"]:
        model.sample(scale=args.scale, cov=sample_with_cov)

    if args.method == "cSWAG":
        model.update_models()

    if args.method == "SWAG":
        utils.bn_update(loaders["train"], model)
    model.eval()
    if args.method in ["Dropout", "SWAGDrop"]:
        model.apply(train_dropout)
        # torch.manual_seed(i)
        # utils.bn_update(loaders['train'], model)

    k = 0
    if args.adv and "SWAG" in args.method:
        pg = tqdm.tqdm(adv_trainloader)
    else:
        pg = tqdm.tqdm(loaders["test"])
    for input, target in pg:
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        ##TODO: is this needed?
        # if args.method == 'Dropout':
        #    model.apply(train_dropout)
        torch.manual_seed(i)
        if args.adv and "SWAG" not in args.method:
            if args.adv_mode == "linf":
                attacker = PGD(model, eps=args.adv_norm, steps=args.adv_steps)
            else:
                attacker = PGDL2(
                    model,
                    eps=args.adv_norm,
                    alpha=(2 * args.adv_norm) / 2,
                    steps=args.adv_steps,
                )
            print(model)
            eval_input = attacker(input, target)
        else:
            eval_input = input
        if args.method == "KFACLaplace":
            output = model.net(eval_input)
        else:
            output = model(eval_input)

        with torch.no_grad():
            predictions[k : k + input.size()[0]] += (
                F.softmax(output, dim=1).cpu().numpy()
            )
        targets[k : (k + target.size(0))] = target.cpu().numpy()
        k += input.size()[0]

    print("Accuracy:", np.mean(np.argmax(predictions, axis=1) == targets))
    # nll is sum over entire dataset
    print("NLL:", nll(predictions / (i + 1), targets))
predictions /= args.N

entropies = -np.sum(np.log(predictions + eps) * predictions, axis=1)
np.savez(args.save_path, entropies=entropies, predictions=predictions, targets=targets)
