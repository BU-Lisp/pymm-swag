# SYSTEM IMPORTS
from typing import List
from tqdm import tqdm
import argparse
import numpy as np
import pymm
import os
import sys
import torch as pt
import torch.nn.functional as F
import torchvision as ptv


_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from src.posterior import Posterior
from src.pymmposterior import PymmPosterior


from mnist_models import Model_1_5Gb, Model_50Gb, Model_113Gb, Model_317Gb, Model_5Tb


num_times_params_sent: int = 0
model_map = {m.__name__.lower(): m for m in [Model_1_5Gb, Model_50Gb,
                                             Model_113Gb, Model_317Gb,
                                             Model_5Tb]}


def posterior_func(child_pipe: mp.Pipe, child_progress: mp.Queue,
                   args, num_params: int) -> None:
    shelf = pymm.shelf("mnist_pymm_posterior",
                       size_mb=args.size_mb,
                       pmem_path=args.shelf_file,
                       force_new=True)

    posterior = PymmPosterior(num_params, shelf)

    stop: bool = False
    count: int = 0
    while not stop:
        params_maybe = child_pipe.recv()
        stop = params_maybe is None

        if not stop:
            count += 1
            posterior.update(params_maybe)
            child_progress.put(count)

    child_progress.put(None)
    child_pipe.close()

def train_one_epoch(m: pt.nn.Module,
                    optim: pt.optim.Optimizer,
                    loader: pt.utils.data.DataLoader,
                    epoch: int,
                    parent_pipe: mp.Pipe,
                    batch_posterior: int,
                    cuda: int) -> None:

    global num_times_params_sent
    i = 1
    for batch_idx, (X, Y_gt) in tqdm(enumerate(loader),
                                     desc="training epoch %s" % epoch,
                                     total=len(loader)):
        optim.zero_grad()

        Y_hat: pt.Tensor = m.forward(X.to(cuda))
        loss: pt.Tensor = F.nll_loss(Y_hat.cpu(), Y_gt.cpu())
        loss.backward()
        optim.step()
        if ( i%batch_posterior == 0):
            parent_pipe.send(m.get_params())
            num_times_params_sent += 1
        i = i + 1


def main() -> None:
    global model_map, num_times_params_sent

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, choices=sorted(model_map.keys()),
                        help="the model type")
    parser.add_argument("-b", "--batch_size", type=int, default=100,
                        help="batch size")
    parser.add_argument("-n", "--learning_rate", type=float,
                        default=1e-4,
                        help="learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0,
                        help="sgd momentum")
    parser.add_argument("-e", "--epochs", type=int, default=int(1e6),
                        help="num epochs")
    parser.add_argument("-p", "--path", type=str,
                        default="/scratch/aewood/data/mnist")
    parser.add_argument("-c", "--cuda", type=int,
                        default=0)
    parser.add_argument("-s", "--size_mb", type=int, default=40000,
                        help="size of shelf in mb")
    parser.add_argument("-f", "--shelf_file", type=str,
                        default="/mnt/pmem0",
                        help="pymm shelf directory")
    parser.add_argument("-r", "--bpost", type=int,
                        default=1)
    args = parser.parse_args()

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    train_loader = pt.utils.data.DataLoader(
        ptv.datasets.MNIST(args.path, train=True, download=True,
                           transform=ptv.transforms.ToTensor()
                           # transform=ptv.transforms.Compose([
                           #      ptv.transforms.ToTensor(),
                           #      ptv.transforms.Normalize((0.1307,),
                           #                               (0.3081,))
                           #  ])
                          ),
        batch_size=args.batch_size,
        shuffle=True)

    test_loader = pt.utils.data.DataLoader(
        ptv.datasets.MNIST(args.path, train=False, download=True,
                           transform=ptv.transforms.ToTensor()
                            # transform=ptv.transforms.Compose([
                            #     ptv.transforms.ToTensor(),
                            #     ptv.transforms.Normalize((0.1307,),
                            #                              (0.3081,))
                            # ])
                          ),
        batch_size=args.batch_size,
        shuffle=True)

    print("loading model")

    m = model_map[args.model]().to(args.cuda)
    optimizer = pt.optim.SGD(m.parameters(), lr=args.learning_rate,
                             momentum=args.momentum)

    num_params: int = m.get_params().shape[0]
    print("num params: %s" % num_params)

    parent_pipe, child_pipe = mp.Pipe(duplex=True)
    child_progress: mp.Queue = mp.Queue()
    posterior_process: mp.Process = mp.Process(target=posterior_func,
                                               args=(child_pipe, child_progress,
                                                     args, num_params))
    posterior_process.start()

    # update posterior with first parameter sample
    parent_pipe.send(m.get_params())
    num_times_params_sent += 1

    for e in range(args.epochs):
        train_one_epoch(m,
                        optimizer,
                        train_loader,
                        e+1,
                        parent_pipe,
                        args.bpost
                        args.cuda)

    """"""
    parent_pipe.send(None)
    print("waiting for posterior to finish...")
    with tqdm(total=num_times_params_sent, desc="posterior progress") as pbar:
        stop: bool = False
        while not stop:
            count_maybe = child_progress.get()
            stop = count_maybe is None
            if not stop:
                pbar.update(1)

    posterior_process.join()
    """"""


if __name__ == "__main__":
    main()

