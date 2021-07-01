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


class Model(pt.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = pt.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = pt.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = pt.nn.Dropout2d()
        self.fc1 = pt.nn.Linear(320, 50)
        self.fc2 = pt.nn.Linear(50, 10)

    def forward(self,
                x: pt.Tensor):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def get_params(self) -> np.ndarray:
        params_list: List[np.ndarray] = list()
        for P in self.parameters():
            params_list.append(P.cpu().detach().numpy().reshape(-1))
        return np.hstack(params_list).reshape(-1,1)

    def set_params(self,
                   theta: np.ndarray) -> None:
        param_idx: int = 0
        for P in self.parameters():
            if len(P.size() > 0):
                num_params: int = np.prod(P.size())
                P.copy_(theta[param_idx:param_idx+num_params]
                    .reshape(P.size()))
                param_idx += num_params


def train_one_epoch(m: pt.nn.Module,
                    optim: pt.optim.Optimizer,
                    loader: pt.utils.data.DataLoader,
                    epoch: int,
                    posterior: Posterior,
                    cuda: int) -> None:
    for batch_idx, (X, Y_gt) in tqdm(enumerate(loader),
                                     desc="training epoch %s" % epoch,
                                     total=len(loader)):
        optim.zero_grad()

        Y_hat: pt.Tensor = m.forward(X.to(cuda))
        loss: pt.Tensor = F.nll_loss(Y_hat, Y_gt)
        loss.backward()
        optim.step()

        posterior.update(m.get_params())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=100,
                        help="batch size")
    parser.add_argument("-n", "--learning_rate", type=float,
                        default=1e-4,
                        help="learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0,
                        help="sgd momentum"
    parser.add_argument("-e", "--epochs", type=int, default=int(1e6),
                        help="num epochs")
    parser.add_argument("-p", "--path", type=str,
                        default="/scratch/aewood/data/mnist")
    parser.add_argument("-c", "--cuda", type=int,
                        default=0)
    parser.add_argument("-s", "--size_mb", type=int, default=50000,
                        help="size of shelf in mb")
    parser.add_argument("-f", "--shelf_file", type=str,
                        default="/mnt/pmem0",
                        help="pymm shelf directory")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    train_loader = pt.utils.data.DataLoader(
        ptv.datasets.MNIST(args.path, train=True, download=True,
                            transform=ptv.transforms.Compose([
                                ptv.transforms.ToTensor(),
                                ptv.transforms.Normalize((0.1307,),
                                                         (0.3081,))
                            ])),
        batch_size=args.batch_size,
        shuffle=True)

    test_loader = pt.utils.data.DataLoader(
        ptv.datasets.MNIST(args.path, train=False, download=True,
                            transform=ptv.transforms.Compose([
                                ptv.transforms.ToTensor(),
                                ptv.transforms.Normalize((0.1307,),
                                                         (0.3081,))
                            ])),
        batch_size=args.batch_size,
        shuffle=True)

    shelf = pymm.shelf("mnist_pymm_posterior",
                       size_mb=args.size_mb,
                       pmem_path=args.shelf_file,
                       force_new=True)

    print("loading model")

    m = Model().to(args.cuda)
    optimizer = pt.optim.SGD(m.parameters(), lr=args.learning_rate,
                             momentum=args.momentum)

    num_params: int = m.get_params().shape[0]
    print("num params: %s" % num_params)
    posterior = DramPosterior(num_params, shelf)

    # update posterior with first parameter sample
    posterior.update(m.get_params())
    for e in range(args.epochs):
        train_one_epoch(m,
                        optimizer,
                        train_loader,
                        e+1,
                        posterior,
                        args.cuda)


if __name__ == "__main__":
    main()

