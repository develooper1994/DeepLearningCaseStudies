"""
The main idea behind that file, making clean, easy to modify pytorch model trainers, tensorboard loggers and
beautiful printing. And also saving network(.pt, .onnx) and results in different ways(.csv, .json, ...)
# Soruce: https://deeplizard.com/learn/video/ozpv_peZ894
### Source code: https://towardsdatascience.com/build-a-fashion-mnist-cnn-pytorch-style-efb297e22582
*** Note: if you want to run on google colab, you should use ngrok service to access tensorboard


Tensor Board is a TensorFlow visualization tool now also supported by PyTorch. We’ve already taken the efforts to export everything into the ‘./runs’ folder where Tensor Board will be looking into for records to consume. What we need to do now is just to launch the Tensor Board and check. Since I’m running this model on Google Colab, we’ll use a service called ngrok to proxy and access our Tensor Board running on Colab virtual machine. Install ngrok first:
!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip
Then, specify the folder we want to run Tensor Board from and launch the Tensor Board web interface (./runs is the default):
LOG_DIR = './runs'
get_ipython().system_raw(
'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
.format(LOG_DIR)
)
Launch ngrok proxy:
get_ipython().system_raw('./ngrok http 6006 &')
Generate an URL so we can access our Tensor Board from within the Jupyter Notebook:
! curl -s http://localhost:4040/api/tunnels | python3 -c \
"import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

"""
# imports

# pytorch imports
from typing import List, Any

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# tensorboard imports
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# standart module imports
from collections import OrderedDict
from collections import namedtuple
from itertools import product  # cartesian product
import time
import json

# classical machine learning imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from IPython.display import display, clear_output

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)  # On by default, leave it here for clarity


# default `log_dir` is "runs" - we'll be more specific here
# tb = SummaryWriter('runs/fashion_mnist_experiment_1')

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    # define forward function
    def forward(self, t):
        # conv 1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # conv 2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # fc1
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # fc2
        t = self.fc2(t)
        t = F.relu(t)

        # output
        t = self.out(t)
        # don't need softmax here since we'll use cross-entropy as activation.

        return t


class RunBuilder:
    """
    Takes all parameters with variable names and values and insert into a list
    """

    @staticmethod
    def get_runs(params) -> List[Any]:
        """
        Takes parameters and makes first it dictionary.
        Second appends all values or keys of variable in to a list.
        @param params: takes an OrderedDict to try all experiments in one loop and also to show results on any report.
        @return: list of values or keys of variable

        """
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


class Epoch:
    """
    Refactored epoch variables
    """
    count: int
    loss: int
    num_correct: int

    def __init__(self):
        self.count = 0
        self.loss = 0
        self.num_correct = 0
        self.start_time = None


class Run:
    """
    Refactored run variables
    """
    count: int
    data: List[Any]

    def __init__(self):
        self.params = None
        self.count = 0
        self.data = []
        self.start_time = None


class RunManager:
    """
    Controls all learning process functions
    """

    # TODO! make a evaluatin method
    def __init__(self):

        # tracking every epoch count, loss, accuracy, time
        self.epoch = Epoch()

        # tracking every run count, run data, hyper-params used, time
        self.run = Run()

        # record model, loader and TensorBoard
        self.network = None
        self.loader = None
        self.tb = None

    # record the count, hyper-param, model, loader of each run
    # record sample images and network graph to TensorBoard
    # TODO! refactor begin_run and end_run functions
    def begin_run(self, run, network, loader):
        """
        Configures and gives a start the one experiment.
        @param run: Represents the one experiment. Information comes from RunBuilder class.
        @param network: Pytorch Your neural network class
        @param loader: Pytorch dataloader.
        @return: None
        """

        self.run.start_time = time.time()

        self.run.params = run
        self.run.count += 1

        self.network = network
        self.loader = loader

        # one batch data
        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)

        # Tensorboard configuration
        self.tb = SummaryWriter(comment=f'-{run}')
        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images)

    # when run ends, close TensorBoard, zero epoch count
    def end_run(self):
        """
        Takes nothing
        Concludes one experiment and closes Tensorboard session.
        @return: None
        """
        self.tb.flush()  # not required
        self.tb.close()
        self.epoch.count = 0

    # zero epoch count, loss, accuracy,
    # TODO! refactor begin_epoch and end_epoch functions
    def begin_epoch(self):
        """
        Takes nothing
        Configures and gives a start the one epoch experiment.
        @return: None
        """
        self.epoch.start_time = time.time()

        self.epoch.count += 1
        self.epoch.loss = 0
        self.epoch.num_correct = 0

    #
    def end_epoch(self):
        """
        Takes nothing
        1) Measures taken time to complete.
        2) Calculates loss and accuracy for each epoch.
        3) Adds scalar plots to Tensorboard.
        4) Makes results and pandas frame and print in nice way.

        Concludes one experiment.
        @return: None
        """
        # calculate epoch duration and run duration(accumulate)
        epoch_duration = time.time() - self.epoch.start_time
        run_duration = time.time() - self.run.start_time

        # record epoch loss and accuracy
        loss = self.epoch.loss / len(self.loader.dataset)
        accuracy = self.epoch.num_correct / len(self.loader.dataset)

        # Record epoch loss and accuracy to TensorBoard
        self.tb.add_scalar('Loss', loss, self.epoch.count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch.count)

        # Record params to TensorBoard
        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch.count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch.count)

        # Write into 'results' (OrderedDict) for all run related data
        results = OrderedDict()
        results["run"] = self.run.count
        results["epoch"] = self.epoch.count
        results["loss"] = loss
        results["accuracy"] = accuracy
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration

        # Record hyper-params into 'results'
        for k, v in self.run.params._asdict().items(): results[k] = v
        self.run.data.append(results)
        df: DataFrame = pd.DataFrame.from_dict(self.run.data, orient='columns')

        # display epoch information and show progress
        clear_output(wait=True)
        display(df)

    # accumulate loss of batch into entire epoch loss
    @torch.no_grad()
    def track_loss(self, loss):
        """
        Tracks loss function for loss for each epoch
        @param loss: loss function
        @return: None
        """
        # multiply batch size so variety of batch sizes can be compared
        self.epoch.loss += loss.item() * self.loader.batch_size

    # accumulate number of corrects of batch into entire epoch num_correct
    @torch.no_grad()
    def track_num_correct(self, preds, labels):
        """
        Takes predictions and labels
        Tracks num correct predictions with respect to actual labels for each epoch
        @param preds: correct predictions
        @type preds: torch.tensor
        @param labels: actual labels
        @return: None
        """
        self.epoch.num_correct += self._get_num_correct(preds, labels)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        """
        Takes predictions and labels
        Counts num correct predictions with respect to actual labels for each epoch
        @param preds: correct predictions
        @type preds: torch.tensor
        @param labels: actual labels
        @return: number of corrections
        """
        return preds.argmax(dim=1).eq(labels).sum().item()

    # save end results of all runs into csv, json for further analysis
    def save(self, filename):
        """
        Creates .csv file with pandas and a .json dump file with json.
        @param filename: a filename string
        @return: None
        """

        pd.DataFrame.from_dict(
            self.run.data,
            orient='columns',
        ).to_csv(f'{filename}.csv')

        with open(f'{filename}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run.data, f, ensure_ascii=False, indent=4)


def main():
    """
    Put all to gather
    @return: None
    """
    # put all hyper params into a OrderedDict, easily expandable
    params = OrderedDict(
        lr=[.01, .001],
        batch_size=[100, 1000],
        shuffle=[True, False]
    )
    dataset_path = '~/.pytorch'  # './data/FashionMNIST'
    train_set = torchvision.datasets.FashionMNIST(
        root=dataset_path,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    epochs = 1
    m = RunManager()
    # get all runs from params using RunBuilder class
    runs = RunBuilder.get_runs(params)
    print(f"number of experiments: {runs * epochs}")

    # if params changes, following line of code should reflect the changes too
    network = Network()
    for run in runs:

        # if params changes, following line of code should reflect the changes too
        # network = Network() # !!!if network hyperparameters changes each time then it should inside of the loop!!!
        loader = torch.utils.data.DataLoader(train_set, batch_size=run.batch_size, num_workers=1)  # num_workers=1 is
        # good for small data
        optimizer = optim.Adam(network.parameters(), lr=run.lr)

        m.begin_run(run, network, loader)
        for epoch in range(epochs):
            m.begin_epoch()

            # one batch
            for batch in loader:
                images, labels = batch[0], batch[1]
                preds = network(images)
                optimizer.zero_grad()  # clear gradient accumulator

                # loss and gradient
                loss = F.cross_entropy(preds, labels)
                loss.backward()
                optimizer.step()

                m.track_loss(loss)
                m.track_num_correct(preds, labels)

            m.end_epoch()
        m.end_run()

    # when all runs are done, save results to files
    m.save('results')
    print("End of all training experiments")


if __name__ == '__main__':
    main()
