import os
import json
import torch

torch.manual_seed(0)
import networkx as nx
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import sys

sys.path.append("../")
from nn_homology.nn_graph import activation_graph, parameter_graph
import matplotlib.pyplot as plt
import re
import torchvision

import torch.nn.functional as F
import glob
import tqdm.autonotebook as tqdm
import pickle
import numpy as np

np.random.seed(0)
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler


def inverse_abs(x):
    return np.abs(1 / x)


def inverse_abs_zero(x):
    return np.abs(1 / (1 + x))


def identity(x):
    return x


WEIGHT_TRANSFORMS = {
    "identity": identity,
    "inverse_abs_zero": inverse_abs_zero,
    "inverse_abs": inverse_abs,
}


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {
            label: np.where(self.labels.numpy() == label)[0]
            for label in self.labels_set
        }
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(
                    self.label_to_indices[class_][
                        self.used_label_indices_count[
                            class_
                        ] : self.used_label_indices_count[class_]
                        + self.n_samples
                    ]
                )
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(
                    self.label_to_indices[class_]
                ):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size


"""
Please Note: I have modified this base net very slightly
so that it would work with the activation-based graph provided
in the nn_homology repo. 

Base Net w/ Param Info
Note, that dropout net can be directly loaded into this
b/c we are not using dr for evaluation anyway. 
"""


class CIFAR_Net(nn.Module):
    def __init__(self):
        super(CIFAR_Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.features = [
            self.conv1,
            self.pool1,
            self.conv2,
            self.pool2,
            self.fc1,
            self.fc2,
        ]
        self.classifier = [self.fc3]

        self.param_info = [
            {
                "layer_type": "Conv2d",
                "kernel_size": (5, 5),
                "stride": 1,
                "padding": 0,
                "name": "Conv1",
            },
            {
                "layer_type": "MaxPool2d",
                "kernel_size": (2, 2),
                "stride": 2,
                "padding": 0,
                "name": "MaxPool1",
            },
            {
                "layer_type": "Conv2d",
                "kernel_size": (5, 5),
                "stride": 1,
                "padding": 0,
                "name": "Conv2",
            },
            {
                "layer_type": "MaxPool2d",
                "kernel_size": (2, 2),
                "stride": 2,
                "padding": 0,
                "name": "MaxPool2",
            },
            {"layer_type": "Linear", "name": "Linear1"},
            {"layer_type": "Linear", "name": "Linear2"},
            {"layer_type": "Linear", "name": "Linear3"},
        ]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_acts(self, x):
        intermediates = []
        intermediates.append(x)
        x = self.conv1(x)
        x = F.relu(x)
        intermediates.append(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        intermediates.append(x)
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        intermediates.append(x)
        x = F.relu(self.fc2(x))
        intermediates.append(x)
        x = self.fc3(x)
        intermediates.append(x)
        return x, intermediates


"""
Extract the data needed for the activation graph
"""


def get_data(batch_size=100, single_class=-1):
    # get data
    CIFAR_DATA_PATH = "/z/pujat/NN_Dynamics/src/data"
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    testset = torchvision.datasets.CIFAR10(
        root=CIFAR_DATA_PATH, train=False, download=False, transform=transform
    )
    balanced_batch_sampler = BalancedBatchSampler(testset, 10, batch_size)

    testloader = torch.utils.data.DataLoader(
        testset, shuffle=False, num_workers=2, batch_sampler=balanced_batch_sampler
    )
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # init load + sample
    sample_data = next(iter(testloader))

    if single_class > -1:
        # pull out only that classe's values
        tmp = torch.ones_like(sample_data[1]) * single_class
        sample_data = sample_data[0][sample_data[1] == tmp]
        print("=> SELECTING SINGLE CLASS! -- {}".format(single_class))
        print("=> Sample Shape: ", sample_data.shape)
    else:
        sample_data = sample_data[0]
    return sample_data


def convert_model_to_activation_network(
    net, ckpt_files, batch_size, save_path="../data/sample_experiment/", single_class=-1
):
    samples = get_data(batch_size, single_class)

    if single_class > -1:
        try:
            # nx_graph
            save_nx = os.path.join(
                save_path, "nx_graphs_activations_{}".format(single_class)
            )
            os.mkdir(save_nx)
        except:
            print("nx_graphs -- {} -- already exists".format(str(save_nx)))
        try:
            save_edge_list = os.path.join(
                save_path, "edge_list_activations_{}".format(single_class)
            )
            os.mkdir(save_edge_list)
        except:
            print("edge_list -- {} -- already exists".format(str(save_edge_list)))

    else:
        try:
            # nx_graph
            save_nx = os.path.join(save_path, "nx_graphs_activations")
            os.mkdir(save_nx)
        except:
            print("nx_graphs -- {} -- already exists".format(str(save_nx)))
        try:
            save_edge_list = os.path.join(save_path, "edge_list_activations")
            os.mkdir(save_edge_list)
        except:
            print("edge_list -- {} -- already exists".format(str(save_edge_list)))

    for e_num, file in tqdm.tqdm(enumerate(ckpt_files)):

        print("Processing: ", file)

        # load the model
        state_dict = torch.load(file)
        net.load_state_dict(state_dict)
        net.eval()

        G = activation_graph(
            net,
            net.param_info,
            samples,
            ignore_zeros=False,
            verbose=False,
            weight_func=lambda x: np.abs(x),
        )

        ckpt_name = Path(file).stem
        print("=> Checkpoint Name: {}".format(ckpt_name))

        graph_save_path = os.path.join(save_nx, "{}_ph_nx.pkl".format(ckpt_name))
        with open(graph_save_path, "wb") as file:
            pickle.dump(G, file)

        edge_save_path = os.path.join(save_edge_list, "{}.txt".format(str(e_num)))
        with open(edge_save_path, "wb") as file:
            nx.set_edge_attributes(G, e_num, "time")  # give edge the same time point
            nx.write_edgelist(G, file, data=["weight", "time"], delimiter=",")

    return


# save a dictionary of the accuracies.
def get_classwise_accuracy(net, ckpt_file, save_path="../data/sample_experiment/"):

    # get data
    CIFAR_DATA_PATH = "/z/pujat/NN_Dynamics/src/data"
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    testset = torchvision.datasets.CIFAR10(
        root=CIFAR_DATA_PATH, train=False, download=False, transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testset, shuffle=False, num_workers=2, batch_size=100
    )
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # load the model
    state_dict = torch.load(ckpt_file)
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    # init load + sample
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    total = 0
    correct = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            c = (predicted == labels).squeeze()
            correct += c.sum().item()
            for i in range(100):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    accs = {}
    for i in range(10):
        accs[classes[i]] = 100 * class_correct[i] / class_total[i]
        print(
            "Accuracy of %5s : %2d %%"
            % (classes[i], 100 * class_correct[i] / class_total[i])
        )
    accs["all"] = 100 * correct / total
    print("Accuracy of all : %2d %%" % (accs["all"]))

    # save class-wise accuracy.
    json_path = os.path.join(save_path, "class_acc.json")
    with open(json_path, "w") as f:
        json.dump(accs, f, indent=2)


def make_n2b_graph(
    net, input_shape=(1, 3, 32, 32), kernel_size=25, verbose=False, args=None
):
    weight_list = [
        p.weight.data
        for n, p in net._modules.items()
        if isinstance(p, nn.modules.conv.Conv2d)
        or isinstance(p, nn.modules.linear.Linear)
    ]

    if args.activation_graph:
        samples = get_data(128, -1)  # not a single class.
        _, attribute_info = net.forward_acts(samples)

    else:
        attribute_info = [
            p.weight.data
            for n, p in net._modules.items()
            if isinstance(p, nn.modules.conv.Conv2d)
        ] + [
            p.bias.data
            for n, p in net._modules.items()
            if isinstance(p, nn.modules.linear.Linear)
        ]

    node_count = 0
    g = nx.Graph()

    # add input nodes
    node_ids_old = list(range(input_shape[1]))
    g.add_nodes_from(node_ids_old)
    node_count += input_shape[1]
    # add the nodes from consecutive layers

    # we need to keep the preceding node ids
    # so that we form the bipartite graph correctly
    count = 0
    for idx in weight_list:
        count += 1
        node_idx = idx.shape[0]
        if verbose:
            print("=> Layer {} has {} nodes".format(count, node_idx))

        node_ids_new = np.arange(node_idx) + node_count
        g.add_nodes_from(list(node_ids_new))

        # generate edge weights
        if verbose:
            print("=> Node Count: {}".format(node_count))
        # compute the edge weights

        # its a conv layer
        if len(idx.shape) == 4:
            edge_weights = torch.norm(idx, p=2, dim=(2, 3))
            # for each nodes, for each channel, form a conenction
            [
                g.add_edge(
                    node_ids_new[i], node_ids_old[j], weight=edge_weights[i, j].item()
                )
                for i in range(edge_weights.shape[0])
                for j in range(edge_weights.shape[1])
            ]

        # it is a linear layer
        elif len(idx.shape) == 2:
            # no need to take norm directly available
            edge_weights = torch.abs(idx)
            try:
                [
                    g.add_edge(
                        node_ids_new[i],
                        node_ids_old[j],
                        weight=edge_weights[i, j].item(),
                    )
                    for i in range(edge_weights.shape[0])
                    for j in range(edge_weights.shape[1])
                ]
            except:
                # reshape to recover the previous number of channels
                edge_weights = torch.norm(
                    idx.reshape(idx.shape[0], 16, -1), p=2, dim=(2)
                )

                [
                    g.add_edge(
                        node_ids_new[i],
                        node_ids_old[j],
                        weight=edge_weights[i, j].item(),
                    )
                    for i in range(edge_weights.shape[0])
                    for j in range(edge_weights.shape[1])
                ]

        else:
            print("=> Warning! Unsuspected Layer Type")
        # update the node values
        node_ids_old = node_ids_new
        node_count += node_idx

    # add the last linear weight

    # make the node attributes
    # the plus one is for the node idx.
    node_count = 0
    print("Number of Nodes: ", g.number_of_nodes())
    if not args.activation_graph:
        node_attrs = np.zeros((g.number_of_nodes(), kernel_size + 1))
        node_count = input_shape[1]
        for idx in attribute_info:
            node_idx = idx.shape[0]
            if len(idx.shape) == 4:
                node_attrs[node_count : node_count + node_idx, 1:] = (
                    torch.norm(idx, p=2, dim=(1)).view(-1, kernel_size).numpy()
                )
            # use bias as the node attributes.
            elif len(idx.shape) == 1:
                node_attrs[node_count : node_count + node_idx, 1:] = np.repeat(
                    idx, kernel_size, axis=0
                ).reshape(-1, kernel_size)
            node_count += node_idx
        node_attrs[:, 0] = np.arange(node_count)

    if args.activation_graph:

        batch_shape, c, h, w = attribute_info[0].shape
        feature_size = h * w
        node_attrs = np.zeros((g.number_of_nodes(), feature_size + 1))
        node_count = 0
        for e, idx in enumerate(attribute_info):
            node_idx = idx.shape[1]
            if len(idx.shape) == 4:
                _, _, h_1, w_1 = idx.shape
                feature_1 = h_1 * w_1
                padding = feature_size - feature_1
                arr = torch.mean(idx, dim=0).view(idx.shape[1], -1).detach().numpy()
                if padding > 0:
                    node_attrs[node_count : node_count + node_idx, 1:] = np.pad(
                        arr, ((0, 0), (0, int(padding))), mode="wrap"
                    )
                else:
                    node_attrs[node_count : node_count + node_idx, 1:] = (
                        torch.mean(idx, dim=0).view(idx.shape[1], -1).detach().numpy()
                    )
            # use bias as the node attributes.
            elif len(idx.shape) == 2:
                node_attrs[node_count : node_count + node_idx, 1:] = np.repeat(
                    torch.mean(idx, dim=0).detach(), feature_size
                ).reshape(-1, feature_size)
            node_count += node_idx
        node_attrs[:, 0] = np.arange(node_count)

    return g, node_attrs


"""
Takes net and creates an attributed graph representation 
that will be used for node2bits. 
Future work will extend to include activation-based
feature representations. 
"""


def convert_to_n2b(net, ckpt_files, save_path="../data/sample_experiment/", args=None):

    try:
        # nx_graph

        if args.activation_graph:
            save_nx = os.path.join(save_path, "nx_graphs_acts_n2b")
        else:
            save_nx = os.path.join(save_path, "nx_graphs_n2b")
        os.mkdir(save_nx)
    except:
        print("nx_graphs_n2b -- {} -- already exists".format(str(save_nx)))
    try:
        if args.activation_graph:
            save_edge_list = os.path.join(save_path, "edge_list_acts_n2b")
        else:
            save_edge_list = os.path.join(save_path, "edge_list_n2b")
        os.mkdir(save_edge_list)
    except:
        print("edge_list -- {} -- already exists".format(str(save_edge_list)))
    try:
        if args.activation_graph:
            save_node_attrs = os.path.join(save_path, "node_acts_attrs")
        else:
            save_node_attrs = os.path.join(save_path, "node_attrs")
        os.mkdir(save_node_attrs)
    except:
        print("node_attrs -- {} -- already exists".format(str(save_node_attrs)))

    for e_num, file in tqdm.tqdm(enumerate(ckpt_files)):

        print("Processing: ", file)

        # load the model
        state_dict = torch.load(file)
        net.load_state_dict(state_dict)
        net.eval()

        G, node_attrs = make_n2b_graph(
            net, input_shape=(1, 3, 32, 32), kernel_size=25, verbose=False, args=args
        )

        ckpt_name = Path(file).stem
        print("=> Checkpoint Name: {}".format(ckpt_name))

        # numpy_save_path = os.path.join(save_np, "{}_ph_adj.npy".format(ckpt_name))
        # np.save(numpy_save_path,G_adj)

        graph_save_path = os.path.join(save_nx, "{}_ph_nx.pkl".format(ckpt_name))
        with open(graph_save_path, "wb") as file:
            pickle.dump(G, file)

        edge_save_path = os.path.join(save_edge_list, "{}.txt".format(str(e_num)))
        with open(edge_save_path, "wb") as file:
            nx.set_edge_attributes(G, e_num, "time")  # give edge the same time point
            nx.write_edgelist(G, file, data=["weight", "time"], delimiter=",")

        node_attr_save_path = os.path.join(save_node_attrs, "{}.npy".format(str(e_num)))
        np.save(node_attr_save_path, node_attrs)
    return


"""
Takes net and creates nx files + edge list
you need to pass the list of ckpt files
"""


def convert_model_to_network(
    net,
    ckpt_files,
    input_size=(1, 3, 32, 32),
    save_path="../data/sample_experiment/",
    args=None,
):

    weight_func = WEIGHT_TRANSFORMS[args.weight_transform]
    if args.activation_graph and not args.n2b:
        print("*********************************************")
        print("               ACTIVATION GRAPH              ")
        print("*********************************************")
        convert_model_to_activation_network(
            net,
            ckpt_files,
            batch_size=100,
            save_path=save_path,
            single_class=args.single_class,
        )
    elif args.n2b:
        print("*********************************************")
        print("                N2B  GRAPH                  ")
        print("*********************************************")
        convert_to_n2b(net, ckpt_files, save_path=save_path, args=args)

    else:
        try:
            save_np = os.path.join(save_path, "np_arrays")
            os.mkdir(save_np)
        except:
            print("np_arrays -- {}  already exists".format(str(save_np)))

        try:
            # nx_graph
            save_nx = os.path.join(save_path, "nx_graphs")
            os.mkdir(save_nx)
        except:
            print("nx_graphs -- {} -- already exists".format(str(save_nx)))
        try:
            save_edge_list = os.path.join(save_path, "edge_list")
            os.mkdir(save_edge_list)
        except:
            print("edge_list -- {} -- already exists".format(str(save_edge_list)))

        for e_num, file in tqdm.tqdm(enumerate(ckpt_files)):

            print("Processing: ", file)

            # load the model
            state_dict = torch.load(file)
            missing, unexpected = net.load_state_dict(state_dict, strict=False)
            print("Missing keys: ", missing)
            print("Unexpected keys: ", unexpected)
            net.eval()

            if not args.activation_graph:
                print("Prcessed")
                G = parameter_graph(
                    model=net,
                    param_info=net.param_info,
                    input_size=input_size,
                    weight_func=weight_func,
                    ignore_zeros=False,
                )

            ckpt_name = Path(file).stem
            print("=> Checkpoint Name: {}".format(ckpt_name))

            # numpy_save_path = os.path.join(save_np, "{}_ph_adj.npy".format(ckpt_name))
            # np.save(numpy_save_path,G_adj)

            graph_save_path = os.path.join(save_nx, "{}_ph_nx.pkl".format(ckpt_name))
            with open(graph_save_path, "wb") as file:
                pickle.dump(G, file)

            edge_save_path = os.path.join(save_edge_list, "{}.txt".format(str(e_num)))
            with open(edge_save_path, "wb") as file:
                nx.set_edge_attributes(
                    G, e_num, "time"
                )  # give edge the same time point
                nx.write_edgelist(G, file, data=["weight", "time"], delimiter=",")

    return

