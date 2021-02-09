import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import copy
import os
import json
from model import EarlyStopping


def save_configs(args, exp_name, acc, early_stop_num):
    json_path = os.path.join(exp_name, "configs.json")
    with open(json_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
        f.write("\n Acc: {}".format(acc))
        if early_stop_num > 0:
            f.write("\n Early Stopping: {}".format(early_stop_num))


def make_save_name(args):
    name = "exp_lr_{lr}_momentum_{mo}_dr_{dr}_bs_{bs}_ne_{epoch}_sf_{sf}".format(
        lr=str(args.lr).replace("0.", ""),
        mo=str(args.momentum).replace("0.", ""),
        dr=str(args.dr).replace("0.", ""),
        bs=args.batch_size,
        epoch=args.num_epochs,
        sf=args.save_freq,
    )

    # now the boolean values
    if args.use_adam:
        name = name + "_adam_"
    else:
        name = name + "_sgd_"
    if args.augment_data:
        name = name + "AUGDATA_"
    if args.save_by_minibatch:
        name = name + "sbmb_"
    else:
        name = name + "sbe_"
    name = name + str(args.rep_num)
    print("=> Experiment Name: {}".format(name))
    exp_name = os.path.join(args.output_path, name)
    print("=> Experiment Directory: {}".format(exp_name))

    try:
        os.makedirs(str(exp_name))
        print("=> Saving path: ", exp_name)
    except OSError as e:
        print("=> {} already exists".format(exp_name))
    return exp_name


def make_experiment(net, args, trainloader, testloader):
    exp_name = make_save_name(args)
    criterion = nn.CrossEntropyLoss()
    es = EarlyStopping(patience=5)
    if args.use_adam:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    writer = SummaryWriter(
        comment="_LR_{}_M_{}_EPOCH_{}_SAVEFREQ_{}".format(
            args.lr, args.momentum, args.num_epochs, args.save_freq
        )
    )
    losses = []
    accuracy = []
    acc = 0
    early_stop_num = -1
    num_saved_counter = 0  # we will only save at most num_saved
    for epoch in range(args.num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                writer.add_scalar(
                    "training loss", loss.item(), epoch * len(trainloader) + i
                )
                running_loss = 0.0

                # get accuracy 2k minibatches
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                acc = 100 * correct / total
                writer.add_scalar("test accuracy", acc, epoch * len(trainloader) + i)

                # get histograms!
                for j in net.state_dict().keys():
                    name_j = j + "_dist"
                    writer.add_histogram(
                        name_j,
                        net.state_dict()[j].numpy().flatten(),
                        epoch * len(trainloader) + i,
                    )

            # save every x minibatches or save every x epochs
            if (
                args.save_by_minibatch
                and i % args.save_freq == 0
                and num_saved_counter < args.max_saved
            ):
                checkpoint = {
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "acc": acc,
                }
                num_saved_counter += 1

                if not os.path.exists(exp_name):
                    os.mkdir(exp_name)

                file_name = str(exp_name) + "/checkout_{}_{:03d}.pth".format(epoch, i)
                torch.save(net.state_dict(), file_name)

                if i == 0:
                    print("File name: {}".format(file_name))
            losses.append(loss)
        if i % args.save_freq == 0 and num_saved_counter < args.max_saved:
            checkpoint = {
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "acc": acc,
            }
            num_saved_counter += 1

            file_name = str(exp_name) + "/checkout_{}_{:03d}.pth".format(epoch, i)
            torch.save(checkpoint, file_name)

            if i == 0:
                print("File name: {}".format(file_name))

        print("Epoch num: {}".format(epoch))
        # get accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        print("Accuracy of the network on the 10000 test images: %d %%" % (epoch_acc))
        # PT: 6/22/2020, burnin period of 10, not used in prior experiments
        if es.step(torch.tensor(epoch_acc)) and epoch > 10:
            early_stop_num = epoch
            break

    save_configs(args, exp_name, epoch_acc, early_stop_num)

    checkpoint = {
        "state_dict": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "acc": acc,
    }
    file_name = str(exp_name) + "/FINAL_{}_{:03d}.pth".format(epoch, i)
    torch.save(net.state_dict(), file_name)
    print("File name: {}".format(file_name))
    return epoch_acc
