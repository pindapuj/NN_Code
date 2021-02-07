import argparse
import torch
from model import Net, get_cifar_data, Dropout_Net, BatchNorm_Net
from train import make_experiment


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def arg_parser():

    # add command line arguments for easy training
    parser = argparse.ArgumentParser(description="Training Driver")
    parser.add_argument(
        "--output_path",
        type=str,
        help="Where the statedict should be saved",
        default="runs/",
    )
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.001)
    parser.add_argument("--dr", type=float, help="Dropout", default=0.0)
    parser.add_argument("--num_epochs", type=int, help="Number of Epochs", default=200)
    parser.add_argument("--batch_size", type=int, help="Batchsize", default=128)
    parser.add_argument("--save_freq", type=int, help="How often to save", default=100)
    parser.add_argument("--momentum", type=float, help="Use momemtum", default=0.9)
    parser.add_argument(
        "--use_adam", type=str2bool, help="Use ADAM vs. SGD", default=False
    )
    parser.add_argument(
        "--save_by_minibatch",
        type=str2bool,
        help="Whether to save every x minibatches",
        default=False,
    )
    parser.add_argument(
        "--augment_data",
        type=str2bool,
        default=False,
        help="Whether to include augmentation",
    )
    parser.add_argument(
        "--use_bn", type=str2bool, default=False, help="Use BN Layer or not"
    )
    parser.add_argument("--max_saved", type=int, default=False, help="Max Number Saved")
    parser.add_argument("--rep_num", type=int, default=0, help="Replicate Number")
    args = parser.parse_args()

    print("=> Output path: {}".format(args.output_path))
    print("=> lr: {}".format(args.lr))
    print("=> dr: {}".format(args.dr))
    print("=> num_epochs: {}".format(args.num_epochs))
    print("=> batch_size: {}".format(args.batch_size))
    print("=> save_freq: {}".format(args.save_freq))
    print("=> save_by_minibatch: {}".format(args.save_by_minibatch))
    print("=> augment_data: {}".format(args.augment_data))
    print("=> use_bn: {}".format(args.use_bn))
    return args


def main():
    args = arg_parser()

    # make the net
    if args.use_bn:  # make batch_norm net
        print("=> Using BN Net!")
        net = BatchNorm_Net()

    elif args.dr > 0.0:  # make dropout net
        print("=> Using DR Net!")
        net = Dropout_Net(p=args.dr)
    else:  # make a normal net
        net = Net()

    # load data
    trainloader, testloader, _ = get_cifar_data(
        augment_data=args.augment_data, batch_size=args.batch_size
    )

    # make experiment
    make_experiment(net, args, trainloader, testloader)

    print("*****************************************")
    print("                 DONE                    ")
    print("*****************************************")


if __name__ == "__main__":
    main()
