import argparse
import glob
import re

from gen_graph import CIFAR_Net, Alex_Net, VGG_Net, convert_model_to_network


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    """Turn a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split("([0-9]+)", s)]


def sort_nicely(l):
    """Sort the given list in the way that humans expect."""
    l.sort(key=alphanum_key)
    return l


def sort_epoch(l):
    ll = []
    for item in l:
        if "_390" in item:
            ll.append(item)

    return ll


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
    parser = argparse.ArgumentParser(description="GenGraph Driver")
    # note the default output_path is a new folder in the input paths
    parser.add_argument("--output_path", type=str, help="Where the nx_graphs should be saved", default=None)
    # if glob, then assume we are in the exp folder doing multiple exp
    parser.add_argument("--glob", type=str2bool, help="Read in ckpts the directories", default=False)
    parser.add_argument("--input_path", type=str, help="Specify Input directory", default="../GenData/runs")
    # default is to convert all in the directory
    parser.add_argument("--activation_graph", type=str2bool,
                        help="Whether to generate an activation or a weight-based graph", default=False)
    parser.add_argument("--num_convert", type=int, help="Number of ckpt to convert", default=10)
    parser.add_argument("--single_class", type=int, help="Only pull out samples from a single class", default=-1)
    parser.add_argument("--n2b", type=str2bool, default=False, help="Generate n2b attributed representation")
    parser.add_argument("--weight_transform", type=str, default="inverse_abs_zero", help="Edge weight transform")
    parser.add_argument("--dr", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--net", type=str, default="vgg", help="which net")
    args = parser.parse_args()

    print("=> Output Path: {}".format(args.output_path))
    print("=> Input Path: {}".format(args.input_path))
    print("=> Num Convert: {}".format(args.num_convert))
    print("=> Activation Graph?: {}".format(args.activation_graph))
    print("=> N2B Graph?: {}".format(args.n2b))
    print("=> Single Class Only?: {}".format(args.single_class))
    print("=> Edge weight transform: {}".format(args.weight_transform))
    print("=> is glob: {}".format(args.glob))
    print("=> dropout rate: {}".format(args.dr))
    print("=> which net: {}".format(args.net))
    return args


def main():
    args = arg_parser()
    dir_list = ["/cluster/home/it_stu157/iris/new2/exp_lr_2_momentum_9_dr_0_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_2_momentum_9_dr_1_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_2_momentum_9_dr_2_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_2_momentum_9_dr_3_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_2_momentum_9_dr_4_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_2_momentum_9_dr_5_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_2_momentum_9_dr_6_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_2_momentum_9_dr_7_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_2_momentum_9_dr_8_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_2_momentum_9_dr_9_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_2.5e-06_momentum_9_dr_0_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_2.5e-06_momentum_9_dr_1_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_2.5e-06_momentum_9_dr_2_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_2.5e-06_momentum_9_dr_3_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_2.5e-06_momentum_9_dr_4_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_2.5e-06_momentum_9_dr_5_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_2.5e-06_momentum_9_dr_6_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_2.5e-06_momentum_9_dr_7_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_2.5e-06_momentum_9_dr_8_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_2.5e-06_momentum_9_dr_9_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_25_momentum_9_dr_0_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_25_momentum_9_dr_1_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_25_momentum_9_dr_2_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_25_momentum_9_dr_3_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_25_momentum_9_dr_4_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_25_momentum_9_dr_5_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_25_momentum_9_dr_6_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_25_momentum_9_dr_7_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_25_momentum_9_dr_8_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu157/iris/new2/exp_lr_25_momentum_9_dr_9_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu187/iris/exp3/exp_lr_35_momentum_9_dr_0_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu187/iris/exp3/exp_lr_35_momentum_9_dr_1_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu187/iris/exp3/exp_lr_35_momentum_9_dr_2_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu187/iris/exp3/exp_lr_35_momentum_9_dr_3_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu187/iris/exp3/exp_lr_35_momentum_9_dr_4_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu187/iris/exp3/exp_lr_35_momentum_9_dr_5_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu187/iris/new1/exp_lr_15_momentum_9_dr_0_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu187/iris/new1/exp_lr_15_momentum_9_dr_1_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu187/iris/new1/exp_lr_15_momentum_9_dr_2_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu187/iris/new1/exp_lr_15_momentum_9_dr_3_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu187/iris/new1/exp_lr_15_momentum_9_dr_4_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu187/iris/new1/exp_lr_15_momentum_9_dr_5_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu187/iris/new1/exp_lr_15_momentum_9_dr_6_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu187/iris/new1/exp_lr_15_momentum_9_dr_7_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu187/iris/new1/exp_lr_15_momentum_9_dr_8_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu187/iris/new1/exp_lr_15_momentum_9_dr_9_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu187/iris/exp3/exp_lr_35_momentum_9_dr_6_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu187/iris/exp3/exp_lr_35_momentum_9_dr_7_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu187/iris/exp3/exp_lr_35_momentum_9_dr_8_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0",
                "/cluster/home/it_stu187/iris/exp3/exp_lr_35_momentum_9_dr_9_bs_128_ne_50_sf_130_sgd_AUGDATA_sbmb_0"
                ]

    # are we globbing the directories?
    if args.glob:
        glob_path = args.input_path + "/exp*"
        all_exp_names = sort_nicely(glob.glob(glob_path))
        # now we need to iterate over all the exps
        for exp in all_exp_names:
            if exp not in dir_list:
                print("*" * 20)
                print(exp)
                print("*" * 20)
                exp_glob_path = exp + "/*390.pth"
                all_ckpt_files = sort_nicely(glob.glob(exp_glob_path))
                # all_ckpt_files = sort_epoch(glob.glob(exp_glob_path))

                if args.num_convert > 0:
                    all_ckpt_files = all_ckpt_files[1: args.num_convert + 1]
                    print("=> Num Convert: {}".format(args.num_convert))
                    print("=> Num. Ckpt Files: {}".format(len(all_ckpt_files)))
                    print("=> First. Ckpt Files: {}".format(all_ckpt_files[0]))
                    print("=> Second. Ckpt Files: {}".format(all_ckpt_files[1]))
                else:
                    print("=> All Ckpt Files: {}".format(len(all_ckpt_files)))

                """
                Warning! At this point, I'm not doing BN nets b/c I don't think the 
                nn_homology code will handle it correctly. 
                """
                if args.net == "alex":
                    net = Alex_Net(init_weights=True)
                elif args.net == "let":
                    net = CIFAR_Net()
                else:
                    net = VGG_Net(init_weights=True)

                if args.output_path:
                    convert_model_to_network(
                        net=net,
                        ckpt_files=all_ckpt_files,
                        input_size=(1, 3, 32, 32),
                        save_path=args.output_path,
                        args=args,
                    )
                else:
                    convert_model_to_network(
                        net=net,
                        ckpt_files=all_ckpt_files,
                        input_size=(1, 3, 32, 32),
                        save_path=exp,
                        args=args,
                    )
                del net
            else:
                print("drop*************************")

    # TODO: Can more elegantly do this.
    else:
        glob_path = args.input_path + "/*.pth"
        print(glob_path)
        all_ckpt_files = sort_nicely(glob.glob(glob_path))
        # all_ckpt_files = sort_epoch(glob.glob(glob_path))
        # get all the ckpt files in the directory
        if args.num_convert > 0:
            all_ckpt_files = all_ckpt_files[0: args.num_convert]
            print("=> Num Convert: {}".format(args.num_convert))
            print("=> Num. Ckpt Files: {}".format(len(all_ckpt_files)))
        else:
            print("=> All Ckpt Files: {}".format(len(all_ckpt_files)))

        """
        Warning! At this point, I'm not doing BN nets b/c I don't think the 
        nn_homology code will handle it correctly. 
        """

        if args.net == "alex":
            net = Alex_Net(init_weights=True)
        elif args.net == "let":
            net = CIFAR_Net()
        else:
            net = VGG_Net(init_weights=True)

        if args.output_path:
            # save_path = output
            convert_model_to_network(
                net=net,
                ckpt_files=all_ckpt_files,
                input_size=(1, 3, 32, 32),
                save_path=args.output_path,
                args=args,
            )
        else:
            convert_model_to_network(
                net=net,
                ckpt_files=all_ckpt_files,
                input_size=(1, 3, 32, 32),
                save_path=args.input_path,
                args=args,
            )


if __name__ == "__main__":
    main()
