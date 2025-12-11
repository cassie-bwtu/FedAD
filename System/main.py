import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import time
import numpy as np
import copy
import torch
import torch.nn as nn
import argparse
import warnings
import torchvision
import logging
from flcore.trainmodel.models import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.regnet import *
from flcore.trainmodel.mobilenet_v2 import *
from utils.result_utils import average_data
from flcore.servers.server_fedavg import FedAvg
from flcore.servers.server_freeze import FedFreeze
from flcore.servers.server_drop import FedDrop
from flcore.servers.server_fedad import FedAD
from pytorch_lightning import seed_everything
from flcore.trainmodel.policy_network import *
from torchvision import models

seed = 42
seed_everything(seed)

def run(args):
    time_list = []
    model_name = args.model
    pn_name = args.pn_name

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        if model_name == 'resnet18':
            if "mnist" in args.dataset:
                model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)
                model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                args.model = model.to(args.device)

                if pn_name == 'ResBlock':
                    policy_network = policynet(num_blocks=args.num_pn_blocks, num_classes=args.num_blocks * 2)
                    policy_network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                elif pn_name == 'CNN':
                    policy_network = policy_netCNN(input_channels=1, num_classes=args.num_blocks * 2)
                    policy_network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    policy_network.fc = nn.Linear(128 * 7 * 7, args.num_blocks * 2)

                args.policynet = policy_network.to(args.device)

            else:
                args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
                if pn_name == 'ResBlock':
                    policy_network = policynet(num_blocks=args.num_pn_blocks, num_classes=args.num_blocks * 2)
                elif pn_name == 'CNN':
                    policy_network = policy_netCNN(input_channels=3, num_classes=args.num_blocks * 2)
                args.policynet = policy_network.to(args.device)

        elif model_name == 'resnet34':
            if "mnist" in args.dataset:
                model = resnet34(num_classes=args.num_classes)
                model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                args.model = model.to(args.device)

                if pn_name == 'ResBlock':
                    policy_network = policynet(num_blocks=args.num_pn_blocks, num_classes=args.num_blocks * 2)
                    policy_network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                elif pn_name == 'CNN':
                    policy_network = policy_netCNN(input_channels=1, num_classes=args.num_blocks * 2)
                    policy_network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    policy_network.fc = nn.Linear(128 * 7 * 7, args.num_blocks * 2)

                args.policynet = policy_network.to(args.device)

            else:
                args.model = resnet34(num_classes=args.num_classes).to(args.device)

                if pn_name == 'ResBlock':
                    policy_network = policynet(num_blocks=args.num_pn_blocks, num_classes=args.num_blocks * 2)
                elif pn_name == 'CNN':
                    policy_network = policy_netCNN(input_channels=3, num_classes=args.num_blocks * 2)
                args.policynet = policy_network.to(args.device)

        elif model_name == 'resnet50':
            if "mnist" in args.dataset:
                model = resnet50(num_classes=args.num_classes)
                model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                args.model = model.to(args.device)

                if pn_name == 'ResBlock':
                    # batch
                    policy_network = policynet(num_blocks=args.num_pn_blocks, num_classes=args.num_blocks * 2)
                    policy_network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                elif pn_name == 'CNN':
                    policy_network = policy_netCNN(input_channels=1, num_classes=args.num_blocks * 2)
                    policy_network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    policy_network.fc = nn.Linear(128 * 7 * 7, args.num_blocks * 2)

                args.policynet = policy_network.to(args.device)

            else:
                args.model = resnet50(num_classes=args.num_classes).to(args.device)
                if pn_name == 'ResBlock':
                    policy_network = policynet(num_blocks=args.num_pn_blocks, num_classes=args.num_blocks * 2)
                elif pn_name == 'CNN':
                    policy_network = policy_netCNN(input_channels=3, num_classes=args.num_blocks * 2)
                args.policynet = policy_network.to(args.device)

        elif model_name == "regnet":
            args.model = regnet_200m().to(args.device)
            policy_network = PolicyNetRegNet(num_classes=args.num_blocks * 2)
            args.policynet = policy_network.to(args.device)

        elif model_name == "mobilenet_v2":
            if 'mnist' in args.dataset:
                model = models.mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)
                model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                args.model = model.to(args.device)

                policy_network = PolicyNetMN2(num_classes=args.num_blocks * 2)
                policy_network.base[0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                args.policynet = policy_network.to(args.device)
            else:
                args.model = models.mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)
                policy_network = PolicyNetMN2(num_classes=args.num_blocks * 2)
                args.policynet = policy_network.to(args.device)

        elif model_name == "vit":
            if "Cifar" in args.dataset:
                args.model = ViTForImageClassification.from_pretrained('WinKawaks/vit-small-patch16-224', num_labels=args.num_classes, ignore_mismatched_sizes=True).to(args.device)
                if pn_name == 'ResBlock':
                    policy_network = policynet(num_blocks=args.num_pn_blocks, num_classes=args.num_blocks * 2)
                elif pn_name == 'CNN':
                    policy_network = policy_netCNN(input_channels=3, num_classes=args.num_blocks * 2)
                args.policynet = policy_network.to(args.device)

        elif model_name == 'bert':
            if 'SST' in args.dataset:
                args.model = DistilBertForSequenceClassification.from_pretrained('prajjwal1/bert-mini',num_labels=2,ignore_mismatched_sizes=True).to(args.device)
                policy_network = policy_netLSTM(num_classes=args.num_blocks * 2)
                args.policynet = policy_network.to(args.device)

        else:
            raise NotImplementedError

        print(args.model)


        # select algorithm

        if args.algorithm == 'FedAvg':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)

        elif args.algorithm == 'Freeze':
            if args.model_name == 'resnet18':
                args.model = BlockSplitRN18(args.model)
                server = FedFreeze(args, i)

        elif args.algorithm == 'Drop':
            if args.model_name == 'resnet18':
                args.model = BlockSplitRN18(args.model)
                server = FedDrop(args, i)

            elif args.model_name == 'resnet34':
                args.model = BlockSplitRN34(args.model)
                server = FedDrop(args, i)

            elif args.model_name == 'resnet50':
                args.model = BlockSplitRN50(args.model)
                server = FedDrop(args, i)

            elif args.model_name == 'regnet':
                args.model = BlockSplitRegNet(args.model)
                server = FedDrop(args, i)

            elif model_name == "mobilenet_v2":
                args.model = BlockSplitMN2(args.model)
                server = FedDrop(args, i)

            elif model_name == "vit":
                args.model = BlockSplitViT(args.model)
                server = FedDrop(args, i)

            elif model_name == "bert":
                args.model = BlockSplitDistilBERT(args.model)
                server = FedDrop(args, i)

        elif args.algorithm == 'Fedad':
            if args.model_name == 'resnet18':
                args.model = BlockSplitRN18(args.model)
                if args.dataset == 'Tiny-imagenet':
                    args.block_flops = torch.tensor([37.94, 37.94, 28.40, 37.84, 28.36, 37.80, 28.34, 37.78])
                elif args.dataset == 'mnist' or args.dataset == 'emnist':
                    args.block_flops = torch.tensor([7.26, 7.26, 7.1, 9.46, 7.08, 9.44, 7.08, 9.44])
                elif args.dataset == 'Cifar100':
                    args.block_flops = torch.tensor([9.48, 9.48, 7.1, 9.46, 7.08, 9.44, 7.08, 9.44])
                server = FedAD(args, i)

            elif args.model_name == 'resnet34':
                args.model = BlockSplitRN34(args.model)
                if args.dataset == 'Tiny-imagenet':
                    args.block_flops = torch.tensor([37.92, 37.92, 37.92, 28.4, 37.84, 37.84, 37.84, 28.36, 37.78, 37.78, 37.78, 37.78, 37.78, 28.34, 37.76, 37.76])
                elif args.dataset == 'mnist' or args.dataset == 'emnist':
                    args.block_flops = torch.tensor([7.26, 7.26, 7.26, 7.1, 9.46, 9.46, 9.46, 7.08, 9.44, 9.44, 9.44, 9.44, 9.44, 7.08, 9.44, 9.44])
                elif args.dataset == 'Cifar10':
                    args.block_flops = torch.tensor([9.48, 9.48, 9.48, 7.1, 9.46, 9.46, 9.46, 7.08, 9.44, 9.44, 9.44, 9.44, 9.44, 7.08, 9.44, 9.44])
                server = FedAD(args, i)

            elif args.model_name == 'resnet50':
                args.model = BlockSplitRN50(args.model)
                if args.dataset == 'Tiny-imagenet':
                    args.block_flops = torch.tensor([29.76, 36.04, 36.04, 44.34, 35.84, 35.84, 35.84, 44.18, 35.74, 35.74, 35.74, 35.74, 35.74, 44.12, 35.7, 35.7])
                elif args.dataset == 'Cifar100' or args.dataset == 'Cifar10':
                    args.block_flops = torch.tensor([7.44, 9.02, 9.02, 11.08, 8.96, 8.96, 8.96, 11.04, 8.94, 8.94, 8.94, 8.94, 8.94, 11.02, 8.92, 8.92])
                elif args.dataset == 'mnist' or args.dataset == 'emnist':
                    args.block_flops = torch.tensor([5.7, 6.9, 6.9, 10.1, 8.96, 8.96, 8.96, 11.04, 8.94, 8.94, 8.94, 8.94, 8.94, 11.02, 8.92, 8.92])
                server = FedAD(args, i)

            elif model_name == "regnet":
                args.model = BlockSplitRegNet(args.model)
                if args.dataset == 'Cifar100':
                    args.block_flops = torch.tensor([6.68, 6.9, 1.12, 9.04, 7.48, 7.48, 7.5, 12.54, 9.62, 9.62, 9.62, 9.62, 9.62, 9.62])
                server = FedAD(args, i)

            elif model_name == "mobilenet_v2":
                args.model = BlockSplitMN2(args.model)
                if args.dataset == 'Tiny-imagenet':
                    args.block_flops = torch.tensor([4.66, 1.95, 1.95, 1.762, 1.762, 1.762, 3.82, 3.82, 2.58, 2.58])
                server = FedAD(args, i)

            elif model_name == "vit":
                args.model = BlockSplitViT(args.model)
                if args.dataset == 'Cifar100':
                    args.block_flops = torch.tensor([8.87, 8.87, 8.87, 8.87, 8.87, 8.87, 8.87, 8.87, 8.87, 8.87, 8.87, 8.87])
                server = FedAD(args, i)

            elif model_name == "bert":
                args.model = BlockSplitDistilBERT(args.model)
                if args.dataset == 'SST2':
                    # args.block_flops = torch.tensor([907.05, 907.05, 907.05, 907.05, 907.05, 907.05])
                    args.block_flops = torch.tensor([101.02, 101.02, 101.02, 101.02])
                server = FedAD(args, i)

        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # average_data(dataset=args.dataset, algorithm=args.algorithm, dataset_info=args.dataset_info, model_name=args.model_name, goal=args.goal, num_clients=args.num_clients, static=args.static, ratio=args.ratio, times=args.times)

    torch.cuda.empty_cache()

    print("All done!")

if __name__ == '__main__':
    total_start = time.time()
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-datainfo', "--dataset_info", type=str, default="iid")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-mn', "--model_name", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.1,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=1e-5)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=0.1,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=100,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=True)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    parser.add_argument('-nd', "--num_drop", type=int, default=0)
    parser.add_argument('-rt', "--ratio", type=float, default=1.0)
    parser.add_argument('-static', "--static", type=bool, default=False,
                        help="Whether the drop policy for each client is static")
    parser.add_argument('-rc', "--roundchange", type=bool, default=False,
                        help="Whether the drop policy for each client change each round")
    parser.add_argument('-nbs', "--num_blocks", type=int, default=8)
    parser.add_argument('-npbs', "--num_pn_blocks", type=int, default=1)

    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")

    # Feddrop
    parser.add_argument('-mf', "--model_flops", type=float, default=148.86,
                        help="The FLOPs for the global model")
    parser.add_argument('-basicf', "--basic_flops", type=float, default=11.61,
                        help="The FLOPs for the backbone of the global model")
    parser.add_argument('-blockf', "--block_flops", type=float, default=0.0,
                        help="The FLOPs for each block")
    parser.add_argument('-alp', "--alpha", type=float, default=0.0,
                        help="The weight for efficiency loss")
    parser.add_argument('-ga', "--gamma", type=float, default=1.0,
                        help="The weight for efficiency loss")
    parser.add_argument('-fr', "--flops_rate", type=str, default='0.5,0.3,0.2',
                        help="The rate for each level of devices")
    parser.add_argument('-fl', "--flops_level", type=str, default='1,0.66,0.33',
                        help="The levels for computational ability of devices")
    parser.add_argument('-maxf', "--max_flops", type=float, default=16.2,
                        help="The FLOPs for the strong devices")
    parser.add_argument('-pnlr', "--policy_network_learning_rate", type=float, default=1,
                        help="Policy network learning rate")
    parser.add_argument('-s', "--sigma", type=float, default=0.5)
    parser.add_argument('-ac', "--add_cons", type=bool, default=False,
                        help="Whether or not adding flops constraint")
    parser.add_argument('-act', "--add_cons_test", type=bool, default=False,
                        help="Whether or not adding flops constraint for testing")
    parser.add_argument('-avgtrain', "--avg_num_train", type=int, default=990,
                        help="average number of train samples")
    parser.add_argument('-avgtest', "--avg_num_test", type=int, default=110,
                        help="average number of test samples")
    parser.add_argument('-tg', '--time_gap', type=int, default=5,
                        help="evaluation gap")
    parser.add_argument('-c0fl', "--client0_flops_level", type=float, default=1.0)
    parser.add_argument('-dc', "--dynamic", type=bool, default=False,
                        help="Whether or not dynamic change of the device flops")
    parser.add_argument('-cg', '--change_gap', type=int, default=1,
                        help="change gap")
    parser.add_argument('-ta', "--tau", type=float, default=0.6)
    parser.add_argument('-record', "--record", type=bool, default=False)
    parser.add_argument('-pnn', "--pn_name", type=str, default="ResBlock")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local epochs: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Complete model ratio: {}".format(args.ratio))
    print("Using device: {}".format(args.device))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    print("=" * 50)

    run(args)

    print(f"\nTotal time cost: {round(time.time() - total_start, 2)}s.")

