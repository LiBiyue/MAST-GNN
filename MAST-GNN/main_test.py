import time
import random
import argparse
import torch

import numpy as np
import torch.optim as optim
import torch.nn as nn

from utils.adj_normalization import load_adj, load_multi_adj
from utils.data_generator import generate_train_val_test
from models.trainer import Trainer
from models.model import AirspaceModel


def init_seed(seed=2021):
    '''
    Disable cudnn to maximize reproducibility
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True


def print_model_parameters(model, only_num=True):
    print('\n***************** Model Parameter *****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))


def get_args():
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--feature_data_path", type=str, default=r"E:\zhaohaoran\all_day_feature_single_daytime.npy")
    parser.add_argument("--adj_data_path", type=str, default=r"E:\zhaohaoran\adj_mx_geo_126.csv")
    parser.add_argument("--adj_type", type=str, default="doubletransition")
    parser.add_argument('--n_vertex', type=int, default=126)
    parser.add_argument("--seq_length_x", type=int, default=120)
    parser.add_argument("--seq_length_y", type=int, default=120)
    parser.add_argument('--blocks', default=4, type=int)
    parser.add_argument('--layers', default=5, type=int)
    parser.add_argument("--y_start", type=int, default=1)
    parser.add_argument("--use_multi_graph", dest='multi_graph', action='store_true', default=False)
    # model args
    parser.add_argument('--in_dims', type=int, default=17)
    parser.add_argument('--out_dims', type=int, default=1)
    parser.add_argument('--hid_dims', type=int, default=128)
    parser.add_argument('--nonuse_graph_conv', dest='use_graph_conv', action='store_false', default=True)
    parser.add_argument('--nonuse_graph_learning', dest='use_graph_learning', action='store_false', default=True)#自适应邻居矩阵

    # train args
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_rate', default=0.1, type=float)
    parser.add_argument('--lr_decay_step', default='5,20,40,70', type=str)
    parser.add_argument('--drop_rate', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_norm', default=True, type=bool)
    parser.add_argument('--max_grad_norm', default=3, type=int)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--device', type=str, default='cuda:2')
    # log & save args
    parser.add_argument('--log_dir', type=str, default='./checkpoints')
    parser.add_argument('--use_debug', dest='debug', action='store_true', default=False)
    parser.add_argument('--checkpoint',default=r"./checkpoints\log_04221152\best_model_04221152.pth", type=str)
    # default='./checkpoints/log_06072159/best_model_06072159.pth'
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resumedir', type=str, default="./checkpoints/log_04192141/best_model_04192141.pth")

    args = parser.parse_args()
    return args


def main():
    #--------------------------------------------------------
    #------------------ Init data & args --------------------
    #--------------------------------------------------------
    init_seed()
    args = get_args()
    train_loader, val_loader, test_loader, scaler = generate_train_val_test(args)
    if args.multi_graph:
        adaptive_mat, adj_mats = load_multi_adj('data/adj_mx_geo_126.csv', 'data/adj_mx_flow_126.csv', use_graph_learning=args.use_graph_learning)
    else:
        adaptive_mat, adj_mats = load_adj(args.adj_data_path, args.adj_type, use_graph_learning=args.use_graph_learning)
    adj_mats_torch = [torch.tensor(i).to(args.device) for i in adj_mats]
    adaptive_mat_torch = torch.tensor(adaptive_mat).to(args.device) if args.use_graph_learning else None
    print(f'\n***************** Input Args ******************\n{args}')

    #--------------------------------------------------------
    #----------------- Init model & trainer -----------------
    #--------------------------------------------------------
    model = AirspaceModel(in_channels=args.in_dims,
                          out_channels=args.seq_length_y,
                          residual_channels=args.hid_dims,
                          dilation_channels=args.hid_dims,
                          skip_channels=args.hid_dims*8,
                          end_channels=args.hid_dims*4,
                          supports=adj_mats_torch,
                          use_graph_conv=args.use_graph_conv,
                          adaptive_mat_init=adaptive_mat_torch,
                          blocks=args.blocks,
                          layers=args.layers,
                          device=args.device
                          )
    model = model.to(args.device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    print_model_parameters(model, only_num=True)

    loss = nn.MSELoss().to(args.device)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, eps=1.0e-8, weight_decay=args.weight_decay, amsgrad=False)
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_steps, gamma=args.lr_decay_rate)
    if not args.checkpoint:
        if args.resume:
            print("load model at %s", args.resumedir)
            model.load_state_dict(torch.load(args.resumedir, map_location=args.device))
            trainer = Trainer(model=model,
                              loss=loss,
                              optimizer=optimizer,
                              lr_scheduler=lr_scheduler,
                              train_loader=train_loader,
                              val_loader=val_loader,
                              test_loader=test_loader,
                              scaler=scaler,
                              args=args)
            trainer.train()
        else:
            trainer = Trainer(model=model,
                              loss=loss,
                              optimizer=optimizer,
                              lr_scheduler=lr_scheduler,
                              train_loader=train_loader,
                              val_loader=val_loader,
                              test_loader=test_loader,
                              scaler=scaler,
                              args=args)
            trainer.train()
    else:
        model.load_state_dict(torch.load(args.checkpoint))
        tester = Trainer(model=model,
                        loss=loss,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        scaler=scaler,
                        args=args)
        tester.test()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f'Total time spent: {end_time-start_time:.4f}')
