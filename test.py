import pickle

import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='data/METR-LA', help='data path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=8, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--checkpoint', type=str, help='')
parser.add_argument('--plotheatmap', type=str, default='True', help='')
parser.add_argument('--ag_level', type=int)
parser.add_argument('--save', type=str)
parser.add_argument('--model', type=str)

args = parser.parse_args()


def main():
    device = torch.device(args.device)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    _, _, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    print(args)

    model = gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool,
                  addaptadj=args.addaptadj, aptinit=adjinit)
    model.to(device)
    # torch_load = torch.load(args.checkpoint)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    print('model load successfully')

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    amae = []
    amape = []
    armse = []

    for i in range(args.num_nodes):
        if args.num_nodes == 1:
            pred = scaler.inverse_transform(yhat[:, :])
            pred = torch.unsqueeze(pred, 1)
        else:
            pred = scaler.inverse_transform(yhat[:, i, :])
        real = realy[:, i, :]
        metrics = util.metric(pred, real)
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])


    # for i in range(args.seq_length):
    #     if args.num_nodes == 1:
    #         pred = scaler.inverse_transform(yhat[:, i])
    #         pred = torch.unsqueeze(pred, 1)
    #     else:
    #         pred = scaler.inverse_transform(yhat[:, :, i])
    #     real = realy[:, :, i]
    #     # print(f'pred: {type(pred)}, real: {type(real)}')
    #     metrics = util.metric(pred, real)
    #     log = 'Ag level: {:d} Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    #     # print(log.format(args.ag_level, i + 1, metrics[0], metrics[1], metrics[2]))
    #     amae.append(metrics[0])
    #     amape.append(metrics[1])
    #     armse.append(metrics[2])

    # log = 'On average over 8 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    # print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))

    if args.plotheatmap == "True":
        # print('plotting heatmap')
        adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
        device = torch.device('cpu')
        adp.to(device)
        adp = adp.cpu().detach().numpy()
        adp = adp * (1 / np.max(adp))
        df = pd.DataFrame(adp)
        sns.heatmap(df, cmap="RdYlBu")
        plt.savefig(args.save + '.pdf')

    # y12 = realy[:, args.num_nodes - 1, args.seq_length - 1].cpu().detach().numpy()
    # if args.num_nodes == 1:
    #     yhat12 = scaler.inverse_transform(yhat[:, args.seq_length - 1]).cpu().detach().numpy()
    #     # yhat12 = torch.unsqueeze(yhat12, 1)
    # else:
    #     yhat12 = scaler.inverse_transform(yhat[:, args.num_nodes - 1, args.seq_length - 1]).cpu().detach().numpy()
    #
    # y3 = realy[:, args.num_nodes - 1, 2].cpu().detach().numpy()
    # if args.num_nodes == 1:
    #     yhat3 = scaler.inverse_transform(yhat[:, 2]).cpu().detach().numpy()
    # else:
    #     yhat3 = scaler.inverse_transform(yhat[:, args.num_nodes - 1, 2]).cpu().detach().numpy()
    #
    # df2 = pd.DataFrame({'real12': y12, 'pred12': yhat12, 'real3': y3, 'pred3': yhat3})
    # df2.to_csv(args.save + '.csv', index=False)

    res = {}
    res['amae'] = amae
    res['amape'] = amape
    res['armse'] = armse

    with open(f'garage/al{args.ag_level}/{args.model}_res.pkl', 'wb') as pkl:
        pickle.dump(res, pkl)

    return


if __name__ == "__main__":
    main()