import torch
import argparse
from models import *
import wandb
import numpy as np
import trainer
import scipy
import datasets
import torch_geometric as pyg

np.random.seed(0)
seeds = np.random.randint(low=0, high=10000, size=10)

grid_n_layers = [3, 4, 5]
grid_early_stop = [0, 1]
grid_dropout = [0, 0.5]


class EarlyStopping:
    def __init__(self, metric_name, patience=3, min_is_better=False):
        self.metric_name = metric_name
        self.patience = patience
        self.min_is_better = min_is_better
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def reset(self):
        self.counter = 0

    def __call__(self, score):
        if self.min_is_better:
            score = -score

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def run_exp(seeds, n_layers, early_stop_flag, dropout, model_name, num_epochs, wandb_flag, wd,
            hidden_channels, lr, bias, patience, loss_thresh, debug, data_name):
    if debug:
        wandb_flag=False
    for i, seed in enumerate(seeds):

        if model_name == 'gcn':
            model = GCNModel(in_channels=dataset.num_features,
                             hidden_channels=args.hidden_channels, num_layers=args.n_layers,
                             out_channels=dataset.num_classes)
        elif model_name == 'gin':
            model = GINModel(in_channels=dataset.num_features,
                             hidden_channels=hidden_channels, num_layers=n_layers,
                             out_channels=dataset.num_classes, dropout=dropout, bias=bias)

        if model_name == 'gatv2':
            model = GATv2Model(in_channels=dataset.num_features,
                               hidden_channels=hidden_channels, num_layers=n_layers,
                               out_channels=dataset.num_classes, dropout=dropout, bias=bias)

        if model_name == 'graphconv':
            model = GraphConvModel(in_channels=dataset.num_features,
                                   hidden_channels=hidden_channels, num_layers=n_layers,
                                   out_channels=dataset.num_classes, dropout=dropout, bias=bias)

        elif model_name == 'gnam':
            model = GNAM(in_channels=dataset.num_features,
                                   hidden_channels=hidden_channels, num_layers=n_layers,
                                   out_channels=dataset.num_classes, dropout=dropout, bias=bias)

        train_loader = pyg.loader.DataLoader([data], batch_size=1)
        val_loader = pyg.loader.DataLoader([data], batch_size=1)
        test_loader = pyg.loader.DataLoader([data], batch_size=1)

        if debug:
            device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        model.to(device)

        config = {
            'lr': lr,
            'loss': loss_type.__name__,
            'hidden_channels': hidden_channels,
            'n_conv_layers': n_layers,
            'output_dim': dataset.num_classes,
            'num_epochs': num_epochs,
            'optimizer': optimizer_type.__name__,
            'model': model.__class__.__name__,
            'device': device.type,
            'loss_thresh': loss_thresh,
            'debug': debug,
            'wd': wd,
            'bias': bias,
            'dropout': dropout,
            'seed': i,
            'data_name': data_name,
        }

        if wandb_flag:
            for name, val in config.items():
                print(f'{name}: {val}')
            exp_name = f'GNAM_{model.__class__.__name__}_{data_name}'
            wandb.init(project='GNAM', reinit=True, entity='gnnsimplbias',
                       settings=wandb.Settings(start_method='thread'),
                       config=config, name=exp_name)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=wd)
        loss = loss_type()
        early_stop = EarlyStopping(metric_name='Loss', patience=patience, min_is_better=True)
        for epoch in range(num_epochs):
            train_loss, train_acc, train_auc = trainer.train_epoch(model, dloader=train_loader,
                                                                   loss_fn=loss,
                                                                   optimizer=optimizer,
                                                                   classify=True, device=device)
            val_loss, va_acc, val_auc = trainer.test_epoch(model, dloader=val_loader,
                                                           loss_fn=loss, classify=True,
                                                           device=device)

            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Val Acc: {va_acc:.4f}')
            early_stop(val_loss)
            if train_loss < loss_thresh:
                print(f'loss under {loss_thresh} at epoch: {epoch}')
                break
            if early_stop_flag and early_stop.early_stop:
                print(f'early stop at epoch: {epoch}')
                break
            if wandb_flag:
                wandb.log({'train_loss': train_loss,
                           'train_acc': train_acc,
                           'val_loss': val_loss,
                           'val_acc': va_acc,
                           'epoch': epoch})

        # test
        test_loss, test_acc, test_auc = trainer.test_epoch(model, dloader=test_loader,
                                                           loss_fn=loss, classify=True,
                                                           device=device)
        print(f'Test Loss: {test_loss:.4f}, '
              f'Test Acc: {test_acc:.4f}')

        if wandb_flag:
            wandb.log({'test_loss': test_loss,
                       'test_acc': test_acc})
            wandb.finish()

if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', dest='lr', type=float, default=0.001)
    parser.add_argument('--hidden_channels', dest='hidden_channels', type=int, default=64)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--n_layers', dest='n_layers', type=int, default=3)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=1000)
    parser.add_argument('--wandb_flag', dest='wandb_flag', type=int, default=1)
    parser.add_argument('--bias', dest='bias', type=int, default=1)
    parser.add_argument('--patience', dest='patience', type=int, default=500)
    parser.add_argument('--early_stop', dest='early_stop', type=int, default=1)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.0)
    parser.add_argument('--wd', dest='wd', type=float, default=0.00005)
    parser.add_argument('--data_name', dest='data_name', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed'])
    parser.add_argument('--model_name', dest='model_name', type=str, default='gnam',
                        choices=['gcn', 'gin', 'gatv2', 'graphconv', 'gnam'])
    parser.add_argument('--seed', dest='seed', type=int, default=-1, choices=[-1, 10])
    parser.add_argument('--run_grid_search', dest='run_grid_search', type=int, default=0)

    args = parser.parse_args()

    loss_thresh = 0.00001
    loss_type = torch.nn.CrossEntropyLoss
    optimizer_type = torch.optim.Adam

    data_path = 'data'
    if args.data_name == 'cora':
        dataset = datasets.Cora(path=data_path)

    elif args.data_name == 'citeseer':
        dataset = datasets.Citeseer(path=data_path)

    elif args.data_name == 'pubmed':
        dataset = datasets.Pubmed(path=data_path)

    data = dataset[0]
    if args.seed == -1:
        seeds = seeds[:1]

    if args.run_grid_search:
        for n_layers in grid_n_layers:
            for early_stop in grid_early_stop:
                for dropout in grid_dropout:
                    run_exp(n_layers=n_layers, early_stop_flag=early_stop, dropout=dropout,
                            model_name=args.model_name, apply_contractive=args.apply_contractive,
                            num_epochs=args.num_epochs, wandb_flag=args.wandb_flag, wd=args.wd,
                            hidden_channels=args.hidden_channels, lr=args.lr, bias=args.bias, patience=args.patience,
                            debug=args.debug, loss_thresh=loss_thresh, seeds=seeds, data_name=args.data_name)

    else:
        run_exp(n_layers=args.n_layers, early_stop_flag=args.early_stop, dropout=args.dropout, model_name=args.model_name,
                num_epochs=args.num_epochs, wandb_flag=args.wandb_flag, wd=args.wd,
                hidden_channels=args.hidden_channels, lr=args.lr, bias=args.bias, patience=args.patience, debug=args.debug,loss_thresh=loss_thresh, seeds=seeds, data_name=args.data_name)


