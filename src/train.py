# pylint: disable=E1101, E0401, E1102, W0621, W0221
import argparse
import numpy as np
import torch
import torch.optim as optim

from random import SystemRandom
import models
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--latent-dim', type=int, default=32)
parser.add_argument('--rec-hidden', type=int, default=32)
parser.add_argument('--width', type=int, default=50)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--num-ref-points', type=int, default=128)
parser.add_argument('--k-iwae', type=int, default=10)
parser.add_argument('--save', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--norm', action='store_true')
parser.add_argument('--kl-annealing', action='store_true')
parser.add_argument('--kl-zero', action='store_true')
parser.add_argument('--enc-num-heads', type=int, default=1)
parser.add_argument('--dataset', type=str, default='toy')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--intensity', action='store_true')
parser.add_argument('--net', type=str, default='hetvae')
parser.add_argument('--const-var', action='store_true')
parser.add_argument('--var-per-dim', action='store_true')
parser.add_argument('--std', type=float, default=0.1)
parser.add_argument('--sample-tp', type=float, default=0.5)
parser.add_argument('--bound-variance', action='store_true')
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--recon-loss', action='store_true')
parser.add_argument('--normalize-input', type=str, default='znorm')
parser.add_argument('--mse-weight', type=float, default=0.0)
parser.add_argument('--elbo-weight', type=float, default=1.0)
parser.add_argument('--mixing', type=str, default='concat')
args = parser.parse_args()


if __name__ == '__main__':
    experiment_id = int(SystemRandom().random() * 10000000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'toy':
        data_obj = utils.get_synthetic_data(args)
    elif args.dataset == 'physionet':
        data_obj = utils.get_physionet_data(args.batch_size)
    elif args.dataset == 'mimiciii':
        data_obj = utils.get_mimiciii_data(args.batch_size, filter_anomalies=True)

    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]
    union_tp = utils.union_time(train_loader)

    net = models.load_network(args, dim, union_tp)
    params = list(net.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    print('parameters:', utils.count_parameters(net))

    for itr in range(1, args.niters + 1):
        train_loss = 0
        train_n = 0
        avg_loglik, avg_kl, mse, mae = 0, 0, 0, 0
        if args.kl_annealing:
            wait_until_kl_inc = 10000
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.999999 ** (itr - wait_until_kl_inc))
        elif args.kl_zero:
            kl_coef = 0
        else:
            kl_coef = 1
        for train_batch in train_loader:
            batch_len = train_batch.shape[0]
            train_batch = train_batch.to(device)
            if args.dataset == 'toy':
                subsampled_mask = torch.zeros_like(
                    train_batch[:, :, dim:2 * dim]).to(device)
                seqlen = train_batch.size(1)
                for i in range(batch_len):
                    length = np.random.randint(low=3, high=10)
                    obs_points = np.sort(
                        np.random.choice(np.arange(seqlen), size=length, replace=False)
                    )
                    subsampled_mask[i, obs_points, :] = 1
            else:
                subsampled_mask = utils.subsample_timepoints(
                    train_batch[:, :, dim:2 * dim].clone(),
                    args.sample_tp,
                    shuffle=args.shuffle,
                )
            if args.recon_loss or args.sample_tp == 1.0:
                recon_mask = train_batch[:, :, dim:2 * dim]
            else:
                recon_mask = train_batch[:, :, dim:2 * dim] - subsampled_mask
            context_y = torch.cat((
                train_batch[:, :, :dim] * subsampled_mask, subsampled_mask
            ), -1)

            loss_info = net.compute_unsupervised_loss(
                train_batch[:, :, -1],
                context_y,
                train_batch[:, :, -1],
                torch.cat((
                    train_batch[:, :, :dim] * recon_mask, recon_mask
                ), -1),
                num_samples=args.k_iwae,
                beta=kl_coef,
            )
            optimizer.zero_grad()
            loss_info.composite_loss.backward()
            optimizer.step()
            train_loss += loss_info.composite_loss.item() * batch_len
            avg_loglik += loss_info.loglik * batch_len
            avg_kl += loss_info.kl * batch_len
            mse += loss_info.mse * batch_len
            mae += loss_info.mae * batch_len
            train_n += batch_len
        print(
            'Iter: {}, train loss: {:.4f}, avg nll: {:.4f}, avg kl: {:.4f}, '
            'mse: {:.6f}, mae: {:.6f}'.format(
                itr,
                train_loss / train_n,
                -avg_loglik / train_n,
                avg_kl / train_n,
                mse / train_n,
                mae / train_n
            )
        )
        if itr % 10 == 0:
            for loader, num_samples in [(val_loader, 5), (test_loader, 100)]:
                utils.evaluate_hetvae(
                    net,
                    dim,
                    loader,
                    0.5,
                    shuffle=False,
                    k_iwae=num_samples,
                )
        if itr % 100 == 0 and args.save:
            torch.save({
                'args': args,
                'epoch': itr,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss / train_n,
            }, args.dataset + '_' + str(experiment_id) + '.h5')
