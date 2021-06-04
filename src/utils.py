# pylint: disable=E1101
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn import model_selection


def union_time(data_loader, classif=False):
    tu = []
    for batch in data_loader:
        if classif:
            batch = batch[0]
        tp = batch[:, :, -1].numpy().flatten()
        for val in tp:
            if val not in tu:
                tu.append(val)
    tu.sort()
    return torch.from_numpy(np.array(tu))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2.0 * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -0.5 * (const + logvar + (x - mean) ** 2.0 / torch.exp(logvar)) * mask


def mog_log_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2.0 * np.pi])).float().to(x.device)
    const = torch.log(const)
    const2 = torch.from_numpy(np.array([mean.size(0)])).float().to(x.device)
    loglik = -0.5 * (const + logvar + (x - mean) ** 2.0 / torch.exp(logvar)) * mask
    return torch.logsumexp(loglik - torch.log(const2), 0)


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.0
    lstd2 = lv2 / 2.0
    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.0) / (2.0 * v2)) - 0.5
    return kl


def mean_squared_error(orig, pred, mask):
    error = (orig - pred) ** 2
    error = error * mask
    return error.sum() / mask.sum()


def mean_absolute_error(orig, pred, mask):
    error = torch.abs(orig - pred)
    error = error * mask
    return error.sum() / mask.sum()


def evaluate_hetvae(
    net,
    dim,
    train_loader,
    sample_tp=0.5,
    shuffle=False,
    k_iwae=1,
    device='cuda',
):
    torch.manual_seed(seed=0)
    np.random.seed(seed=0)
    train_n = 0
    avg_loglik, mse, mae = 0, 0, 0
    mean_mae, mean_mse = 0, 0
    with torch.no_grad():
        for train_batch in train_loader:
            train_batch = train_batch.to(device)
            subsampled_mask = subsample_timepoints(
                train_batch[:, :, dim:2 * dim].clone(),
                sample_tp,
                shuffle=shuffle,
            )
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
                num_samples=k_iwae,
            )
            num_context_points = recon_mask.sum().item()
            mse += loss_info.mse * num_context_points
            mae += loss_info.mae * num_context_points
            mean_mse += loss_info.mean_mse * num_context_points
            mean_mae += loss_info.mean_mae * num_context_points
            avg_loglik += loss_info.mogloglik * num_context_points
            train_n += num_context_points
    print(
        'nll: {:.4f}, mse: {:.4f}, mae: {:.4f}, '
        'mean_mse: {:.4f}, mean_mae: {:.4f}'.format(
            - avg_loglik / train_n,
            mse / train_n,
            mae / train_n,
            mean_mse / train_n,
            mean_mae / train_n
        )
    )


def get_mimiciii_data(batch_size, test_batch_size=5, filter_anomalies=True):
    input_dim = 12
    x = np.load("../../neuraltimeseries/Dataset/final_input3.npy")
    x = x[:, :25]
    x = np.transpose(x, (0, 2, 1))
    observed_vals, observed_mask, observed_tp = (
        x[:, :, :input_dim],
        x[:, :, input_dim: 2 * input_dim],
        x[:, :, -1],
    )
    print(observed_vals.shape, observed_mask.shape, observed_tp.shape)

    if np.max(observed_tp) != 0.0:
        observed_tp = observed_tp / np.max(observed_tp)

    if filter_anomalies:
        data_mean, data_std = [], []
        var_dict = {}
        hth = []
        lth = []
        for i in range(input_dim):
            var_dict[i] = []
        for i in range(observed_vals.shape[0]):
            for j in range(input_dim):
                indices = np.where(observed_mask[i, :, j] > 0)[0]
                var_dict[j] += observed_vals[i, indices, j].tolist()

        for i in range(input_dim):
            th1 = np.quantile(var_dict[i], 0.001)
            th2 = np.quantile(var_dict[i], 0.9995)
            hth.append(th2)
            lth.append(th1)
            temp = []
            for val in var_dict[i]:
                if val <= th2 and val >= th1:
                    temp.append(val)
            if len(np.unique(temp)) > 10:
                data_mean.append(np.mean(temp))
                data_std.append(np.std(temp))
            else:
                data_mean.append(0)
                data_std.append(1)

        # normalizing
        observed_vals = (observed_vals - data_mean) / data_std
        observed_vals[observed_mask == 0] = 0
    else:
        for k in range(input_dim):
            data_min, data_max = float("inf"), 0.0
            for i in range(observed_vals.shape[0]):
                for j in range(observed_vals.shape[1]):
                    if observed_mask[i, j, k]:
                        data_min = min(data_min, observed_vals[i, j, k])
                        data_max = max(data_max, observed_vals[i, j, k])
            # print(data_min, data_max)
            if data_max == 0:
                data_max = 1
            observed_vals[:, :, k] = (observed_vals[:, :, k] - data_min) / data_max
        # set masked out elements back to zero
        observed_vals[observed_mask == 0] = 0

    total_dataset = np.concatenate(
        (observed_vals, observed_mask, observed_tp[:, :, None]), -1)
    print(total_dataset.shape)
    # Shuffle and split
    train_data, test_data = model_selection.train_test_split(
        total_dataset, train_size=0.8, random_state=42, shuffle=True
    )
    # for interpolation, we dont need a non-overlapping validation set as
    # we can condition on different set of time points from same set to
    # create distinct examples
    _, val_data = model_selection.train_test_split(
        train_data, train_size=0.8, random_state=11, shuffle=True
    )
    print(train_data.shape, val_data.shape, test_data.shape)
    train_data = torch.from_numpy(train_data).float()
    val_data = torch.from_numpy(val_data).float()
    test_data = torch.from_numpy(test_data).float()

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=100, shuffle=False)

    data_objects = {
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "val_dataloader": val_dataloader,
        "input_dim": input_dim,
    }
    return data_objects


def get_physionet_data(batch_size, test_batch_size=5):
    input_dim = 41
    data = np.load("../data/physionet_compressed.npz")
    train_data, test_data = data['train'], data['test']
    # for interpolation, we dont need a non-overlapping validation set as
    # we can condition on different set of time points from same dataset to
    # create a distinct example
    _, val_data = model_selection.train_test_split(
        train_data, train_size=0.8, random_state=11, shuffle=True
    )
    print(train_data.shape, test_data.shape)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=100, shuffle=False)

    data_objects = {
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "val_dataloader": val_dataloader,
        "input_dim": input_dim,
    }
    return data_objects


def get_synthetic_data(
    args,
    alpha=120.0,
    seed=0,
    ref_points=10,
    total_points=51,
    add_noise=True,
):
    np.random.seed(seed)
    ground_truth, ground_truth_tp = [], []
    observed_values = []
    for _ in range(args.n):
        key_values = np.random.randn(ref_points)
        key_points = np.linspace(0, 1, ref_points)
        query_points = np.linspace(0, 1, total_points)
        weights = np.exp(-alpha * (
            np.expand_dims(query_points, 1) - np.expand_dims(key_points, 0)
        ) ** 2)
        weights /= weights.sum(1, keepdims=True)
        query_values = np.dot(weights, key_values)
        ground_truth.append(query_values)
        if add_noise:
            noisy_query_values = query_values + 0.1 * np.random.randn(total_points)
        observed_values.append(noisy_query_values)
        ground_truth_tp.append(query_points)

    observed_values = np.array(observed_values)
    ground_truth = np.array(ground_truth)
    ground_truth_tp = np.array(ground_truth_tp)
    observed_mask = np.ones_like(observed_values)

    observed_values = np.concatenate(
        (
            np.expand_dims(observed_values, axis=2),
            np.expand_dims(observed_mask, axis=2),
            np.expand_dims(ground_truth_tp, axis=2),
        ),
        axis=2,
    )
    print(observed_values.shape)
    train_data, test_data = model_selection.train_test_split(
        observed_values, train_size=0.8, random_state=42, shuffle=True
    )
    _, ground_truth_test = model_selection.train_test_split(
        ground_truth, train_size=0.8, random_state=42, shuffle=True
    )
    _, val_data = model_selection.train_test_split(
        train_data, train_size=0.8, random_state=42, shuffle=True
    )
    print(train_data.shape, val_data.shape, test_data.shape)
    train_dataloader = DataLoader(
        torch.from_numpy(train_data).float(), batch_size=args.batch_size, shuffle=False
    )
    val_dataloader = DataLoader(
        torch.from_numpy(val_data).float(), batch_size=args.batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        torch.from_numpy(test_data).float(), batch_size=5, shuffle=False
    )

    data_objects = {
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "val_dataloader": val_dataloader,
        "input_dim": 1,
        "ground_truth": ground_truth_test,
    }
    return data_objects


def subsample_timepoints(mask, percentage_tp_to_sample=None, shuffle=False):
    # Subsample percentage of points from each time series
    if not shuffle:
        seed = 0
        np.random.seed(seed)
    else:
        seed = np.random.randint(0, 100000)
        np.random.seed(seed)
    for i in range(mask.size(0)):
        # take mask for current training sample and sum over all features --
        # figure out which time points don't have any measurements at all in this batch
        current_mask = mask[i].sum(-1).cpu()
        non_missing_tp = np.where(current_mask > 0)[0]
        n_tp_current = len(non_missing_tp)
        n_to_sample = int(n_tp_current * percentage_tp_to_sample)
        subsampled_idx = sorted(
            np.random.choice(non_missing_tp, n_to_sample, replace=False))
        tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)
        if mask is not None:
            mask[i, tp_to_set_to_zero] = 0.
    return mask
