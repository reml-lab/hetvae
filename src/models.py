# pylint: disable=E1101
from vae_models import (
    HeTVAE_DET,
    HeTVAE,
    HeTVAE_PROB,
)


def load_network(args, dim, union_tp=None, device="cuda"):
    if args.net == 'hetvae':
        net = HeTVAE(
            input_dim=dim,
            nhidden=args.rec_hidden,
            latent_dim=args.latent_dim,
            embed_time=args.embed_time,
            num_heads=args.enc_num_heads,
            intensity=args.intensity,
            union_tp=union_tp,
            width=args.width,
            num_ref_points=args.num_ref_points,
            std=args.std,
            is_constant=args.const_var,
            is_bounded=args.bound_variance,
            is_constant_per_dim=args.var_per_dim,
            elbo_weight=args.elbo_weight,
            mse_weight=args.mse_weight,
            norm=args.norm,
            mixing=args.mixing,
        ).to(device)
    elif args.net == 'hetvae_det':
        net = HeTVAE_DET(
            input_dim=dim,
            nhidden=args.rec_hidden,
            latent_dim=args.latent_dim,
            embed_time=args.embed_time,
            num_heads=args.enc_num_heads,
            intensity=args.intensity,
            union_tp=union_tp,
            width=args.width,
            num_ref_points=args.num_ref_points,
            std=args.std,
            is_constant=args.const_var,
            is_bounded=args.bound_variance,
            is_constant_per_dim=args.var_per_dim,
            elbo_weight=args.elbo_weight,
            mse_weight=args.mse_weight,
            norm=args.norm,
            mixing=args.mixing,
        ).to(device)
    elif args.net == 'hetvae_prob':
        net = HeTVAE_PROB(
            input_dim=dim,
            nhidden=args.rec_hidden,
            latent_dim=args.latent_dim,
            embed_time=args.embed_time,
            num_heads=args.enc_num_heads,
            intensity=args.intensity,
            union_tp=union_tp,
            width=args.width,
            num_ref_points=args.num_ref_points,
            std=args.std,
            is_constant=args.const_var,
            is_bounded=args.bound_variance,
            is_constant_per_dim=args.var_per_dim,
            elbo_weight=args.elbo_weight,
            mse_weight=args.mse_weight,
            norm=args.norm,
            mixing=args.mixing,
        ).to(device)
    else:
        raise ValueError("Network not available")
    return net
