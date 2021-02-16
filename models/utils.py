# -*- coding: UTF-8 -*-
# Local modules
import os

# 3rd party modules
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.tensorboard import SummaryWriter

# Self-written modules
from models.dataset import RCGANDataset

def sample_C(Y_true):
    """Sample wrong conditional vector (B x 1) for discriminator
    """
    size = Y_true.size()
    labels = set(Y_true)

    # Generate from 0 ~ n-1
    if len(labels):
        Y_false = Y_true
    else:
        Y_false = torch.randint(min(labels).item(), max(labels).item(), size=size)

    # Add 1 to values that are the same as the index
    for idx, t in enumerate(Y_true):
        if Y_false[idx] >= t:
            Y_false[idx] += 1

    return Y_false

def get_feat_batch(feat, i, batch_size):
    start = i * batch_size
    end = start + batch_size
    if end > len(feat):
        return feat[start:]
    return feat[start:end]

def get_labels(data, time):
    no, seq_len, dim = data.shape
    labels = np.empty([no, 1])
    for idx, t in enumerate(time):
        label = data[idx, :t, -1]
        labels[idx] = (label.mean() > 0.5).astype(float)
    return labels

def rgan_trainer(model, data, time, args):
    # Initialize TimeGAN dataset and dataloader
    dataset = RGANDataset(data, time)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Move model to device
    model.to(args.device)

    # Initialize Optimizers
    g_opt = torch.optim.Adam(model.generator.parameters(), lr=args.learning_rate)
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=args.learning_rate)

    # TensorBoard writer
    writer = SummaryWriter(os.path.join(f"tensorboard/{args.exp}"))

    logger = trange(
        args.epochs,
        desc=f"Epoch: 0, G_loss: 0., D_loss: 0."
    )

    for epoch in logger:
        for X_mb, T_mb, in dataloader:
            # Sample random noise
            Z_mb = torch.rand(
                (X_mb.size(0), args.max_seq_len, args.Z_dim)
            ).to(args.device)
            X_mb = X_mb.to(args.device)

            #########################
            # Discriminator forward pass
            #########################
            for _ in range(1):
                model.zero_grad()
                d_opt.zero_grad()
                g_opt.zero_grad()

                d_loss, _ = model(
                    X=X_mb,
                    Z=Z_mb,
                    T=T_mb
                )
                d_loss.backward()
                d_opt.step()

            #########################
            # Generator forward pass
            #########################
            for _ in range(1):
                model.zero_grad()
                d_opt.zero_grad()
                g_opt.zero_grad()

                _, g_loss = model(
                    X=X_mb,
                    Z=Z_mb,
                    T=T_mb,
                )
                g_loss.backward()
                g_opt.step()

        logger.set_description(
            f"Epoch: {epoch}, G: {g_loss:.4f}, D: {d_loss:.4f}"
        )

        if writer:
            writer.add_scalar(
                'Generator_Loss:',
                g_loss,
                epoch
            )
            writer.add_scalar(
                'Discriminator_Loss:',
                d_loss,
                epoch
            )
            writer.flush()

    # Save model, args, and hyperparameters
    torch.save(args, f"{args.model_path}/args.pickle")
    torch.save(model.state_dict(), f"{args.model_path}/model.pt")
    print(f"\nSaved at path: {args.model_path}\n")

def rgan_generator(model, T, batch_size, model_path):
    """Inference procedure for RGAN
    """
    # Load model for inference
    if not os.path.exists(model_path):
        raise ValueError(f"Model directory not found...")

    # Load arguments and model
    with open(f"{model_path}/args.pickle", "rb") as fb:
        args = torch.load(fb)

    model.load_state_dict(torch.load(f"{model_path}/model.pt"))

    print("Generating Data...")
    generated_data = []

    # Initialize model to evaluation mode and run without gradients
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        iters = (len(T) // batch_size + 1)
        for i in range(iters):
            # Sample random input
            T_mb = get_feat_batch(T, i, batch_size)
            Z_mb = torch.rand(
                (batch_size, args.max_seq_len, args.Z_dim)
            ).to(args.device)
            generated_data.append(
                model.generate(Z=Z, T=T).cpu().numpy()
            )

    generated_data = np.vstack(generated_data)[:len(T)]
    generated_labels = get_labels(generated_data, T)
    generated_time = T

    return generated_data, generated_labels, generated_time

def rcgan_trainer(model, data, time, label, args):
    # Initialize TimeGAN dataset and dataloader
    dataset = RCGANDataset(data, label, time)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Move model to device
    model.to(args.device)

    # Initialize Optimizers
    g_opt = torch.optim.Adam(model.generator.parameters(), lr=args.learning_rate)
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=args.learning_rate)

    # TensorBoard writer
    writer = SummaryWriter(os.path.join(f"tensorboard/{args.exp}"))

    logger = trange(
        args.epochs,
        desc=f"Epoch: 0, G_loss: 0., D_loss: 0."
    )

    for epoch in logger:
        for X_mb, T_mb, Y_mb in dataloader:
            # Sample random noise
            Z_mb = torch.rand(
                (X_mb.size(0), args.max_seq_len, args.Z_dim)
            ).to(args.device)
            # Sample random label
            C_mb = sample_C(Y_mb)

            X_mb = X_mb.to(args.device)
            Y_mb = Y_mb.to(args.device)

            #########################
            # Discriminator forward pass
            #########################
            for _ in range(1):
                model.zero_grad()
                d_opt.zero_grad()
                g_opt.zero_grad()

                d_loss, _ = model(
                    X=X_mb,
                    Z=Z_mb,
                    T=T_mb,
                    C=Y_mb,
                    C_wrong=C_mb
                )
                d_loss.backward()
                d_opt.step()

            #########################
            # Generator forward pass
            #########################
            for _ in range(1):
                model.zero_grad()
                d_opt.zero_grad()
                g_opt.zero_grad()

                _, g_loss = model(
                    X=X_mb,
                    Z=Z_mb,
                    T=T_mb,
                    C=Y_mb,
                    C_wrong=C_mb
                )
                g_loss.backward()
                g_opt.step()

        logger.set_description(
            f"Epoch: {epoch}, G: {g_loss:.4f}, D: {d_loss:.4f}"
        )
        if writer:
            writer.add_scalar(
                'Generator_Loss:',
                g_loss,
                epoch
            )
            writer.add_scalar(
                'Discriminator_Loss:',
                d_loss,
                epoch
            )
            writer.flush()

    # Save model, args, and hyperparameters
    torch.save(args, f"{args.model_path}/args.pickle")
    torch.save(model.state_dict(), f"{args.model_path}/model.pt")
    print(f"\nSaved at path: {args.model_path}\n")

def rcgan_generator(model, T, labels, batch_size, model_path):
    """Inference procedure for RCGAN
    """
    # Load model for inference
    if not os.path.exists(model_path):
        raise ValueError(f"Model directory not found...")

    # Load arguments and model
    with open(f"{model_path}/args.pickle", "rb") as fb:
        args = torch.load(fb)

    model.load_state_dict(torch.load(f"{model_path}/model.pt"))

    print("Generating Data...")
    generated_data = []

    # Initialize model to evaluation mode and run without gradients
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        iters = (len(T) // batch_size + 1)
        for i in range(iters):
            # Sample random input
            T_mb = torch.LongTensor(
                get_feat_batch(T, i, batch_size)
            )
            C_mb = torch.FloatTensor(
                get_feat_batch(labels, i, batch_size)
            )
            Z_mb = torch.rand(
                (T_mb.shape[-1], args.max_seq_len, args.Z_dim)
            ).to(args.device)

            generated_data.append(
                model.generate(Z=Z_mb, T=T_mb, C=C_mb).cpu().numpy()
            )

    generated_data = np.vstack(generated_data)[:len(T)]
    generated_labels = get_labels(generated_data, T)
    generated_time = T

    return generated_data, generated_labels, generated_time
