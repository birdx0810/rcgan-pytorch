from typing import Dict

import numpy as np
import torch

def sample_Z(batch_size=32, seq_length=100, latent_dim=20, use_time=False):
    """
    return noise vector sampled from normal distribution with shape 
    (32, 100, 20)
    """
    sample = np.float32(np.random.normal(size=[batch_size, seq_length, latent_dim]))
    if use_time:
        print('WARNING: use_time has different semantics')
        sample[:, :, 0] = np.linspace(0, 1.0/seq_length, num=seq_length)

    return sample

def sample_C(batch_size=32, cond_dim=2, max_val=1, one_hot=False):
    """
    return an array of integers (so far we only allow integer-valued
    conditional values)
    """
    if cond_dim == 0:
        return None
    else:
        if one_hot:
            assert max_val == 1
            C = np.zeros(shape=(batch_size, cond_dim))
            labels = np.random.choice(cond_dim, batch_size)
            C[np.arange(batch_size), labels] = 1
        else:
            C = np.random.choice(max_val+1, size=(batch_size, cond_dim))
        return C

def generator_trainer(
    model: torch.nn.Module, 
    Z: torch.FloatTensor,
    C: torch.FloatTensor, 
    T: torch.LongTensor, 
    opt: torch.optim.Optimizer, 
    args: Dict, 
    writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)]=None, 
    privacy_engine: Union[opacus.PrivacyEngine, type(None)]=None
) -> float:
    # TODO
    pass

def discriminator_trainer(
    model: torch.nn.Module, 
    X: torch.FloatTensor, 
    C: torch.FloatTensor, 
    T: torch.LongTensor, 
    opt: torch.optim.Optimizer, 
    args: Dict, 
    writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)]=None, 
    privacy_engine: Union[opacus.PrivacyEngine, type(None)]=None
) -> float:
    # TODO
    pass

def rcgan_trainer(model, data, time, args):
    # Initialize TimeGAN dataset and dataloader
    dataset = RCGANDataset(data, time)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False    
    )

    model.to(args.device)

    # Initialize Optimizers
    g_opt = torch.optim.Adam(model.generator.parameters(), lr=args.learning_rate)
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=args.learning_rate)
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(f"tensorboard/{args.exp}"))

    logger = trange(
        args.sup_epochs, 
        desc=f"Epoch: 0, E_loss: 0, G_loss: 0, D_loss: 0"
    )
    
    for epoch in logger:
        for X_mb, T_mb in dataloader:
            # Discriminator forward pass
            for _ in range(d_iters):
                d_loss = discriminator_trainer(
                    model=model, 
                    X=X_mb, 
                    T=T_mb, 
                    opt=d_opt, 
                    args=args, 
                    writer=writer
                )
            # Generator forward pass
            for _ in range(g_iters):
                g_loss = generator_trainer(
                    model=model,  
                    X=X_mb, 
                    T=T_mb, 
                    opt=g_opt, 
                    args=args, 
                    writer=writer
                )

def rcgan_generator():
    # TODO
    pass
