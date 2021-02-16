import torch

from models.rgan import (
    RGANDiscriminator, 
    RGANGenerator, 
    RGAN
)

class RCGANGenerator(RGANGenerator):
    def __init__(self, args):
        super(RCGANGenerator, self).__init__(args)
        # The model architecture is basically similar to that of RGAN
        # The difference is on the input dimension
        if args.C_dim is not None:
            self.input_dim = args.Z_dim + args.C_dim

        # Override RNN layer
        self.gen_rnn = torch.nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def forward(self, Z, T, C):
        # Append conditional vector to noise vector
        # (B x S x Z) -> (B x S x Z+C)
        cond_dim = C.shape[-1]
        repeated_encoding = torch.stack(
            [C] * self.max_seq_len,
            axis=1
        ).to(self.device)
        Z = torch.cat([Z, repeated_encoding], axis=2)

        # Run forward pass
        X_hat = super(RCGANGenerator, self).forward(Z, T)

        return X_hat

class RCGANDiscriminator(RGANDiscriminator):
    def __init__(self, args):
        super(RCGANDiscriminator, self).__init__(args)
        # The model architecture is basically similar to that of RGAN
        # Since RGAN also views labels as one of the features, the inputs are
        # basically the same for RGAN and RCGAN
        
    def forward(self, X, T, C):
        # Append conditional vector to noise vector
        # (B x S x X) -> (B x S x X+C)
        repeated_encoding = torch.stack(
            [C]*self.max_seq_len,
            axis=1
        ).to(self.device)
        X = torch.cat([X, repeated_encoding], axis=2)

        # Run forward pass
        logits = super(RCGANDiscriminator, self).forward(X, T)
        return logits
        
class RCGAN(RGAN):
    def __init__(self, args, moments_loss=True):
        # TODO: Different losses and argument/inputs
        super(RCGAN, self).__init__(args)
        self.generator = RCGANGenerator(args)
        self.discriminator = RCGANDiscriminator(args)
        self.moments_loss = moments_loss

    def forward(self, X, Z, T, C, C_wrong=None):
        # Generate fake data (B x S x F+C)
        X_hat = self.generator(Z, T, C)
        # Discriminator input does not include labels
        X_hat = X_hat[:, :, :-1]    # remove last column
        # Discriminator prediction over real and fake data
        D_real = self.discriminator(X, T, C)
        D_fake = self.discriminator(X_hat, T, C)
        
        ## Discriminator loss (step-wise)
        D_loss_real = torch.nn.functional.binary_cross_entropy_with_logits(
            D_real, 
            torch.ones_like(D_real)
        ).mean()
        D_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(
            D_fake, 
            torch.zeros_like(D_fake)
        ).mean()

        D_loss = D_loss_real + D_loss_fake

        if C_wrong is not None:
            # Discriminator predict fake conditionals
            C_fake = self.discriminator(X, T, C_wrong)
            D_loss_cond = torch.nn.functional.binary_cross_entropy_with_logits(
                C_fake, 
                torch.zeros_like(D_real)
            ).mean()
            D_loss = D_loss + D_loss_cond

        ## Generator loss (step-wise)
        G_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            D_fake, 
            torch.ones_like(D_fake)
        ).mean()
        
        return D_loss, G_loss

    def generate(self, Z, T, C):
        X_hat = self.generator(Z, T, C)
        return X_hat