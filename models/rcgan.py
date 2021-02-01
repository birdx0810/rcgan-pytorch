import torch

class Generator(torch.nn.Module):
    """The generator network for RCGAN
    """
    def __init__(self, args):
        super(Generator, self).__init__()
        self.device = args.device
        self.Z_dim = args.Z_dim
        self.C_dim = args.C_dim
        self.feature_dim = args.feature_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Set input dim for model (w/ or w/o C)
        if self.C_dim:
            self.input_dim = self.Z_dim + self.C_dim
        else:
            self.input_dim = self.Z_dim

        # Generator Architecture
        self.gen_rnn = torch.nn.LSTM(
            input_size=self.input_dim, 
            hidden_size=self.feature_dim, 
            num_layers=self.num_layers, 
            batch_first=True
        )
        self.gen_linear = torch.nn.Linear(self.feature_dim, self.feature_dim)
        self.gen_sigmoid = torch.nn.Sigmoid()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference: 
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.gen_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.gen_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, Z, T, C=None):
        """Takes in random noise (features) and generates synthetic features within the latent space
        Args:
            - Z: input random noise (B x S x Z)
            - T: input temporal information (B)
            - C: input conditional vector (B x C)
        Returns:
            - X_hat: feature space synthetic data (B x S x F)
        """
        if C is not None:
            # Conditional parameters
            _, cond_dim = C.shape

            # Append conditional vector to noise vector
            # (B x S x Z -> B x S x Z+C)
            repeated_encoding = torch.stack(
                [C]*self.max_seq_len, 
                axis=1
            ).to(self.device)
            Z = torch.cat([Z, repeated_encoding], axis=2)

        # Dynamic RNN input for ignoring paddings
        Z_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=Z, 
            lengths=T, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # 128 x 100 x 71
        H_packed, H_t = self.gen_rnn(Z_packed)
        
        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_packed, 
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )


        # # 128 x 100 x 71
        # H_o, H_t = self.gen_rnn(Z)

        # 128 x 100 x 71
        logits = self.gen_linear(H_o)
        # B x S x F
        X_hat = self.gen_sigmoid(logits)
        return X_hat

class Discriminator(torch.nn.Module):
    """The discriminator network for RCGAN
    """
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.device = args.device
        self.feature_dim = args.feature_dim
        self.C_dim = args.C_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Set input dim for model (w/ or w/o C)
        if self.C_dim is not None:
            self.input_dim = self.feature_dim + self.C_dim
        else:
            self.input_dim = self.feature_dim

        # Discriminator Architecture
        self.dis_rnn = torch.nn.LSTM(
            input_size=self.input_dim, 
            hidden_size=self.feature_dim, 
            num_layers=self.num_layers, 
            batch_first=True
        )
        self.dis_linear = torch.nn.Linear(self.feature_dim, 1)

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference: 
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.dis_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.dis_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, X, T, C=None):
        """Forward pass for predicting if the data is real or synthetic
        Args:
            - X: latent representation (B x S x F)
            - T: input temporal information (B)
            - C: input conditional vector (B x C)
        Returns:
            - logits: predicted logits (B x S)
        """
        if C is not None:
            # Conditional parameters
            _, cond_dim = C.shape

            # Append conditional vector to noise vector
            # (B x S x Z -> B x S x Z+C)
            repeated_encoding = torch.stack(
                [C]*self.max_seq_len, 
                axis=1
            ).to(self.device)
            X = torch.cat([X, repeated_encoding], axis=2)

        # Dynamic RNN input for ignoring paddings
        X_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=X, 
            lengths=T, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # 128 x 100 x 10
        H_packed, H_t = self.dis_rnn(X_packed)
        
        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_packed, 
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        # # 128 x 100 x 10
        # H_o, H_t = self.dis_rnn(X)
        
        # 128 x 100
        logits = self.dis_linear(H_o).squeeze(-1)
        return logits

class RCGAN(torch.nn.Module):
    """Recurrent (Conditional) GAN as proposed by Esteban et al., 2017
    Reference:

    - https://github.com/ratschlab/RGAN
    """
    def __init__(self, args):
        super(RCGAN, self).__init__()
        self.generator = Generator(args)
        self.discriminator = Discriminator(args)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, X, Z, T, CG=None, CD=None, CS=None):
        """The forward pass for training the whole RCGAN

        Args:
        - X (torch.FloatTensor): real sequential data with shape (B x S x F)
        - Z (torch.FloatTensor): sampled noise vector with shape (B x S x Z)
        - T (torch.LongTensor): the sequence length of each data (B)
        - CG (torch.LongTensor): the conditional vector (true) for generated
                                 data (B x C) 
        - CD (torch.LongTensor): the conditional vector (label) for real data 
                                 (B x C) 
        - CS (torch.LongTensor): the conditional vector (wrong) for generated
                                 data (B x C) 
        """
        if CG is None and CD is not None:
            raise ValueError(f"")

        # Fake data
        X_hat = self.generator(Z, T, CG)
        D_fake = self.discriminator(X_hat, T, CG)
        # Real data
        D_real = self.discriminator(X, T, CD)

        # Discriminator loss
        D_loss_real = self.criterion(D_real, torch.ones_like(D_real)).mean()
        D_loss_fake = self.criterion(D_fake, torch.zeros_like(D_fake)).mean()

        D_loss = D_loss_real + D_loss_fake

        if CS:
            # Discriminator predict fake conditionals
            C_fake = discriminator(X, T, CS)
            D_loss_cond = self.criterion(C_fake, torch.zeros_like(D_real)).mean()
            D_loss = D_loss + D_loss_cond

        # Generator loss
        G_loss = self.criterion(D_fake, torch.ones_like(D_fake)).mean()

        return D_loss, G_loss

    def generate(self, Z, T, C=None):
        X_hat = self.generator(Z, T, C)
        return X_hat
