import torch

class RGANGenerator(torch.nn.Module):
    def __init__(self, args):
        super(RGANGenerator, self).__init__()
        # Basic parameters
        self.device = args.device
        self.input_dim = args.Z_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.feature_dim
        self.num_layers = args.num_layers
        self.max_seq_len = args.max_seq_len
        self.padding_value = args.padding_value

        # Model architecture
        self.gen_rnn = torch.nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.gen_linear = torch.nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.output_dim
        )
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

    def forward(self, Z, T):
        """
        Args:
            - Z (torch.FloatTensor): input noise with shape (B x S x Z)
            - T (torch.LongTensor): the sequence length of the input (B)
        Returns
            - X_hat (torch.FloatTensor): the sequence output
        """
        # Dynamic RNN input for ignoring paddings
        Z_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=Z,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        # B x S x Z
        H_packed, H_t = self.gen_rnn(Z_packed)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_packed,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        # B x S x F
        logits = self.gen_linear(H_o)

        # B x S x F
        X_hat = self.gen_sigmoid(logits)

        return X_hat

class RGANDiscriminator(torch.nn.Module):
    def __init__(self, args):
        super(RGANDiscriminator, self).__init__()
        # Basic parameters
        self.device = args.device
        self.input_dim = args.feature_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = 1     # Binary classification
        self.num_layers = args.num_layers
        self.max_seq_len = args.max_seq_len
        self.padding_value = args.padding_value

        # Model architecture
        self.dis_rnn = torch.nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.dis_linear = torch.nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.output_dim,
        )

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

    def forward(self, X, T):
        """
        Args:
            - X (torch.FloatTensor): real sequential data with shape (B x S x F)
            - T (torch.LongTensor): the sequence length of the input (B)
        Returns:
            - logits (torch.FloatTensor): output logits for BCE w/ logits
                                          (B x S x 1)
        """
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

        # 128 x 100
        logits = self.dis_linear(H_o).squeeze(-1)

        return logits

class RGAN(torch.nn.Module):
    """Recurrent (Conditional) GAN as proposed by Esteban et al., 2017
    Reference:
    - https://github.com/ratschlab/RGAN
    - https://https://github.com/3778/Ward2ICU/
    """
    def __init__(self, args):
        super(RGAN, self).__init__()
        self.generator = RGANGenerator(args)
        self.discriminator = RGANDiscriminator(args)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, X, Z, T, gamma=100):
        # Generate fake data
        X_hat = self.generator(Z, T)

        # Discriminator prediction over real and fake data
        D_real = self.discriminator(X, T)
        D_fake = self.discriminator(X_hat, T)

        # Calculate loss (step-wise)
        D_loss_real = self.criterion(D_real, torch.ones_like(D_real)).mean()
        D_loss_fake = self.criterion(D_fake, torch.zeros_like(D_fake)).mean()

        D_loss = D_loss_real + D_loss_fake

        # Generator loss (step-wise)
        G_loss = self.criterion(D_fake, torch.ones_like(D_fake)).mean()

        # Moments loss
        if self.moments_loss:
            G_loss_V1 = torch.mean(torch.abs(
                    torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6) - 
                    torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)
            ))
            G_loss_V2 = torch.mean(torch.abs(
                (X_hat.mean(dim=0)) - (X.mean(dim=0))
            ))

            G_loss = G_loss + gamma * (G_loss_V1 + G_loss_V2)

        return D_loss, G_loss

    def generate(self, Z, T):
        X_hat = self.generator(Z, T)
        return X_hat
