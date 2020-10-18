import torch

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.emb_rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.emb_linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, Z):
        B, S, Z = Z.shape
        
        H_3d, _ = self.emb_rnn(Z)
        
        H_2d = H_3d.reshape(B*S, Z)
        logits_2d = self.emb_linear(H_2d)
        
        X_2d = torch.nn.functional.tanh(logits_2d)
        X_3d = X_2d.reshape(B, S, self.hidden_dim)
        
        return X_3d

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.rec_rnn = torch.nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.rec_linear = torch.nn.Linear(hidden_dim, input_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, X):
        B, S, Z = X.shape
        X = X.reshape(B*S, Z)

        H_o, H_t = self.rec_rnn(X)
        logits = self.rec_linear(H_o)
        Y = self.sigmoid(logits)

        return Y

