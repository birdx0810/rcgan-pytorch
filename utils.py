import numpy as np

def sample_Z(batch_size, max_seq_len, Z_dim):
    """Random noise sample from a Gaussian Distribution for given batch
    """
    return np.random.normal((batch_size, max_seq_len, Z_dim))

def sample_C(batch_size, C_dim, num_labels=1, one_hot=False):
    """Generate conditional values (labels) for input noise of given batch
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
