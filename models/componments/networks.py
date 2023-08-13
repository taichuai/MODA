import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.nn.functional import softmax
from torch.nn.init import kaiming_normal_
from torch.nn.parallel import DistributedDataParallel as DDP


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>



def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], useDDP=False):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if type(gpu_ids) is list and len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if useDDP:
            net = net().to(gpu_ids)
            net = DDP(net, device_ids=gpu_ids)  # DDP
            print(f'use DDP to apply models on {gpu_ids}')
        else:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.n_epochs) / \
                float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_rule, last_epoch=opt.epoch_count-2)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=opt.gamma, last_epoch=opt.epoch_count-2)
        for _ in range(opt.epoch_count-2):
            scheduler.step()
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


class ConvSeqMLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, kz) -> None:
        super().__init__()
        self.temporal_net = nn.Sequential(
            nn.Conv1d(dim_in, dim_hidden, kernel_size=kz, padding=kz//2), nn.BatchNorm1d(dim_hidden), nn.LeakyReLU(0.2), nn.LeakyReLU(0.2), 
            nn.Conv1d(dim_in, dim_hidden, kernel_size=kz, padding=kz//2), nn.BatchNorm1d(dim_hidden), nn.LeakyReLU(0.2), nn.LeakyReLU(0.2), 
            nn.Conv1d(dim_in, dim_hidden, kernel_size=kz, padding=kz//2), nn.BatchNorm1d(dim_hidden), nn.LeakyReLU(0.2),
            )
        self.formatter = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), nn.LeakyReLU(0.2), nn.Linear(dim_hidden, dim_out)
        )

    def forward(self, x_in):
        assert len(x_in.shape) > 3
        x = self.temporal_net(x_in)
        bs, seq_len = x.shape[:2]
        x = x.reshape(bs * seq_len, -1)
        return self.layers(x).reshape(bs, seq_len, -1)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, n_layers, with_act=True, with_norm=False) -> None:
        super().__init__()
        assert n_layers > 1
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(dim_in, dim_hidden))
            elif i == (n_layers - 1):
                layers.append(nn.Linear(dim_hidden, dim_out))
                break
            else:
                layers.append(nn.Linear(dim_hidden, dim_hidden))
            if with_norm: layers.append(nn.BatchNorm1d(dim_hidden))
            if with_act: layers.append(nn.LeakyReLU(0.2))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x_in):
        if len(x_in.shape) > 2:
            bs, seq_len = x_in.shape[:2]
            x = x_in.reshape(bs * seq_len, -1)
            return self.layers(x).reshape(bs, seq_len, -1)
        else:
            return self.layers(x_in)


class ClassifierBasedEstimator(nn.Module):
    def __init__(self, hidden_feature_dim=128, drop_p=0.5, out_heads_range=[150]*3+[256]*3+[5]) -> None:
        super().__init__()
        self.hidden_feature_dim = hidden_feature_dim
        self.out_heads_range = out_heads_range      # todo: check here, set proper range
        self.act_func = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(drop_p)
        self.fc_formatter = nn.Linear(hidden_feature_dim, hidden_feature_dim)
        self.out_heads = nn.ModuleList([nn.Linear(hidden_feature_dim, x//3 ) for x in self.out_heads_range])
        self.register_buffer('idx_tensors', torch.cat([torch.from_numpy(np.array([idx for idx in range(x//3)])).float()[None, ...] for x in self.out_heads_range], dim=0))
        
    def forward(self, x_in, full_info=False):
        if len(x_in.shape) > 2:
            bs, n_seq = x_in.shape[:2]
            x = x_in.reshape(bs*n_seq, -1)
        else:
            x = x_in
        base_out = self.fc_formatter(x)
        base_out = torch.flatten(base_out, start_dim=1)
        output = self.drop(base_out)

        heads = [self.softmax(head_decoder(output)) for head_decoder in self.out_heads]

        pred_heads = [torch.sum(heads[i] * self.idx_tensors[i], dim=1) * 3 - self.out_heads_range[i] / 2 for i in range(len(self.out_heads_range))]

        preds = torch.cat([x[..., None] for x in pred_heads], dim=1)

        if full_info:
            return preds, heads
        else:
            return preds


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim_q_k):
        """Scaled Dot-Product Attention model: :math:`softmax(QK^T/sqrt(dim))V`.
        Args:
            dim_q_k (int): dimension of `queries` and `keys`.
        Inputs: query, key, value, mask
            - **value** of shape `(batch, seq_len, dim_v)`:  a float tensor containing `value`.
            - **key** of shape `(batch, seq_len, dim_q_k)`: a float tensor containing `key`.
            - **query** of shape `(batch, q_len, dim_q_k)`: a float tensor containing `query`.
            - **mask** of shape `(batch, q_len, seq_len)`, default None: a byte tensor containing mask for
              illegal connections between query and value.
        Outputs: attention, attention_weights
            - **attention** of shape `(batch, q_len, dim_v)` a float tensor containing attention
              along `query` and `value` with the corresponding `key`.
            - **attention_weights** of shape `(batch, q_len, seq_len)`: a float tensor containing distribution of
              attention weights.
        """
        super(ScaledDotProductAttention, self).__init__()

        self.scale_factor = np.power(dim_q_k, -0.5)

    def forward(self, value, key, query, mask=None):
        # (batch, q_len, seq_len)
        adjacency = query.bmm(key.transpose(1, 2)) * self.scale_factor

        if mask is not None:
            adjacency.data.masked_fill_(mask.data, -float('inf'))

        attention = softmax(adjacency, 2)
        return attention.bmm(value), attention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim_m, dim_q_k, dim_v, dropout=0.1):
        """Multi-Head Attention model.
        Args:
            n_heads (int): number of heads.
            dim_m (int): hidden size of model.
            dim_q_k (int): dimension of projection `queries` and `keys`.
            dim_v (int): dimension of projection `values`.
            dropout (float, optional): dropout probability.
        Inputs:
            - **value** of shape `(batch, seq_len, dim_m)`: a float tensor containing `value`.
            - **key** of shape `(batch, seq_len, dim_m)`: a float tensor containing `key`.
            - **query** of shape `(batch, q_len, dim_m)`: a float tensor containing `query`.
            - **mask** of shape `(batch, q_len, seq_len)`: default None: a byte tensor containing mask for
              illegal connections between query and value.
        Outputs:
            - **attention** of shape `(batch, q_len, dim_m)`: a float tensor containing attention
              along `query` and `value` with the corresponding `key` using Multi-Head Attention mechanism.
        """
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.dim_m = dim_m
        self.dim_q_k = dim_q_k
        self.dim_v = dim_v

        self.query_projection = nn.Parameter(
            torch.FloatTensor(n_heads, dim_m, dim_q_k))
        self.key_projection = nn.Parameter(
            torch.FloatTensor(n_heads, dim_m, dim_q_k))
        self.value_projection = nn.Parameter(
            torch.FloatTensor(n_heads, dim_m, dim_v))
        self.attention = ScaledDotProductAttention(dim_q_k)
        self.output = nn.Linear(dim_v * n_heads, dim_m)
        self.dropout = nn.Dropout(dropout)
        self.layer_normalization = nn.LayerNorm(dim_m, eps=1e-12)

        # Initialize projection tensors
        for parameter in [
                self.query_projection, self.key_projection,
                self.value_projection
        ]:
            kaiming_normal_(parameter.data)

    def forward(self, value, key, query, mask=None):
        seq_len = key.shape[1]
        q_len = query.shape[1]
        batch_size = query.shape[0]

        residual = query
        # (batch, x, dim_m) -> (n_heads, batch * x, dim_m)
        value, key, query = map(self.stack_heads, [value, key, query])

        if mask is not None:
            mask = self.stack_mask(mask)

        # (n_heads, batch * x, dim_m) -> (n_heads, batch * x, projection) -> (n_heads * batch, x, projection)
        # where `projection` is `dim_q_k`, `dim_v` for each input respectively.
        value = value.bmm(self.value_projection).view(-1, seq_len, self.dim_v)
        key = key.bmm(self.key_projection).view(-1, seq_len, self.dim_q_k)
        query = query.bmm(self.query_projection).view(-1, q_len, self.dim_q_k)

        # (n_heads * batch, q_len, dim_v)
        context, _ = self.attention(value, key, query, mask)

        # # (n_heads * batch, q_len, dim_v) -> (batch * q_len, n_heads, dim_v) -> (batch, q_len, n_heads * dim_v)
        # context = context.view(self.n_heads, -1, self.dim_v).transpose(0, 1).view(-1, q_len, self.n_heads * self.dim_v)

        # (n_heads * batch, q_len, dim_v) -> (batch, q_len, n_heads * dim_v)
        context_heads = context.split(batch_size, dim=0)
        concat_heads = torch.cat(context_heads, dim=-1)

        # (batch, q_len, n_heads * dim_v) -> (batch, q_len, dim_m)
        out = self.output(concat_heads)
        out = self.dropout(out)

        return self.layer_normalization(out + residual)

    def stack_mask(self, mask):
        """Prepare mask tensor for multi-head Scaled Dot-Product Attention.
        Args:
            mask: A byte tensor of shape `(batch, q_len, seq_len)`.
        Returns:
            A byte tensor of shape `(n_heads * batch, q_len, seq_len)`.
        """
        return mask.repeat(self.n_heads, 1, 1)

    def stack_heads(self, tensor):
        """Prepare tensor for multi-head projection.
        Args:
            tensor: A float input tensor of shape `(batch, x, dim_m)`.
        Returns:
            Stacked input tensor n_head times of shape `(n_heads, batch * x, dim_m)`.
        """
        return tensor.view(-1, self.dim_m).repeat(self.n_heads, 1, 1)


class PositionWise(nn.Module):
    def __init__(self, dim_m, dim_i, dropout=0.1):
        """Position-wise Feed-Forward Network.
        Args:
            dim_m (int): input and output dimension.
            dim_i (int): inner dimension.
            dropout (float, optional): dropout probability.
        Inputs:
            - **input** of shape `(batch, *, dim_m)`: a float tensor.
        Outputs:
            - **output** of shape `(batch, *, dim_m)`: a float tensor.
        """
        super(PositionWise, self).__init__()

        self.feedforward = nn.Sequential(
            nn.Linear(dim_m, dim_i), nn.ReLU(), nn.Linear(dim_i, dim_m),
            nn.Dropout(dropout))
        self.normalization = nn.LayerNorm(dim_m, eps=1e-12)

    def forward(self, input):
        # There's nothing difficult here.
        residual = input
        output = self.feedforward(input)
        output = self.normalization(output + residual)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_m, dim_q_k, dim_v, n_heads, dim_i, dropout):
        """Transformer encoder layer.
        Args:
            dim_m (int): Dimension of model.
            dim_q_k (int): Dimension of `query` & `key` attention projections.
            dim_v (int): Dimension of `value` attention projection.
            n_heads (int): Number of attention heads.
            dim_i (int): Inner dimension of feed-forward position-wise sublayer.
            dropout (float): Dropout probability.
        Inputs:
            - **input** of shape `(batch, enc_seq_len, dim_m)`, a float tensor, where `batch` is batch size,
              `enc_seq_len` is length of encoder sequence for this batch and `dim_m` is hidden size of model.
              Input embedding has `dim_m` size too.
        Outputs:
            - **output** of shape `(batch, seq_len, dim_m)`, a float tensor.
        """
        super(TransformerEncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(n_heads, dim_m, dim_q_k, dim_v,
                                            dropout)
        self.positionwise = PositionWise(dim_m, dim_i, dropout)

    def forward(self, input):
        enc_att = self.attention(input, input, input)
        output = self.positionwise(enc_att)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim_m, dim_q_k, dim_v, n_heads, dim_i, dropout):
        """Transformer decoder layer.
        Args:
            dim_m (int): Dimension of model.
            dim_q_k (int): Dimension of `query` & `key` attention projections.
            dim_v (int): Dimension of `value` attention projection.
            n_heads (int): Number of attention heads.
            dim_i (int): Inner dimension of feed-forward position-wise sublayer.
            dropout (float): Dropout probability.
        Inputs:
            - **input** of shape `(batch, dec_seq_len, dim_m)`, a float tensor, where `batch` is batch size,
              `dec_seq_len` is length of decoder sequence for this batch and `dim_m` is hidden size of model.
              Input embedding has `dim_m` size too.
            - **encoder_output** of shape `(batch, enc_seq_len, dim_m)`, a float tensor, where `enc_seq_len` is length
              of encoder sequence.
            - **mask** of shape `(batch, dec_seq_len, dec_sec_len)`, a byte tensor containing mask for
              illegal connections between encoder and decoder sequence tokens. It's used to preserving
              the auto-regressive property.
        Outputs:
            - **output** of shape `(batch, dec_seq_len, dim_m)`, a float tensor.
        """
        super(TransformerDecoderLayer, self).__init__()

        self.masked_attention = MultiHeadAttention(n_heads, dim_m, dim_q_k,
                                                   dim_v, dropout)
        self.attention = MultiHeadAttention(n_heads, dim_m, dim_q_k, dim_v,
                                            dropout)
        self.positionwise = PositionWise(dim_m, dim_i, dropout)

    def forward(self, input, encoder_output, mask):
        dec_att = self.masked_attention(input, input, input, mask)
        adj_att = self.attention(
            value=encoder_output, key=encoder_output, query=dec_att)
        output = self.positionwise(adj_att)

        return output


# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=30, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=600):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.div(torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1), period, rounding_mode='floor')
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


def init_blind_future_mask(max_seq_len):
    m_zeros = torch.zeros(max_seq_len, max_seq_len)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + m_zeros
    return mask


class TemporalAlignedBlock(nn.Module):
    def __init__(self, dim_in, dim_hidden, n_layers, period=30, max_seq_len=600) -> None:
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=4, dim_feedforward=dim_hidden, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.ppe   = PeriodicPositionalEncoding(dim_in, period=period, max_seq_len=max_seq_len)     # blind further feature effect on history
        biased_mask = init_biased_mask(n_head=1, max_seq_len=max_seq_len, period=period)
        memory_mask = (1 - torch.eye(max_seq_len)) == 1
        self.register_buffer('biased_mask', biased_mask)
        self.register_buffer('memory_mask', memory_mask)
    
    def forward(self, xa_in, xs_in):
        emb_with_pos = self.ppe(torch.stack([xs_in]*xa_in.shape[1], dim=1))
        tgt_mask = self.biased_mask[:, :emb_with_pos.shape[1], :emb_with_pos.shape[1]].clone().detach()
        tgt_mask = torch.cat([tgt_mask]*emb_with_pos.shape[0]*4, dim=0)
        memory_mask = self.memory_mask[:emb_with_pos.shape[1], :xa_in.shape[1]].clone().detach()
        x = self.decoder(emb_with_pos, xa_in, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return x


class TemporalVAEBlock(nn.Module):
    def __init__(self, dim_in, dim_hidden, n_layers, n_heads=4, max_seq_len=600) -> None:
        super().__init__()

        # Transformer
        self.embedding = PositionalEncoding(dim_in, max_len=max_seq_len)
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_in, nhead=4, dim_feedforward=dim_hidden, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=4, dim_feedforward=dim_hidden, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.out = nn.Sequential(
            nn.Linear(dim_in, dim_hidden)
        )
        
        # VAE
        self.to_mu = nn.Linear(dim_in, dim_hidden)
        self.to_logvar = nn.Linear(dim_in, dim_hidden)
        self.decode_latent = nn.Linear(dim_hidden, dim_in)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    @staticmethod
    def autoregressive_mask(tensor):
        """Generate auto - regressive mask for tensor. It's used to preserving the auto - regressive property.
        Args:
            tensor(torch.Tensor): of shape ``(batch, seq_len)``.
        Returns:
            torch.Tensor: a byte mask tensor of shape ``(batch, seq_len, seq_len)`` containing mask for
            illegal attention connections between decoder sequence tokens.
        """
        batch_size, seq_len = tensor.shape[:2]
        x = torch.ones(seq_len, seq_len, device=tensor.device).tril(-1).transpose(0, 1)

        return x.repeat(batch_size, 1, 1).float()

    def forward(self, xa_in: torch.Tensor, xs_in: torch.Tensor):
        input = torch.stack([xs_in]*xa_in.shape[1], dim=1)
        input_embedded = self.embedding(input)

        encoder_state = self.encoder(input_embedded + xa_in)

        # Reparameterize
        mu = self.to_mu(encoder_state)
        logvar = self.to_logvar(encoder_state)
        z = self.reparameterize(mu, logvar)

        encoder_state = self.decode_latent(z)

        # Decode
        mask = self.autoregressive_mask(input)
        mask = torch.cat([mask]*4, dim=0)
        decoder_state = input_embedded + xa_in
        decoder_state = self.decoder(decoder_state, encoder_state, mask)

        output = self.out(decoder_state)            # [:, :-1, :]
        return output.contiguous(), mu, logvar


class DualTemporalMoudleV2(nn.Module):
    def __init__(self, dim_in, dim_hidden, n_layers, dropout=0, type='LSTM', period=30, seq_len=600) -> None:
        super().__init__()
        self.decoder_type = type
        if self.decoder_type == 'RNN':
            self.short_layer = nn.GRU(
                input_size=dim_in,
                hidden_size=dim_hidden,
                num_layers=n_layers,
                dropout=dropout,
                bidirectional=False,
                batch_first=True)
            self.long_layer = nn.LSTM(
                input_size=dim_in,
                hidden_size=dim_hidden,
                num_layers=n_layers,
                dropout=dropout,
                bidirectional=False,
                batch_first=True)
        elif self.decoder_type == 'FormerHybrid':
            self.short_layer = TemporalAlignedBlock(dim_in=dim_in, dim_hidden=dim_hidden, n_layers=n_layers, period=period, max_seq_len=seq_len)
            self.long_layer  = TemporalVAEBlock(dim_in=dim_in, dim_hidden=dim_hidden, n_layers=n_layers, max_seq_len=seq_len)
        elif self.decoder_type == 'Identity':
            self.short_layer = nn.Identity()
            self.long_layer  = nn.Identity()
        else:
            print(f'{type} is not defined.')
            raise NotImplementedError(f'{self.type} is not defined.')
    
    def forward(self, x_a_in, x_s_in):
        if self.decoder_type == 'RNN':
            x = x_a_in + x_s_in[:, None, ...]
            n_inter = 4
            a_len = x.shape[1] // n_inter

            x_shorts = []
            h_short = None
            for i in range(n_inter):
                xt, h_short = self.short_layer(x[:, i*a_len:(i+1)*a_len, ...], h_short)
                x_shorts.append(xt)

            x_short = torch.cat(x_shorts, 1)
            x_long, _  = self.long_layer(x)
            return torch.concat([x_short, x_long], dim=-1), None, None
        elif self.decoder_type == 'FormerHybrid':
            x_short            = self.short_layer(x_a_in, x_s_in)
            x_long, mu, logvar = self.long_layer(x_a_in, x_s_in)
            return torch.concat([x_short, x_long], dim=-1), mu, logvar
        elif self.decoder_type == 'Identity':
            x = x_a_in + x_s_in[:, None, ...]
            x_short = self.short_layer(x)
            x_long  = self.long_layer(x)
            return torch.concat([x_short, x_long], dim=-1), None, None
        else:
            raise NotImplementedError


class RenderGenerator_normal(nn.Module):
    def __init__(self, input_nc=4, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        #import ipdb; ipdb.set_trace()
        super(RenderGenerator_normal, self).__init__()
        # construct unet structure
        unet_block = ResUnetSkipConnectionBlock_small(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                                      innermost=True)

        for i in range(num_downs - 5):
            unet_block = ResUnetSkipConnectionBlock_small(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                          norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = ResUnetSkipConnectionBlock_small(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                      norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock_small(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                      norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock_small(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                      norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock_small(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                                      norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        #import ipdb; ipdb.set_trace()
        output = self.model(input)
        output = torch.tanh(output)   # scale to [-1, 1]

        return output


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class ResUnetSkipConnectionBlock_small(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetSkipConnectionBlock_small, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1, bias=use_bias)
        # add two resblock
        res_downconv = [ResidualBlock(inner_nc, norm_layer)]
        res_upconv = [ResidualBlock(outer_nc, norm_layer)]

        # res_downconv = [ResidualBlock(inner_nc)]
        # res_upconv = [ResidualBlock(outer_nc)]

        downrelu = nn.ReLU(True)
        uprelu = nn.ReLU(True)
        if norm_layer != None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc,
                               kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            # up = [uprelu, upsample, upconv, upnorm]
            up = [upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3,
                               stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            if norm_layer == None:
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                up = [upsample, upconv, upnorm, uprelu] + res_upconv
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc,
                               kernel_size=3, stride=1, padding=1, bias=use_bias)
            if norm_layer == None:
                down = [downconv, downrelu] + res_downconv
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                down = [downconv, downnorm, downrelu] + res_downconv
                up = [upsample, upconv, upnorm, uprelu] + res_upconv

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class RenderGenerator_large(nn.Module):
    def __init__(self, input_nc=4, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(RenderGenerator_large, self).__init__()
        # construct unet structure
        unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                                innermost=True)

        for i in range(num_downs - 5):
            unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = ResUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                                norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        output = self.model(input)
        output = torch.tanh(output)   # scale to [-1, 1]

        return output


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class ResUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1, bias=use_bias)
        # add two resblock
        res_downconv = [ResidualBlock(
            inner_nc, norm_layer), ResidualBlock(inner_nc, norm_layer)]
        res_upconv = [ResidualBlock(
            outer_nc, norm_layer), ResidualBlock(outer_nc, norm_layer)]

        # res_downconv = [ResidualBlock(inner_nc)]
        # res_upconv = [ResidualBlock(outer_nc)]

        downrelu = nn.ReLU(True)
        uprelu = nn.ReLU(True)
        if norm_layer != None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc,
                               kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            # up = [uprelu, upsample, upconv, upnorm]
            up = [upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3,
                               stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            if norm_layer == None:
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                up = [upsample, upconv, upnorm, uprelu] + res_upconv
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc,
                               kernel_size=3, stride=1, padding=1, bias=use_bias)
            if norm_layer == None:
                down = [downconv, downrelu] + res_downconv
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                down = [downconv, downnorm, downrelu] + res_downconv
                up = [upsample, upconv, upnorm, uprelu] + res_upconv

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# UNet with residual blocks
class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, norm_layer=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU(True)
        if norm_layer == None:
            # hard to converge with out batch or instance norm
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features)
            )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out
        # return self.relu(x + self.block(x))


class RenderGenerator_Unet(nn.Module):
    def __init__(self, input_nc=4, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(RenderGenerator_Unet, self).__init__()

        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        # add intermediate layers with ngf * 8 filters
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        output = self.model(input)

        return output


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        ndf_max = 64

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, min(
                ndf_max, ndf*(2**(num_D-1-i))), n_layers, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer' +
                            str(j), getattr(netD, 'model'+str(j)))
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j))
                         for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                nn.BatchNorm2d(nf),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)
