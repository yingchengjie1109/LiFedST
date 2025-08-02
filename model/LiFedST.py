import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------- temporal module ----------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, L, ...]
        return self.pe[:, :x.size(1)]

class EnhancedLinearBlock(nn.Module):
    def __init__(self, in_len, out_len, period_len):
        super(EnhancedLinearBlock, self).__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.period_len = 8
        # in_channels=1, out_channels=1
        kernel = 1 + 2 * (period_len // 2)
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel,
            stride=1,
            padding=period_len//2,
            padding_mode='zeros',
            bias=False
        )
        self.seg_num_x = in_len // period_len
        self.seg_num_y = out_len // period_len
        self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)

    def forward(self, x_seq):
        # x_seq: [L, B*N, D]
        L, BN, D = x_seq.shape
        # [BN*D, L]
        flat = x_seq.permute(1, 2, 0).reshape(-1, L)
        y = self.conv1d(flat.unsqueeze(1)).squeeze(1) + flat
        # [BN*D, seg_num_x, period_len]
        y = y.reshape(-1, self.seg_num_x, self.period_len)
        # [BN*D, period_len, seg_num_x] -> [BN*D, period_len, seg_num_y]
        y = y.permute(0, 2, 1)
        y = self.linear(y)
        # [BN*D, seg_num_y, period_len] -> [BN*D, out_len]
        y = y.permute(0, 2, 1).reshape(-1, self.out_len)
        # [BN*D, out_len] -> [out_len, BN, D]
        return y.reshape(BN, D, self.out_len).permute(2, 0, 1)

class LTE_layer(nn.Module):
    def __init__(
        self,
        args,
        node_num,
        lag,
        horizon,
        embed_dim,
        time_layers,
        ffn_hidden=None
    ):
        super(LTE_layer, self).__init__()
        self.args = args
        self.node_num = node_num
        self.time_layers = time_layers
        self.lag = lag
        self.horizon = horizon
        self.embed_dim = embed_dim
        self.ffn_hidden = ffn_hidden or horizon
        self.period_len = horizon

        if args.input_dim != embed_dim:
            self.input_projection = nn.Linear(args.input_dim, embed_dim)
        else:
            self.input_projection = None

        self.linear_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        for i in range(time_layers):
            in_len = lag if i == 0 else horizon
            # in_len -> horizon
            self.linear_layers.append(
                EnhancedLinearBlock(in_len, horizon, self.period_len)
            )
            # FFN: horizon -> hidden -> horizon
            ffn = nn.Sequential(
                nn.Linear(horizon, self.ffn_hidden),
                nn.ReLU(),
                nn.Linear(self.ffn_hidden, horizon)
            )
            self.ffn_layers.append(ffn)

    def forward(self, x, init_state=None, node_embeddings=None, poly_coefficients=None):
        """
        input: x - [B, T, N, input_dim]
        output: out - [B, horizon, N, embed_dim]
        """
        if self.input_projection is not None:
            x = self.input_projection(x)

        B, T, N, D = x.shape
        # batch & node: [T, B*N, D]
        x_seq = x.permute(1, 0, 2, 3).reshape(T, B * N, D)

        for i in range(self.time_layers):
            x_seq = self.linear_layers[i](x_seq)  # [horizon, B*N, D]
            # FFN:[B*N*D, horizon]
            flat = x_seq.permute(1, 2, 0).reshape(-1, self.horizon)
            out = self.ffn_layers[i](flat)  # [B*N*D, horizon]
            # [horizon, B*N, D]
            x_seq = out.reshape(B * N, D, self.horizon).permute(2, 0, 1)

        # [horizon, B*N, D] -> [B, horizon, N, D]
        out = x_seq.reshape(self.horizon, B, N, D).permute(1, 0, 2, 3)
        return out, None

    def fedavg(self):
        for blk in self.linear_layers:
            sd = blk.state_dict()
            avg_sd = {}
            for k, v in sd.items():
                gathered = self.comm_socket(v.data, self.args.device)
                avg_sd[k] = gathered / self.args.num_clients
            blk.load_state_dict(avg_sd)
        for ffn in self.ffn_layers:
            sd = ffn.state_dict()
            avg_sd = {}
            for k, v in sd.items():
                gathered = self.comm_socket(v.data, self.args.device)
                avg_sd[k] = gathered / self.args.num_clients
            ffn.load_state_dict(avg_sd)

    def comm_socket(self, msg, device=None):
        if not hasattr(self.args, 'socket') or self.args.socket is None:
            return msg
        self.args.socket.send(msg)
        received = self.args.socket.recv()
        return received.to(device) if device is not None else received

# ---------------------- spatio module----------------------
class spatio_module(nn.Module):
    def __init__(self, args, dim_in, dim_out, embed_dim, ffn_dim=128, ffn_dropout=0.1):
        super(spatio_module, self).__init__()
        self.args = args
        if args.active_mode == "adptpolu":
            # Adaptive polynomial attention parameters
            self.num_heads = args.num_heads
            self.transformer_dim = args.transformer_dim
            self.head_dim = self.transformer_dim // self.num_heads
            in_dim = args.horizon * embed_dim

            self.input_projection = nn.Linear(in_dim, self.transformer_dim, bias=False)
            self.query_projection = nn.Linear(in_dim, self.transformer_dim)
            self.key_projection = nn.Linear(in_dim, self.transformer_dim)
            self.value_projection = nn.Linear(in_dim, self.transformer_dim)
            self.output_projection = nn.Linear(self.transformer_dim, args.horizon * dim_out)

            self.weights_pool = nn.Parameter(torch.Tensor(args.horizon * dim_in, args.horizon * dim_out))
            nn.init.xavier_uniform_(self.weights_pool)

            self.K = args.act_k  # polynomial degree
            self.Pi = nn.Parameter(torch.empty(self.num_heads, self.K + 1))
            nn.init.xavier_uniform_(self.Pi)

            self.ffn = nn.Sequential(
                nn.Linear(self.transformer_dim, ffn_dim), nn.ReLU(),
                nn.Dropout(ffn_dropout), nn.Linear(ffn_dim, self.transformer_dim), nn.Dropout(ffn_dropout)
            )
            self.layer_norm = nn.LayerNorm(self.transformer_dim)
        else:
            # Graph convolution branch
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings, poly_coefficients=None):
        if self.args.active_mode == "adptpolu":
            B, N, _ = x.size()
            H, d = self.num_heads, self.head_dim
            # Project and reshape
            Q = self.query_projection(x).view(B, N, H, d).permute(0, 2, 1, 3)
            K_ = self.key_projection(x).view(B, N, H, d).permute(0, 2, 1, 3)
            V = self.value_projection(x).view(B, N, H, d).permute(0, 2, 1, 3)

            # Flatten for polynomial transform
            Q2 = Q.reshape(-1, d)
            K2 = K_.reshape(-1, d)
            device, dtype = x.device, x.dtype
            weighted_sum = torch.zeros(B, H, N, d, device=device, dtype=dtype)

            for k in range(self.K + 1):
                Qk = self.transform(k, Q2).view(B, H, N, -1)  # [B, H, N, d]
                Kk = self.transform(k, K2).view(B, H, N, -1)  # [B, H, N, d]
                Kk_T = Kk.transpose(2, 3)  # [B, H, d, N]

                f_kv = torch.einsum('bhdn,bhnc->bhdc', Kk_T, V)  # f_kv = sum_n Kk^T * V  → [B, H, d, C]

                sumN = Kk_T.sum(dim=-1, keepdim=True)  # sumN = sum over N 的 Kk^T → [B, H, d, 1]

                SAG = torch.cat([f_kv, sumN], dim=-1)  # [B, H, d, C+1]
                SAG_sum = self.comm_socket(SAG, self.args.device)  # [B, H, d, C+1]

                f_kv_sum = SAG_sum[..., :-1]  # [B,H,d,C]
                sumN_sum = SAG_sum[..., -1:]  # [B,H,d,1]

                f_kv_mean = f_kv_sum / (sumN_sum + 1e-8)  # [B, H, d_k, C]

                d_k = f_kv_mean.size(2)
                f_kv_norm = f_kv_mean / math.sqrt(d_k)  # [B, H, d_k, C]

                c_k = torch.einsum('bhnd,bhdc->bhnc', Qk, f_kv_norm)

                normalization = torch.einsum('bhnd,bhde->bhne', Qk, sumN_sum)  # [B,H,N,1]

                weight = self.Pi[:, k].view(1, H, 1, 1)
                weighted_sum = weighted_sum + c_k * weight
                del Qk, Kk, Kk_T, f_kv, sumN, SAG, SAG_sum, f_kv_sum, sumN_sum, c_k, weight

            attention_out = weighted_sum.view(B, N, -1) / math.factorial(self.K)
            proj = self.input_projection(x)

            # add&norm
            res = self.layer_norm(proj + attention_out)

            # FFN + add&norm
            res = self.layer_norm(res + self.ffn(res))

            x_g = torch.einsum('bni,io->bno', x, self.weights_pool)
            out = self.output_projection(res)
            return x_g + out
        else:
            # Spectral graph convolution branch
            E, Ht = node_embeddings, x
            if self.args.active_mode == "sprtrelu":
                E2, EH = torch.relu(E), torch.einsum('dn,bnc->bdc', E.t(), Ht)
            else:
                E2, EH = E, None
            if EH is not None:
                EH = self.comm_socket(EH, self.args.device)
                Z = Ht + torch.einsum('nd,bdc->bnc', E2, EH)
            else:
                Z = Ht
            W = torch.einsum('nd,dio->nio', E, self.weights_pool)
            b = torch.matmul(E, self.bias_pool)
            return torch.einsum('bni,nio->bno', Z, W) + b

    def comm_socket(self, msg, device=None):
        if not hasattr(self.args, 'socket') or self.args.socket is None:
            return msg
        self.args.socket.send(msg)
        recv = self.args.socket.recv()
        return recv.to(device) if device else recv

    def transform(self, k, E):
        if k == 0:
            return torch.ones(E.size(0), 1, device=E.device, dtype=E.dtype)
        if k == 1:
            return E
        result = None
        base = E
        exp = k
        while exp:
            if exp & 1:
                result = base if result is None else result * base
            base = base * base
            exp >>= 1
        return result

    def fedavg(self):
        if self.args.active_mode == "adptpolu":
            layers = [self.input_projection, self.query_projection,
                      self.key_projection, self.value_projection,
                      self.output_projection]
            for layer in layers:
                w = self.comm_socket(layer.weight.data, self.args.device) / self.args.num_clients
                layer.weight = nn.Parameter(w)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    b = self.comm_socket(layer.bias.data, self.args.device) / self.args.num_clients
                    layer.bias = nn.Parameter(b)

            for idx in [0, 3]:
                linear = self.ffn[idx]
                w = self.comm_socket(linear.weight.data, self.args.device) / self.args.num_clients
                b = self.comm_socket(linear.bias.data, self.args.device) / self.args.num_clients
                linear.weight = nn.Parameter(w)
                linear.bias = nn.Parameter(b)

            ln = self.layer_norm
            w = self.comm_socket(ln.weight.data, self.args.device) / self.args.num_clients
            b = self.comm_socket(ln.bias.data, self.args.device) / self.args.num_clients
            ln.weight = nn.Parameter(w)
            ln.bias   = nn.Parameter(b)
        else:
            mean_w = self.comm_socket(self.weights_pool.data, self.args.device) / self.args.num_clients
            mean_b = self.comm_socket(self.bias_pool.data,    self.args.device) / self.args.num_clients
            self.weights_pool = nn.Parameter(mean_w)
            self.bias_pool    = nn.Parameter(mean_b)

# ---------------------- LiFedST model ----------------------
class LiFedST(nn.Module):
    def __init__(self, args):
        super(LiFedST, self).__init__()
        self.num_layers = args.num_layers
        self.embed_dim = args.embed_dim
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.args = args

        # Persist node embeddings and poly coefficients
        self.node_embeddings = nn.Parameter(torch.randn(args.num_nodes, args.embed_dim))
        self.poly_coefficients = nn.Parameter(torch.randn(args.num_nodes, args.embed_dim))

        # Build a list of (temporal_encoder, spatial_module) pairs
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            temporal = LTE_layer(args, args.num_nodes, args.lag, args.horizon, args.embed_dim, args.time_layers)
            spatial = spatio_module(args, dim_in=args.embed_dim, dim_out=args.output_dim, embed_dim=args.embed_dim)
            self.layers.append(nn.ModuleList([temporal, spatial]))

    def forward(self, x):
        # x: [B, T, N, input_dim]
        B, T, N, D = x.shape
        out = x
        for temporal, spatial in self.layers:
            # 1. Time encoding
            temporal_out, _ = temporal(out, None, self.node_embeddings, self.poly_coefficients)
            # temporal_out: [B, horizon, N, embed_dim]

            # 2. Reshape for spatial conv
            spatial_in = temporal_out.reshape(B, N, self.horizon * self.embed_dim)

            # 3. Spatial conv
            spatial_out = spatial(spatial_in, self.node_embeddings, self.poly_coefficients)
            # spatial_out: [B, N, horizon*output_dim]

            # 4. Reshape back to [B, horizon, N, output_dim]
            out = spatial_out.reshape(B, self.horizon, N, self.output_dim)

        return out

    def fedavg(self):
        for temporal, spatial in self.layers:
            temporal.fedavg()
            spatial.fedavg()

    def comm_socket(self, msg, device=None):
        if not hasattr(self.args, 'socket') or self.args.socket is None:
            return msg
        self.args.socket.send(msg)
        if device:
            return self.args.socket.recv().to(device)
        else:
            return self.args.socket.recv()
