from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F

class nconv(nn.Module):
    """
    Standard graph convolution layer performing multiplication between node features and an adjacency matrix.
    Uses einsum for efficient tensor contraction.
    """
    def __init__(self):
        super(nconv, self).__init__()
    def forward(self, x, A):
        """
        Args:
            x (torch.Tensor): Input features of shape (batch, channels, num_nodes, seq_len).
            A (torch.Tensor): Adjacency matrix of shape (num_nodes, num_nodes).
        Returns:
            torch.Tensor: Output tensor of shape (batch, channels, num_nodes, seq_len).
        """
        x = torch.einsum("ncwl,vw->ncvl", (x, A))
        return x.contiguous()
class dy_nconv(nn.Module):
    """
    Dynamic graph convolution layer for time/batch dependent adjacency matrices.
    """
    def __init__(self):
        super(dy_nconv, self).__init__()
    def forward(self, x, A):
        """
        Args:
            x (torch.Tensor): Input features of shape (batch, channels, num_nodes, seq_len).
            A (torch.Tensor): Dynamic adjacency matrix of shape (batch, num_nodes, num_nodes, seq_len).
        Returns:
            torch.Tensor: Output tensor of shape (batch, channels, num_nodes, seq_len).
        """
        x = torch.einsum("ncvl,nvwl->ncwl", (x, A))
        return x.contiguous()
class linear(nn.Module):
    """
    Linear transformation applied over the channel dimension using a 1x1 2D convolution.
    """
    def __init__(self, c_in, c_out, bias=True):
        """
        Args:
            c_in (int): Number of input channels.
            c_out (int): Number of output channels.
            bias (bool): Whether to include a learnable bias. Defaults to True.
        """
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features.
        Returns:
            torch.Tensor: Linearly transformed features.
        """
        return self.mlp(x)
class prop(nn.Module):
    """
    Information propagation module that applies graph convolution iteratively.
    """
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        """
        Args:
            c_in (int): Number of input channels.
            c_out (int): Number of output channels.
            gdep (int): Graph propagation depth (number of spatial steps).
            dropout (float): Dropout rate.
            alpha (float): Retained proportion of the original node features (residual connection).
        """
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
    def forward(self, x, adj):
        """
        Args:
            x (torch.Tensor): Input node features of shape (batch, channels, num_nodes, seq_len).
            adj (torch.Tensor): Adjacency matrix.
        Returns:
            torch.Tensor: Updated node features.
        """
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
        ho = self.mlp(h)
        return ho
class mixprop(nn.Module):
    """
    Mix-hop propagation module that concatenates information from multiple propagation steps (hops).
    """
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        """
        Args:
            c_in (int): Number of input channels.
            c_out (int): Number of output channels.
            gdep (int): Graph propagation depth (number of hops to concatenate).
            dropout (float): Dropout rate.
            alpha (float): Retained proportion of root node features.
        """
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
    def forward(self, x, adj):
        """
        Args:
            x (torch.Tensor): Input node features.
            adj (torch.Tensor): Adjacency matrix.
        Returns:
            torch.Tensor: Mixed hop node features.
        """
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho
class dy_mixprop(nn.Module):
    """
    Dynamic Mix-hop propagation module. Implements mix-hop using dynamically generated adjacency 
    matrices computed from the node features directly rather than a static adjacency graph.
    """
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        """
        Args:
            c_in (int): Input channel dimension.
            c_out (int): Output channel dimension.
            gdep (int): Graph propagation depth.
            dropout (float): Dropout probability.
            alpha (float): Information retention ratio for root node connection.
        """
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep + 1) * c_in, c_out)
        self.mlp2 = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in, c_in)
        self.lin2 = linear(c_in, c_in)
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (batch, channels, num_nodes, seq_len).
        Returns:
            torch.Tensor: Mixed dynamic hop node features.
        """
        # adj = adj + torch.eye(adj.size(0)).to(x.device)
        # d = adj.sum(1)
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2, 1), x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2, 1), dim=2)
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj0)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho1 = self.mlp1(ho)
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)
        return ho1 + ho2
class dilated_1D(nn.Module):
    """
    Applies a 1D convolution with dilation over the temporal/sequence dimension.
    Uses a 2D Conv layer with (1, kernel) size to act as 1D Conv.
    """
    def __init__(self, cin, cout, dilation_factor=2):
        """
        Args:
            cin (int): Input channels.
            cout (int): Output channels.
            dilation_factor (int): Spacing between kernel elements. Defaults to 2.
        """
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        self.tconv = nn.Conv2d(cin, cout, (1, 7), dilation=(1, dilation_factor))
    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input sequence tensor.
        Returns:
            torch.Tensor: Convolved sequence.
        """
        x = self.tconv(input)
        return x
class dilated_inception(nn.Module):
    """
    Inception-style module employing multiple parallel 1D dilated convolutions.
    """
    def __init__(self, cin, cout, dilation_factor=2):
        """
        Args:
            cin (int): Number of input channels.
            cout (int): Target number of output channels (distributed among kernels).
            dilation_factor (int): Dilation size applied to all filters. Defaults to 2.
        """
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))
    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Temporal features.
        Returns:
            torch.Tensor: Concatenated features from multiple dilated convolutions clipped to the same length.
        """
        x = []
        for i in range(len(self.kernel_set)):x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x
class graph_constructor(nn.Module):
    """
    Dynamically constructs a graph adjacency matrix from node embeddings or provided static node features.
    Learns to infer interactions and top-k neighbors among nodes.
    """
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        """
        Args:
            nnodes (int): Number of nodes in the graph.
            k (int): Number of top edges to keep for each node.
            dim (int): Dimensionality of the node embeddings.
            device (torch.device): Device on which tensors will be allocated.
            alpha (float): Scaling factor for the tanh activation. Defaults to 3.
            static_feat (torch.Tensor, optional): Precomputed static node features. Defaults to None.
        """
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)
        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
    def forward(self, idx):
        """
        Retrieves the node representations to form the adjacency matrix via similarity metric.
        Args:
            idx (torch.Tensor): Indices representing the active nodes or batch.
        Returns:
            torch.Tensor: Weighted graph adjacency matrix retaining only top-k connections.
        """
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1
        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))
        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float("0"))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj
    def fullA(self, idx):
        """
        Calculates the complete adjacency matrix without applying the top-k masking constraint.
        Args:
            idx (torch.Tensor): Indices of nodes.
        Returns:
            torch.Tensor: The full continuous unmasked adjacency matrix.
        """
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1
        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))
        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        return adj
class graph_global(nn.Module):
    """
    Simple learned global adjacency matrix bounded by ReLU.
    """
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)
    def forward(self, idx):
        """
        Args:
            idx (torch.Tensor): Node indices (ignored as graph is global).
        Returns:
            torch.Tensor: Global graph adjacency matrix.
        """
        return F.relu(self.A)
class graph_undirected(nn.Module):
    """
    Constructs an undirected graph using node embeddings or static features.
    """
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        """
        Args:
            nnodes (int): Number of nodes.
            k (int): Number of neighbors to connect.
            dim (int): Dimensionality of embeddings.
            device (torch.device): Execution device.
            alpha (float): Softmax/activation scale.
            static_feat (torch.Tensor, optional): Static node embeddings.
        """
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
    def forward(self, idx):
        """
        Args:
            idx (torch.Tensor): Node indices.
        Returns:
            torch.Tensor: Top-k masked undirected adjacency matrix.
        """
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1
        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin1(nodevec2))
        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float("0"))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj
class graph_directed(nn.Module):
    """
    Constructs a directed graph using node embeddings or static features.
    """
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        """
        Args:
            nnodes (int): Number of nodes.
            k (int): Number of connections per node.
            dim (int): Embed dim.
            device (torch.device): Execution device.
            alpha (float): Scaling factor.
            static_feat (torch.Tensor, optional): Precomputed static node features.
        """
        super(graph_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)
        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
    def forward(self, idx):
        """
        Args:
            idx (torch.Tensor): Node indices.
        Returns:
            torch.Tensor: Top-k masked directed adjacency matrix.
        """
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1
        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))
        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float("0"))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj
class LayerNorm(nn.Module):
    """
    Custom Layer Normalization module allowing subset normalization using an index.
    """
    __constants__ = ["normalized_shape", "weight", "bias", "eps", "elementwise_affine"]
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        Args:
            normalized_shape (int or list/tuple): Input shape from expected input.
            eps (float): A value added to the denominator for numerical stability.
            elementwise_affine (bool): Learnable per-element affine parameters boolean.
        """
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()
    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)
    def forward(self, input, idx):
        """
        Args:
            input (torch.Tensor): Tensor to normalize.
            idx (torch.Tensor): Indices to select specific weights/biases if needed.
        Returns:
            torch.Tensor: Normalized tensor.
        """
        if self.elementwise_affine:
            return F.layer_norm(
                input,
                tuple(input.shape[1:]),
                self.weight[:, idx, :],
                self.bias[:, idx, :],
                self.eps,
            )
        else:return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)
    def extra_repr(self):
        return ("{normalized_shape}, eps={eps}, "
                "elementwise_affine={elementwise_affine}".format(**self.__dict__))
