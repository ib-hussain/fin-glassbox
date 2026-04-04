'''
Module: net.py
(Added by AI Agent)

This file contains the definition of the actual MTGNN (Multivariate Time Series Graph Neural Network) model.
It relies on layers defined in layer.py (e.g., graph_constructor, mixprop, dilated_inception, LayerNorm).
'''
from layer import *


class gtnet(nn.Module):
    """
    MTGNN (Multivariate Time Series Graph Neural Network) backbone class.
    (Added by AI Agent)

    This class stitches together temporal convolution modules, graph construction, 
    and spatial graph convolution modules to learn localized spatial-temporal correlations.
    """
    def __init__(
        self,
        gcn_true,
        buildA_true,
        gcn_depth,
        num_nodes,
        device,
        predefined_A=None,
        static_feat=None,
        dropout=0.3,
        subgraph_size=20,
        node_dim=40,
        dilation_exponential=1,
        conv_channels=32,
        residual_channels=32,
        skip_channels=64,
        end_channels=128,
        seq_length=12,
        in_dim=2,
        out_dim=12,
        layers=3,
        propalpha=0.05,
        tanhalpha=3,
        layer_norm_affline=True,
    ):
        """
        Initialize the MTGNN model.
        (Added by AI Agent)
        
        Args:
            gcn_true (bool): Whether to use the graph convolution module.
            buildA_true (bool): Whether to construct the dependency graph dynamically.
            gcn_depth (int): Max diffusion steps for the mixprop layer.
            num_nodes (int): Total number of time series nodes.
            device (torch.device or str): Computing device.
            predefined_A (tensor, optional): Pre-computed adjacency matrix if not building dynamically.
            static_feat (tensor, optional): Static node attributes.
            dropout (float): Dropout probability.
            subgraph_size (int): Number of nodes to sample for the kNN graph construction.
            node_dim (int): Dimensionality of the node embeddings for graph learning.
            dilation_exponential (int): Dilation factor growth exponent.
            conv_channels (int): Intermediate channels for convolution layers.
            residual_channels (int): Channels for residual connections.
            skip_channels (int): Channels for skip connections.
            end_channels (int): Channels prior to output projection.
            seq_length (int): Length of historical window.
            in_dim (int): Number of input features per node.
            out_dim (int): Forecasting horizon steps.
            layers (int): Number of stacked blocks (layers) in MTGNN.
            propalpha (float): Retention parameter in Information Propagation over graphs.
            tanhalpha (float): Scaling factor for the graph constructor tanh activation.
            layer_norm_affline (bool): If True, use affine transformations in layer normalization.
        """

        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        self.gc = graph_constructor(
            num_nodes,
            subgraph_size,
            node_dim,
            device,
            alpha=tanhalpha,
            static_feat=static_feat,
        )
        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(1 + (kernel_size - 1) * (dilation_exponential**layers - 1) /
                                       (dilation_exponential - 1))
        else:self.receptive_field = layers * (kernel_size - 1) + 1
        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(1 + i * (kernel_size - 1) * (dilation_exponential**layers - 1) /
                                (dilation_exponential - 1))
            else:rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size - 1) * (dilation_exponential**j - 1) /
                                    (dilation_exponential - 1))
                else:rf_size_j = rf_size_i + j * (kernel_size - 1)
                self.filter_convs.append(
                    dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels,
                                                         dilation_factor=new_dilation))
                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=conv_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    ))
                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(
                        nn.Conv2d(
                            in_channels=conv_channels,
                            out_channels=skip_channels,
                            kernel_size=(1, self.seq_length - rf_size_j + 1),
                        ))
                else:self.skip_convs.append(
                        nn.Conv2d(
                            in_channels=conv_channels,
                            out_channels=skip_channels,
                            kernel_size=(1, self.receptive_field - rf_size_j + 1),
                        ))
                if self.gcn_true:
                    self.gconv1.append(mixprop(
                        conv_channels,
                        residual_channels,
                        gcn_depth,
                        dropout,
                        propalpha,
                    ))
                    self.gconv2.append(mixprop(
                        conv_channels,
                        residual_channels,
                        gcn_depth,
                        dropout,
                        propalpha,
                    ))
                if self.seq_length > self.receptive_field:
                    self.norm.append(
                        LayerNorm(
                            (
                                residual_channels,
                                num_nodes,
                                self.seq_length - rf_size_j + 1,
                            ),
                            elementwise_affine=layer_norm_affline,
                        ))
                else:
                    self.norm.append(
                        LayerNorm(
                            (
                                residual_channels,
                                num_nodes,
                                self.receptive_field - rf_size_j + 1,
                            ),
                            elementwise_affine=layer_norm_affline,
                        ))
                new_dilation *= dilation_exponential
        self.layers = layers
        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self.seq_length),
                bias=True,
            )
            self.skipE = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, self.seq_length - self.receptive_field + 1),
                bias=True,
            )
        else:
            self.skip0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self.receptive_field),
                bias=True,
            )
            self.skipE = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, 1),
                bias=True,
            )
        self.idx = torch.arange(self.num_nodes).to(device)
    def forward(self, input, idx=None):
        """
        Forward pass for computing graph and spatio-temporal features.
        (Added by AI Agent)

        Args:
            input (tensor): Shape (batch_size, in_dim, num_nodes, seq_length).
            idx (tensor, optional): Node indices to process, used in graph construction.

        Returns:
            x (tensor): Forecasts of shape (batch, out_dim, num_nodes, 1) or similar depending on the end_conv_2 mapping.
        """
        seq_len = input.size(3)
        assert (seq_len == self.seq_length), "input sequence length not equal to preset sequence length"
        if self.seq_length < self.receptive_field:input = nn.functional.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))
        if self.gcn_true:
            if self.buildA_true: 
                if idx is None:adp = self.gc(self.idx) 
                else:adp = self.gc(idx)
            else:adp = self.predefined_A
        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
