""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)
           

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, activation=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)
        self.activation = activation
    
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        if self.activation:
            output = self.activation(output)
        return output
    

class GCN_E(nn.Module):
    def __init__(self, adj, input_dim, output_dim, dropout, l2_reg=1e-4):
        super(GCN_E, self).__init__()
        self.adj = adj
        self.dropout = dropout
        self.l2_reg = l2_reg
        
        # The first layer: input layer -> 512
        self.layer1 = GraphConvolution(input_dim, 512, dropout, activation=nn.GELU())
        self.bn1 = nn.BatchNorm1d(512)
        
        # Add the attention mechanism
        self.attention = AttentionModule(512)
        
        # The second layer: 512 -> output_dim
        self.layer2 = GraphConvolution(512, output_dim, dropout, activation=nn.GELU())
        self.bn2 = nn.BatchNorm1d(output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj):
        # The first layer
        x1 = self.layer1(x, adj)
        x1 = self.bn1(x1)
        x1 = self.dropout(x1)
        
        # Apply the attention mechanism
        x1 = self.attention(x1)
        
        # The second layer
        x2 = self.layer2(x1, adj)
        x2 = self.bn2(x2)
        
        return x2
        
    def get_l2_loss(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return self.l2_reg * l2_loss


class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        hidden_dim = max(in_dim // 2, out_dim * 2)
        
        self.clf = nn.Sequential(
            # The first layer
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # The second layer
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        
        # Residual connection
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None
        
        # Initialize
        self.clf.apply(xavier_init)
        if self.residual is not None:
            xavier_init(self.residual)
            
    def forward(self, x):
        identity = x
        x = self.clf(x)
        
        # Add the residual connection
        if self.residual is not None:
            x = x + self.residual(identity)
            
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        self.attention_weights = None  # Add this attribute to store the attention weights
    
    def forward(self, x):
        self.attention_weights = self.attention(x)  # Store the attention weights
        return x * self.attention_weights


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.residual = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else None
    
    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.residual is not None:
            identity = self.residual(identity)
        return out + identity


class CrossViewAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(self, x_list):
        # x_list: list of tensors, each with shape [batch_size, dim]
        batch_size = x_list[0].size(0)
        num_views = len(x_list)
        
        # Stack all views into a single tensor [batch_size, num_views, dim]
        x = torch.stack(x_list, dim=1)
        
        # Compute query, key, value
        q = self.query(x)  # [batch_size, num_views, dim]
        k = self.key(x)    # [batch_size, num_views, dim]
        v = self.value(x)  # [batch_size, num_views, dim]
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, num_views, num_views]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [batch_size, num_views, dim]
        
        # Reshape back to list of tensors
        return [out[:, i, :] for i in range(num_views)]


class HAFM(nn.Module):
    def __init__(self, num_view, num_cls, hvcdn_dim):
        super().__init__()
        self.num_cls = num_cls
        self.num_view = num_view
        
        # View-specific feature transformations
        self.view_transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_cls, hvcdn_dim),
                nn.BatchNorm1d(hvcdn_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for _ in range(num_view)
        ])
        
        # View-specific attention mechanisms
        self.view_attention = nn.ModuleList([
            AttentionModule(hvcdn_dim) for _ in range(num_view)
        ])
        
        # Cross-view attention mechanism
        self.cross_view_attention = CrossViewAttention(hvcdn_dim)
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hvcdn_dim * num_view, hvcdn_dim),
            nn.BatchNorm1d(hvcdn_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hvcdn_dim, num_cls)
        )
        
        # Initialize
        self.apply(xavier_init)
    
    def forward(self, in_list):
        # Transform the features of each view
        transformed_features = []
        for i, x in enumerate(in_list):
            x = torch.sigmoid(x)
            x = self.view_transformers[i](x)
            # Apply the view-specific attention mechanism
            x = self.view_attention[i](x)
            transformed_features.append(x)
        
        # Apply the cross-view attention mechanism
        cross_view_features = self.cross_view_attention(transformed_features)
        
        # Feature fusion
        fused_features = torch.cat(cross_view_features, dim=1)
        fused_features = self.fusion(fused_features)
        
        # Classification
        output = self.classifier(fused_features)
        return output


def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc, gcn_dopout=0.5, l2_reg=1e-4):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i+1)] = GCN_E(adj=None, input_dim=dim_list[i], output_dim=dim_he_list[-1], dropout=gcn_dopout, l2_reg=l2_reg)
        model_dict["C{:}".format(i+1)] = Classifier_1(dim_he_list[-1], num_class)
    if num_view >= 2:
        model_dict["C"] = HAFM(num_view, num_class, dim_hc)  
    return model_dict


def init_optim(num_view, model_dict, lr_e=1e-4, lr_c=1e-4):
    optim_dict = {}
    for i in range(num_view):
        # Create separate optimizers for each feature extractor (E) and classifier (C)
        optim_dict["E{:}".format(i+1)] = torch.optim.Adam(model_dict["E{:}".format(i+1)].parameters(), lr=lr_e)
        optim_dict["C{:}".format(i+1)] = torch.optim.Adam(model_dict["C{:}".format(i+1)].parameters(), lr=lr_c)
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict
