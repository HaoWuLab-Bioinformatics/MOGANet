import os
import numpy as np
import torch
import torch.nn.functional as F

cuda = True if torch.cuda.is_available() else False

def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels==i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels==i)[0]] = count[i]/np.sum(count)
    
    return sample_weight


def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)
    
    return y_onehot


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def pearson_correlation_torch(x1, x2=None, eps=1e-8):
    # 确保输入是torch张量
    if not isinstance(x1, torch.Tensor):
        x1 = torch.FloatTensor(x1)
    if x2 is not None and not isinstance(x2, torch.Tensor):
        x2 = torch.FloatTensor(x2)
    
    x2 = x1 if x2 is None else x2
    if cuda:
        x1 = x1.cuda()
        x2 = x2.cuda()
    
    # 中心化数据
    x1_centered = x1 - torch.mean(x1, dim=1, keepdim=True)
    x2_centered = x2 - torch.mean(x2, dim=1, keepdim=True)
    
    # 计算标准差
    x1_std = torch.sqrt(torch.sum(x1_centered ** 2, dim=1, keepdim=True))
    x2_std = torch.sqrt(torch.sum(x2_centered ** 2, dim=1, keepdim=True))
    
    # 计算皮尔逊相关系数
    corr = torch.mm(x1_centered, x2_centered.t()) / (x1_std * x2_std.t()).clamp(min=eps)
    
    # 将相关系数转换为距离（1 - |corr|）
    return 1 - torch.abs(corr)


def to_sparse(x):
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return torch.sparse_coo_tensor(size=x.size(), device=x.device)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return torch.sparse_coo_tensor(indices, values, x.size(), device=x.device)


def cal_adj_mat_parameter(edge_per_node, data, metric="pearson"):
    if metric == "cosine":
        dist = cosine_distance_torch(data, data)
    elif metric == "pearson":
        dist = pearson_correlation_torch(data, data)
    else:
        raise NotImplementedError(f"Metric {metric} not implemented")
    
    parameter = torch.sort(dist.reshape(-1,)).values[edge_per_node*data.shape[0]]
    return parameter.item()


def graph_from_dist_tensor(dist, parameter, self_dist=True):
    if self_dist:
        assert dist.shape[0]==dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0
        
    return g


def gen_adj_mat_tensor(data, parameter, metric="pearson"):
    if metric == "cosine":
        dist = cosine_distance_torch(data, data)
        adj = 1-dist
    elif metric == "pearson":
        dist = pearson_correlation_torch(data, data)
        adj = 1-dist
    else:
        raise NotImplementedError(f"Metric {metric} not implemented")
    
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    adj = adj*g 
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)
    
    return adj


def gen_test_adj_mat_tensor(data, trte_idx, parameter, metric="pearson"):
    adj = torch.zeros((data.shape[0], data.shape[0]))
    if cuda:
        adj = adj.cuda()
    num_tr = len(trte_idx["tr"])
    
    if metric == "cosine":
        dist_tr2te = cosine_distance_torch(data[trte_idx["tr"]], data[trte_idx["te"]])
        adj[:num_tr,num_tr:] = 1-dist_tr2te
        dist_te2tr = cosine_distance_torch(data[trte_idx["te"]], data[trte_idx["tr"]])
        adj[num_tr:,:num_tr] = 1-dist_te2tr
    elif metric == "pearson":
        dist_tr2te = pearson_correlation_torch(data[trte_idx["tr"]], data[trte_idx["te"]])
        adj[:num_tr,num_tr:] = 1-dist_tr2te
        dist_te2tr = pearson_correlation_torch(data[trte_idx["te"]], data[trte_idx["tr"]])
        adj[num_tr:,:num_tr] = 1-dist_te2tr
    else:
        raise NotImplementedError(f"Metric {metric} not implemented")
    
    g_tr2te = graph_from_dist_tensor(dist_tr2te, parameter, self_dist=False)
    adj[:num_tr,num_tr:] = adj[:num_tr,num_tr:]*g_tr2te
    
    g_te2tr = graph_from_dist_tensor(dist_te2tr, parameter, self_dist=False)
    adj[num_tr:,:num_tr] = adj[num_tr:,:num_tr]*g_te2tr
    
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)
    
    return adj


def save_model_dict(model_dict, folder, model_name):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for module in model_dict:
        torch.save(model_dict[module].state_dict(), os.path.join(folder, f"{model_name}_{module}.pth"))
            
    
def load_model_dict(folder, model_dict):
    for module in model_dict:
        if os.path.exists(os.path.join(folder, module+".pth")):
            model_dict[module].load_state_dict(torch.load(os.path.join(folder, module+".pth"), map_location="cuda:{:}".format(torch.cuda.current_device())))
        else:
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
        if cuda:
            model_dict[module].cuda()    
    return model_dict