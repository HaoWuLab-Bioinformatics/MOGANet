""" Training and testing of the model
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn.functional as F
from models import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter
from utils import save_model_dict, load_model_dict  
import torch.nn as nn
import pandas as pd
from biomarker_analysis import BiomarkerAnalyzer

# Set the GPU to use.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  

cuda = True if torch.cuda.is_available() else False
if cuda:
    print(f"Using GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "pearson"  # Use Pearson correlation coefficient.
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))
    
    return adj_train_list, adj_test_list


def train_epoch(model_dict, optim_dict, data_dict, data_folder):
    loss_dict = {}
    for m in model_dict:
        model_dict[m].train()
        optim_dict[m].zero_grad()
        
        # Forward propagation.
        if m.startswith('E'):
            # Feature extractor.
            output = model_dict[m](data_dict["X"][int(m[1])-1], data_dict["adj"][int(m[1])-1])
            # Compute loss.
            loss = F.cross_entropy(output, data_dict["Y"])
        elif m.startswith('C') and m != 'C':
            # Single-view classifier.
            view_idx = int(m[1])-1
            output = model_dict[m](model_dict[f"E{view_idx+1}"](data_dict["X"][view_idx], data_dict["adj"][view_idx]))
            loss = F.cross_entropy(output, data_dict["Y"])
        else:
            # View-fusion classifier.
            ci_list = []
            for i in range(len(data_dict["X"])):
                ci_list.append(model_dict[f"C{i+1}"](model_dict[f"E{i+1}"](data_dict["X"][i], data_dict["adj"][i])))
            output = model_dict[m](ci_list)
            loss = F.cross_entropy(output, data_dict["Y"])
        
        # Add L2 regularization loss, only apply to GCN_E model.
        if data_folder == 'LGG' and m.startswith('E'):
            l2_loss = model_dict[m].get_l2_loss()
            loss = loss + l2_loss
        
        # Backward propagation.
        loss.backward()
        optim_dict[m].step()
        
        loss_dict[m] = loss.item()
    
    return loss_dict


def test_epoch(data_list, adj_list, te_idx, model_dict, data_folder):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])))
    if num_view >= 2:
        c = model_dict["C"](ci_list)    
    else:
        c = ci_list[0]
    c = c[te_idx,:]
    prob = F.softmax(c, dim=1).data.cpu().numpy()
    
    return prob


def load_predefined_data(data_folder, view_list):
    """
    Load the predefined training and test set data.
    Args:
        data_folder: The name of the dataset folder.
        view_list: The list of views.
    Returns:
        data_train_list: The list of training data.
        data_test_list: The list of test data.
        labels_train: The training labels.
        labels_test: The test labels.
    """
    num_view = len(view_list)
    
    # Load the training and test set labels.
    labels_train = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',').astype(int)
    labels_test = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',').astype(int)
    
    # Load the training and test set data for each view.
    data_train_list = []
    data_test_list = []
    for i in view_list:
        train_data = np.loadtxt(os.path.join(data_folder, f"{i}_tr.csv"), delimiter=',')
        test_data = np.loadtxt(os.path.join(data_folder, f"{i}_te.csv"), delimiter=',')
        data_train_list.append(train_data)
        data_test_list.append(test_data)
    
    return data_train_list, data_test_list, labels_train, labels_test


def train_test_predefined(data_folder, view_list, num_class,
                         lr_e_pretrain, lr_e, lr_c, 
                         num_epoch_pretrain, num_epoch):
    """
    Use the predefined training and test set for training and testing.
    """
    test_interval = 100
    num_view = len(view_list)
    dim_hvcdn = pow(num_class,num_view)
    
    # Set the network structure parameters according to the dataset type.
    if data_folder == 'BRCA':
        adj_parameter = 10
        dim_he_list = [1024, 512, 256, 128, 500]
        dim_hc = 500
        gcn_dropout = 0.5
        l2_reg = 0.0
    elif data_folder == 'LGG':
        adj_parameter = 10
        dim_he_list = [1024, 512, 256, 128, 500]
        dim_hc = 500
        gcn_dropout = 0.6
        l2_reg = 1e-3
    elif data_folder == 'KIDNEY':
        adj_parameter = 5
        dim_he_list = [256, 128, 64, 32, 100]
        dim_hc = 500
        gcn_dropout = 0.5
        l2_reg = 1e-4
    else:
        raise ValueError(f"Unknown dataset: {data_folder}")
    
    # Load the predefined data.
    data_train_list, data_test_list, labels_train, labels_test = load_predefined_data(data_folder, view_list)
    
    # Prepare the data tensors.
    data_train_tensor_list = [torch.FloatTensor(mat) for mat in data_train_list]
    data_test_tensor_list = [torch.FloatTensor(mat) for mat in data_test_list]
    if cuda:
        data_train_tensor_list = [t.cuda() for t in data_train_tensor_list]
        data_test_tensor_list = [t.cuda() for t in data_test_tensor_list]
    
    # Prepare the label tensors.
    labels_train_tensor = torch.LongTensor(labels_train)
    onehot_labels_train_tensor = one_hot_tensor(labels_train_tensor, num_class)
    sample_weight_train = cal_sample_weight(labels_train, num_class)
    sample_weight_train = torch.FloatTensor(sample_weight_train)
    if cuda:
        labels_train_tensor = labels_train_tensor.cuda()
        onehot_labels_train_tensor = onehot_labels_train_tensor.cuda()
        sample_weight_train = sample_weight_train.cuda()
    
    # Generate the adjacency matrices.
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_train_list)):
        # Generate the adjacency matrix for the training set.
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_train_list[i], "pearson")
        adj_train = gen_adj_mat_tensor(data_train_list[i], adj_parameter_adaptive, "pearson")
        adj_train_list.append(adj_train)
        
        # Generate the adjacency matrix for the test set.
        adj_test = gen_adj_mat_tensor(data_test_list[i], adj_parameter_adaptive, "pearson")
        adj_test_list.append(adj_test)
    
    # Initialize the model.
    dim_list = [x.shape[1] for x in data_train_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn, gcn_dropout, l2_reg)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    
    # Pretrain stage.
    print("\nPretrain GCNs...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    scheduler_dict = {}
    for m in model_dict:
        if m.startswith('C'):
            scheduler_dict[m] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim_dict[m], mode='max', factor=0.7, patience=100, verbose=True, min_lr=1e-6
            )
    
    # Pretrain loop.
    for epoch in range(num_epoch_pretrain):
        train_epoch(model_dict, optim_dict, 
                   {"X": data_train_tensor_list, "Y": labels_train_tensor, "adj": adj_train_list}, 
                   data_folder)
        
        if epoch % test_interval == 0:
            train_prob = test_epoch(data_train_tensor_list, adj_train_list, 
                                  list(range(len(data_train_list[0]))), model_dict, data_folder)
            train_pred = train_prob.argmax(1)
            
            print("\nEpoch {:d}".format(epoch))
            if data_folder == 'BRCA':
                train_acc = accuracy_score(labels_train, train_pred)
                train_f1_weighted = f1_score(labels_train, train_pred, average='weighted')
                train_f1_macro = f1_score(labels_train, train_pred, average='macro')
                print("Train ACC: {:.3f}".format(train_acc))
                print("Train F1 weighted: {:.3f}".format(train_f1_weighted))
                print("Train F1 macro: {:.3f}".format(train_f1_macro))
            else:  # LGG, Kidney
                train_acc = accuracy_score(labels_train, train_pred)
                train_f1 = f1_score(labels_train, train_pred, average='weighted')
                train_auc = roc_auc_score(labels_train, train_prob[:, 1])
                print("Train ACC: {:.3f}".format(train_acc))
                print("Train F1: {:.3f}".format(train_f1))
                print("Train AUC: {:.3f}".format(train_auc))
    
    # Main training phase.
    print("\nMain training phase...")
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    scheduler_dict = {}
    for m in model_dict:
        if m.startswith('C'):
            scheduler_dict[m] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim_dict[m], mode='max', factor=0.7, patience=100, verbose=True, min_lr=1e-6
            )
    
    # Main training loop.
    for epoch in range(num_epoch):
        train_epoch(model_dict, optim_dict, 
                   {"X": data_train_tensor_list, "Y": labels_train_tensor, "adj": adj_train_list}, 
                   data_folder)
        
        if epoch % test_interval == 0:
            train_prob = test_epoch(data_train_tensor_list, adj_train_list, 
                                  list(range(len(data_train_list[0]))), model_dict, data_folder)
            train_pred = train_prob.argmax(1)
            
            print("\nEpoch {:d}".format(epoch + num_epoch_pretrain))
            if data_folder == 'BRCA':
                train_acc = accuracy_score(labels_train, train_pred)
                train_f1_weighted = f1_score(labels_train, train_pred, average='weighted')
                train_f1_macro = f1_score(labels_train, train_pred, average='macro')
                print("Train ACC: {:.3f}".format(train_acc))
                print("Train F1 weighted: {:.3f}".format(train_f1_weighted))
                print("Train F1 macro: {:.3f}".format(train_f1_macro))
            else:  # LGG, KIDNEY, ROSMAP
                train_acc = accuracy_score(labels_train, train_pred)
                train_f1 = f1_score(labels_train, train_pred, average='weighted')
                train_auc = roc_auc_score(labels_train, train_prob[:, 1])
                print("Train ACC: {:.3f}".format(train_acc))
                print("Train F1: {:.3f}".format(train_f1))
                print("Train AUC: {:.3f}".format(train_auc))
    
    # Perform biomarker analysis before final test evaluation.
    print("\n--- Biomarker analysis ---")
    biomarker_analyzer = BiomarkerAnalyzer(model_dict)
    biomarker_analyzer.load_feature_names(data_folder, view_list)
    attention_weights = biomarker_analyzer.collect_attention_weights(data_train_tensor_list, adj_train_list)
    top_biomarkers = biomarker_analyzer.get_top_biomarkers(attention_weights, top_k=30)
    biomarker_analyzer.print_biomarker_results(top_biomarkers)
    
    # Save the biomarker analysis results.
    biomarker_analyzer.save_results('biomarker_results', data_folder)
    
    # Final test evaluation.
    print("\nFinal evaluation on test set...")
    test_prob = test_epoch(data_test_tensor_list, adj_test_list, 
                          list(range(len(data_test_list[0]))), model_dict, data_folder)
    test_pred = test_prob.argmax(1)
    
    # Calculate the test set metrics.
    test_acc = accuracy_score(labels_test, test_pred)
    if data_folder == 'BRCA':
        test_f1_weighted = f1_score(labels_test, test_pred, average='weighted')
        test_f1_macro = f1_score(labels_test, test_pred, average='macro')
        test_auc = None
    else:  # LGG, KIDNEY, ROSMAP
        test_f1 = f1_score(labels_test, test_pred, average='weighted')
        test_auc = roc_auc_score(labels_test, test_prob[:, 1])
        test_f1_weighted = None
        test_f1_macro = None
    
    # Save the best model.
    save_model_dict(model_dict, data_folder, "best_model")
    
    # Return the final test results.
    if data_folder == 'BRCA':
        return {
            'acc': test_acc,
            'f1_weighted': test_f1_weighted,
            'f1_macro': test_f1_macro
        }
    else:  # LGG, KIDNEY
        return {
            'acc': test_acc,
            'f1': test_f1,
            'auc': test_auc
        }
