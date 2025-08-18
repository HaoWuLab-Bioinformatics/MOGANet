import os
import torch
import numpy as np
from train_test import train_test_predefined

if __name__ == "__main__":
    # Set the random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Dataset configuration
    data_folder = "KIDNEY"  # Optional: "BRCA", "LGG", "KIDNEY"
    view_list = [1, 2, 3]  # The list of views
    
    # Set the parameters according to the dataset
    if data_folder == 'BRCA':
        num_class = 5
        lr_e_pretrain = 1e-5  # The pretraining learning rate
        lr_e = 5e-6  # The feature extractor learning rate
        lr_c = 5e-5  # The classifier learning rate
        num_epoch_pretrain = 800  # The number of pretraining epochs
        num_epoch = 1500  # The total number of training epochs
    elif data_folder == 'LGG':
        num_class = 2
        lr_e_pretrain = 1e-5
        lr_e = 5e-5  # Adjust the feature extractor learning rate
        lr_c = 5e-5  # Adjust the classifier learning rate
        num_epoch_pretrain = 400
        num_epoch = 800
    elif data_folder == 'KIDNEY':
        num_class = 2
        lr_e_pretrain = 5e-5
        lr_e = 2e-5
        lr_c = 2e-4
        num_epoch_pretrain = 200  # The number of pretraining epochs
        num_epoch = 400  # The number of main training epochs
    else:
        raise ValueError(f"Unsupported dataset: {data_folder}")
    
    print("\n" + "="*50)
    print(f"Start training and testing on {data_folder} dataset")
    print("="*50)
    
    # Execute training and testing
    results = train_test_predefined(
        data_folder=data_folder,
        view_list=view_list,
        num_class=num_class,
        lr_e_pretrain=lr_e_pretrain,
        lr_e=lr_e,
        lr_c=lr_c,
        num_epoch_pretrain=num_epoch_pretrain,
        num_epoch=num_epoch
    )
    
    # Print the final results
    print("\n" + "="*50)
    print("Final test results:")
    print("="*50)
    for metric, value in results.items():
        if value is not None:
            print(f"{metric}: {value:.4f}")             