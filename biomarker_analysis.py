import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import os

class BiomarkerAnalyzer:
    def __init__(self, model_dict: Dict):
        """
        Initialize the biomarker analyzer
        
        Args:
            model_dict: The dictionary containing the model components
        """
        self.model_dict = model_dict
        self.feature_names = {}
        self.attention_weights = {}
        self.importance_scores = {}
    
    def load_feature_names(self, data_folder: str, view_list: List[str]) -> None:
        """
        Load the feature names for each view
        
        Args:
            data_folder: The path to the data folder
            view_list: The list of views
        """
        for view_idx in range(len(view_list)):
            # Read the feature name file
            feat_file = f"{view_list[view_idx]}_featname.csv"
            try:
                # Read the single column feature name file, no header
                feature_df = pd.read_csv(os.path.join(data_folder, feat_file), header=None)
                # Get the first column as the feature name
                self.feature_names[view_idx] = feature_df[0].values
                print(f"Successfully loaded the feature names for view {view_idx+1}, with {len(self.feature_names[view_idx])} features")
                print(f"The first 5 feature names: {self.feature_names[view_idx][:5]}")  # Print the first 5 feature names for verification
            except Exception as e:
                print(f"Warning: Error loading the feature names for view {view_idx+1}: {str(e)}")
                # If loading fails, use default feature names
                self.feature_names[view_idx] = [f"Feature_{i}" for i in range(1000)]
    
    def collect_attention_weights(self, data_list: List[torch.Tensor], adj_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Collect the attention weights for all views
        
        Args:
            data_list: The list of data tensors for each view
            adj_list: The list of adjacency matrices for each view
            
        Returns:
            The list of attention weights for each view
        """
        attention_weights = []
        
        # Collect the attention weights in GCN
        for view_idx in range(len(data_list)):
            # Get the attention weights in GCN
            gcn_output = self.model_dict[f"E{view_idx+1}"](data_list[view_idx], adj_list[view_idx])
            gcn_attention = self.model_dict[f"E{view_idx+1}"].attention.attention_weights
            
            # Get the attention weights in HAFM
            HAFM_attention = self.model_dict["C"].view_attention[view_idx].attention_weights
            
            # Broadcast the attention weights in HAFM to match the dimension of the attention weights in GCN
            # First, expand the HAFM attention weights to the correct dimension
            HAFM_attention = HAFM_attention.unsqueeze(1).unsqueeze(2)  # [280, 1, 1, 8]
            HAFM_attention = HAFM_attention.expand(-1, gcn_attention.size(1), -1, -1)  # [280, 512, 1, 8]
            
            # Expand the attention weights in GCN to match the dimension
            gcn_attention = gcn_attention.unsqueeze(-1)  # [280, 512, 1]
            gcn_attention = gcn_attention.expand(-1, -1, HAFM_attention.size(-1))  # [280, 512, 8]
            
            # Combine the two attention weights
            combined_weights = gcn_attention * HAFM_attention.squeeze(2)  # [280, 512, 8]
            
            # Take the average of each feature dimension and sample dimension to get the final importance score
            final_weights = combined_weights.mean(dim=[0, -1])  # [512]
            
            attention_weights.append(final_weights)
            
            # Store the original attention weights
            self.attention_weights[view_idx] = {
                'gcn': gcn_attention,
                'hafm': HAFM_attention,
                'combined': combined_weights,
                'final': final_weights
            }
        
        return attention_weights
    
    def get_top_biomarkers(self, attention_weights: List[torch.Tensor], top_k: int = 30) -> List[List[Tuple[str, float]]]:
        """
        Get the top-k important features for each view
        
        Args:
            attention_weights: The list of attention weights for each view
            top_k: The number of important features to return, default is 30
            
        Returns:
            The list of the top-k important features and their importance scores for each view
        """
        top_biomarkers = []
        
        for view_idx, weights in enumerate(attention_weights):
            # Ensure top_k does not exceed the number of features
            valid_top_k = min(top_k, len(self.feature_names[view_idx]))
            
            # Get the indices of the top-k important features
            top_indices = torch.topk(weights, k=valid_top_k).indices
            
            # Get the corresponding feature names and importance scores
            view_biomarkers = []
            for idx in top_indices:
                if idx < len(self.feature_names[view_idx]):  # Ensure the index is within the valid range
                    feature_name = self.feature_names[view_idx][idx]
                    importance = weights[idx].detach().item()  # Use detach() to separate the computational graph
                    view_biomarkers.append((feature_name, importance))
            
            top_biomarkers.append(view_biomarkers)
            
            # Store the importance scores
            self.importance_scores[view_idx] = {
                'indices': top_indices.cpu().numpy(),
                'scores': weights[top_indices].detach().cpu().numpy()
            }
        
        return top_biomarkers
    
    def print_biomarker_results(self, top_biomarkers: List[List[Tuple[str, float]]]) -> None:
        """
        Print the biomarker analysis results
        
        Args:
            top_biomarkers: The list of the top-k important features and their importance scores for each view
        """
        print("\n" + "="*50)
        print("Biomarker analysis results")
        print("="*50)
        
        for view_idx, biomarkers in enumerate(top_biomarkers):
            print(f"\nView {view_idx+1} top 30 important features:")
            for i, (feature_name, importance) in enumerate(biomarkers, 1):
                print(f"{i}. {feature_name}: {importance:.4f}")
    
    def save_results(self, output_dir: str, dataset_name: str) -> None:
        """
        Save the analysis results to a file
        
        Args:
            output_dir: The output directory
            dataset_name: The dataset name used to create a subdirectory to avoid overwriting across datasets
        """
        dataset_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save the important features for each view
        for view_idx in self.importance_scores:
            # Get the valid feature indices
            indices = self.importance_scores[view_idx]['indices']
            valid_indices = indices[indices < len(self.feature_names[view_idx])]
            
            # Get the corresponding feature names and importance scores
            valid_features = self.feature_names[view_idx][valid_indices]
            valid_scores = self.importance_scores[view_idx]['scores'][:len(valid_indices)]
            
            # Create a DataFrame and save it
            results_df = pd.DataFrame({
                'Feature': valid_features,
                'Importance': valid_scores
            })
            results_df.to_csv(
                os.path.join(dataset_dir, f'view_{view_idx+1}_biomarkers.csv'),
                index=False
            )
    