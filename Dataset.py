import torch
from torch.utils.data import Dataset
import torch
from torch.utils.data import Dataset


class PatientDataset(Dataset):
    def __init__(self, X, y, padding_mask, k=5, single_patient=False):
        # self.core = core
        self.X = X
        self.y = y
        self.padding_mask = padding_mask
        # self.padding_mask_core = padding_mask_core
        self.k = k
        self.single_patient = single_patient
        if self.single_patient:
            self.cal_graphs()

        self.edge_list = None

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.padding_mask[idx]
    
    def cal_graphs(self):
        if self.edge_list is not None:
            return self.edge_list
        
        edge_list = []
        for patient_idx in range(len(self.X)):
            edges = self.build_knn_graph(self.X[patient_idx:patient_idx+1], self.core, 
                                                self.padding_mask[patient_idx:patient_idx+1], self.padding_mask_core, k=self.k)
            edge_list.append(edges)
        self.edge_list = torch.cat(edge_list, dim=1).T
        
    @staticmethod
    def build_knn_graph(batch, core, padding_mask_batch, padding_mask_core, k=5):
        """
        Build a KNN graph from batch and core tensors.
        
        Args:
            batch: 3D tensor (batch_size, seq_len, features)
            core: 3D tensor (core_size, seq_len, features)
            padding_mask_batch: 2D tensor (batch_size, seq_len) indicating valid time points
            padding_mask_core: 2D tensor (core_size, seq_len) indicating valid time points
            k: number of nearest neighbors for patient connections
        
        Returns:
            edge_index: tensor of shape (2, num_edges) representing graph edges
        """
        batch_size, seq_len, _ = batch.shape
        core_size = core.shape[0]
        total_patients = batch_size + core_size
        batch_size = batch.shape[0]
        all_patients = torch.cat([batch, core], dim=0)
        all_padding_mask = torch.cat([padding_mask_batch, padding_mask_core], dim=0)
        
        edges = []
        
        for patient_idx in range(total_patients):
            for t in range(seq_len - 1):
            
                if all_padding_mask[patient_idx, t] > 0 and all_padding_mask[patient_idx, t + 1] > 0:
                    node_curr = patient_idx * seq_len + t
                    node_next = patient_idx * seq_len + t + 1
   
                    edges.append([node_curr, node_next])
                    edges.append([node_next, node_curr])
        
        
        for t in range(seq_len):
            valid_patients = all_padding_mask[:, t] > 0
            valid_indices = torch.where(valid_patients)[0] 
            

            valid_patients_to = all_padding_mask[batch_size:, t] > 0
            valid_indices_to = torch.where(valid_patients_to)[0] + batch_size


            if len(valid_indices) > 1:
                features_t = all_patients[valid_indices, t, :]
                features_t_to = all_patients[valid_indices_to, t, :]

                distances = torch.cdist(features_t, features_t_to, p=2)
                
                for i, patient_idx in enumerate(valid_indices):
                    num_neighbors = min(k, len(valid_indices_to))
                    _, nearest_indices = torch.topk(distances[i], num_neighbors, largest=False)
                    
                    for j in nearest_indices:
                        neighbor_idx = valid_indices_to[j]
                        node_curr = patient_idx * seq_len + t
                        node_neighbor = neighbor_idx * seq_len + t
                        if node_curr >= all_padding_mask.shape[0] * seq_len or node_neighbor >= all_padding_mask.shape[0] * seq_len:
                            flag = 1
                        edges.append([node_curr, node_neighbor])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return edge_index
