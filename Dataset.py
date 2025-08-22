import torch
from torch.utils.data import Dataset
class PatientDataset(Dataset):
    def __init__(self, X, y, core, padding_mask, padding_mask_core, notes, k=5):
        self.core = core
        self.X = X
        self.y = y
        self.padding_mask = padding_mask
        self.padding_mask_core = padding_mask_core
        self.k = k
        self.notes = notes
        self.node_to_neighbors = self.build_knn_graph(X, core, padding_mask, padding_mask_core, k=k)


    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.padding_mask[idx], idx, self.notes[idx]


    def get_edge_index(self, batch, padding_mask_batch, batch_indices):
        """
        Returns a tensor representing the edge index of the graph.
        
        Returns:
            edge_index: tensor of shape (2, num_edges) representing graph edges
        """
        batch_size, seq_len, _ = batch.shape
        core_size = self.core.shape[0]
        total_patients = batch_size + core_size

        batch_size = batch.shape[0]
        # all_patients = torch.cat([batch, self.core], dim=0)
        all_padding_mask = torch.cat([padding_mask_batch, self.padding_mask_core.to(padding_mask_batch.device)], dim=0)

        edges = []
        
        for patient_idx in range(total_patients):
            for t in range(seq_len - 1):
                if all_padding_mask[patient_idx, t] > 0 and all_padding_mask[patient_idx, t + 1] > 0:
                    node_curr = patient_idx * seq_len + t
                    node_next = patient_idx * seq_len + t + 1
   
                    edges.append([node_curr, node_next])
                    edges.append([node_next, node_curr])
        
        edges = [torch.tensor(edges).t()]

        neighbors_list = []
        for idx in batch_indices:
            neighbors_list.extend(self.node_to_neighbors[idx.item() * seq_len: (idx.item() + 1) * seq_len])
            # for i in range(seq_len):
            #     neighbors_list.append(list(self.node_to_neighbors[idx.item() * seq_len + i]))
        
        # for i in range(self.X.shape[0] * seq_len, self.X.shape[0] * seq_len + core_size * seq_len):
        #     neighbors_list.append(list(self.node_to_neighbors[i]))
        neighbors_list.extend(self.node_to_neighbors[self.X.shape[0] * seq_len:])
    
        for t in range(seq_len):
            valid_patients = all_padding_mask[:, t] > 0
            valid_indices = torch.where(valid_patients)[0] 

            if len(valid_indices) > 1:
                for i, patient_idx in enumerate(valid_indices):
                    node_curr = patient_idx * seq_len + t
                    to_append = torch.tensor(neighbors_list[node_curr], dtype=torch.long) - (self.X.shape[0] - batch_size) * seq_len  # Adjust index for core patients
                    edges.append(torch.stack([to_append, torch.full_like(to_append, node_curr)], dim=0))

        
        if edges:
            edge_index = torch.cat(edges, dim=1).to(torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return edge_index


        
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
        
        node_to_neighbors = {i: set() for i in range(total_patients * seq_len)}

        edges = []
           
        for t in range(seq_len):
            valid_patients = all_padding_mask[:, t] > 0
            valid_indices = torch.where(valid_patients)[0] 

            core_patients = all_padding_mask[batch_size:, t] > 0
            core_indices = torch.where(core_patients)[0] + batch_size

            if len(valid_indices) > 1:
                features_t = all_patients[valid_indices, t, :]
                features_t_to = all_patients[core_indices, t, :]

                distances = torch.cdist(features_t, features_t_to, p=2)
                
                for i, patient_idx in enumerate(valid_indices):
                    num_neighbors = min(k, len(core_indices))
                    _, nearest_indices = torch.topk(distances[i], num_neighbors, largest=False)
                    
                    for j in nearest_indices:
                        neighbor_idx = core_indices[j]
                        node_curr = patient_idx * seq_len + t
                        node_neighbor = neighbor_idx * seq_len + t

                        edges.append([node_neighbor, node_curr])
                        
                        node_to_neighbors[node_curr.item()].add(node_neighbor.item())
        
        result = [list(node_to_neighbors[i]) for i in range(total_patients * seq_len)]
        return result
