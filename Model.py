from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE = torch.device('cpu')  # For testing purposes, use CPU


class GraphGRUMortalityModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n1_gat_layers, n2_gru_layers, X_core, core_padding_mask,
                 num_heads=4, dropout=0.1, seq_len=18, k=5, gnn_falg=True):
        """
        Mortality prediction model with Graph Attention + GRU layers
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for GAT and GRU layers
            n1_gat_layers: Number of Graph Attention layers
            n2_gru_layers: Number of GRU layers
            X_core_dim: Core set dimension (number of core patients)
            num_heads: Number of attention heads for GAT
            dropout: Dropout rate
            seq_len: Sequence length
        """
        super(GraphGRUMortalityModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n1_gat_layers = n1_gat_layers
        self.n2_gru_layers = n2_gru_layers
        self.X_core = X_core.to(DEVICE)
        self.core_padding_mask = core_padding_mask.to(DEVICE)
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.gnn_flag = gnn_falg
        
        self.gat_layers = nn.ModuleList().to(DEVICE)
        
        
        self.gat_layers.append(
            GATv2Conv(input_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, concat=True)
        )
        
        for _ in range(n1_gat_layers - 1):
            self.gat_layers.append(
                GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, concat=True)
            )
        
        if self.gnn_flag:
            self.gru = nn.GRU(hidden_dim, hidden_dim, n2_gru_layers, batch_first=True, dropout=dropout)
        else:
            self.gru = nn.GRU(input_dim, hidden_dim, n2_gru_layers, batch_first=True, dropout=dropout)
        
        self.classifier = nn.Linear(hidden_dim, 1).to(DEVICE)
        self.dropout = nn.Dropout(dropout).to(DEVICE)
        self.k = k

        self.best_model = None

        # initialize weights
        for layer in self.gat_layers:
            if isinstance(layer, GATv2Conv):
                nn.init.xavier_uniform_(layer.lin_l.weight)
                nn.init.xavier_uniform_(layer.lin_r.weight) 
        nn.init.xavier_uniform_(self.classifier.weight) 
        nn.init.constant_(self.classifier.bias, 0.0)
        nn.init.xavier_uniform_(self.gru.weight_ih_l0)
        nn.init.xavier_uniform_(self.gru.weight_hh_l0)
        nn.init.constant_(self.gru.bias_ih_l0, 0.0)
        nn.init.constant_(self.gru.bias_hh_l0, 0.0)
        
  
        
    def forward(self, x ,padding_mask, edge_index):
        """
        Forward pass
        
        Args:
            core: Core patients tensor (X_core_dim, seq_len, input_dim)
            x: Batch patients tensor (batch_size, seq_len, input_dim)
            y: Target labels (batch_size, seq_len)
            padding_mask: Padding mask (batch_size, seq_len)
            edge_index: Graph edge indices (2, num_edges)
        
        Returns:
            predictions: Mortality predictions (batch_size, seq_len, 1)
        """
        batch_size = x.size(0)
        
        if self.gnn_flag:
            all_patients = torch.cat([x, self.X_core], dim=0)  # (batch_size + X_core_dim, seq_len, input_dim)
            total_patients = batch_size + self.X_core.shape[0]
            
            # Reshape for graph processing: (total_patients * seq_len, input_dim)
            graph_input = all_patients.view(total_patients * self.seq_len, -1)
            # Apply GAT layers
            for gat_layer in self.gat_layers:
                graph_input = F.relu(gat_layer(graph_input, edge_index))
                # graph_input = self.dropout(graph_input)
            
            # Reshape back to sequence format: (total_patients, seq_len, hidden_dim)
            graph_output = graph_input.view(total_patients, self.seq_len, -1)

            # Extract only batch patients (exclude core)
        
            batch_output = graph_output[:batch_size]  # (batch_size, seq_len, hidden_dim)
        
        else:
            batch_output = x
        # Apply GRU layers
        # Pack sequences for efficient processing
        # lengths from mask (True = pad) → count valid steps
        lengths = (padding_mask.to(bool)).sum(dim=1)                       # (batch,)
        lengths = lengths.clamp(min=1).cpu()
        #packed_input = pack_padded_sequence(batch_output, lengths, batch_first=True, enforce_sorted=False)
        #gru_output, _ = self.gru(packed_input)
        # Unpack sequences
        #gru_output, _ = pad_packed_sequence(gru_output, batch_first=True, total_length=self.seq_len)


        gru_output, _ = self.gru(batch_output)
        # Apply classifier
        predictions = self.classifier(gru_output)  # (batch_size, seq_len, 1)
        
        return predictions.squeeze(-1)  # (batch_size, seq_len)
    
    
    def masked_bce_loss(self, logits, targets, mask, pos_weight=None):
        T = min(logits.shape[1], targets.shape[1], mask.shape[1])
        logits, targets, mask = logits[:, :T], targets[:, :T], mask[:, :T]
        loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)(logits, targets)
        return (loss * mask).sum() / mask.sum()
    

    def train_all(self, dataloaders, datasets, epochs: int = 10, learning_rate: float = 1e-3, pos_lambda : float = 1):
        self.train()
        optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        best_validation_accuracy = - float('inf')
        targets = datasets['train'].y
        pos_weight = (targets == 0).sum() / (targets == 1).sum()  # Adjust pos_weight as needed
        print(f'Pos weight: {pos_weight:.4f}')


        for epoch in range(epochs):
            print(f'Starting epoch {epoch + 1}/{epochs}')
            total = 0
            for x, y, padding_mask in tqdm(dataloaders['train']):
                optim.zero_grad()
                x, padding_mask, y = x.to(DEVICE), padding_mask.to(DEVICE), y.to(DEVICE)
                edge_index = datasets['train'].build_knn_graph(x, self.X_core, padding_mask, self.core_padding_mask, k=self.k).to(DEVICE)
                predictions = self.forward(x, padding_mask, edge_index)
                loss = self.masked_bce_loss(
                    predictions, y, padding_mask, pos_weight=pos_weight * pos_lambda
                )
                #loss = F.binary_cross_entropy_with_logits(predictions.view(-1), y.view(-1))
                loss.backward()
                optim.step()
                
                total += loss.item()
            avg_loss = total / len(dataloaders['train'])
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
            validation_accuracy = self.validate(dataloaders['val'], datasets['val'])
            print(f'Validation Accuracy: {validation_accuracy[0]:.4f} | AUC: {validation_accuracy[1]:.4f} | AP: {validation_accuracy[2]:.4f}')
            if validation_accuracy[2] > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy[2]
                self.best_model = self.state_dict()
                print("Best model updated")
        self.load_state_dict(self.best_model)

    def validate(self, dataloader, dataset):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            all_true_labels = []
            all_predicted_labels = []
            for x, y, padding_mask in dataloader:
                x, padding_mask, y = x.to(DEVICE), padding_mask.to(DEVICE), y.to(DEVICE)
                edge_index = dataset.build_knn_graph(x, self.X_core, padding_mask, self.core_padding_mask, k=self.k).to(DEVICE)
                predictions = self.forward(x, padding_mask, edge_index)
                all_true_labels.extend(y.cpu().numpy().flatten())
                all_predicted_labels.extend(torch.sigmoid(predictions).cpu().numpy().flatten())
                predicted_labels = (torch.sigmoid(predictions) > 0.5).float()
                correct += (predicted_labels == y).sum().item()
                total += y.numel()
        accuracy = correct / total if total > 0 else 0
        self.train()
        return accuracy, roc_auc_score(all_true_labels, all_predicted_labels), average_precision_score(all_true_labels, all_predicted_labels)