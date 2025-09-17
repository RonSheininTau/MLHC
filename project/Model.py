import sys
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv,TransformerConv
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NOTE_DIM = 768  

class GraphGRUMortalityModel(nn.Module):
    def __init__(self, input_dim, hidden_dim,  n1_gat_layers, n2_gru_layers, X_core, core_padding_mask,
                 num_of_bios, num_prescriptions, bios_hidden_dim=None, pres_hidden_dim=None, num_heads=4, dropout=0.1, seq_len=18, k=5, gnn_flag=True,
                 use_notes=True, use_bios=True, use_prescriptions=True, use_x=True):
        """
        Mortality prediction model with Graph Attention + GRU layers
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for GAT and GRU layers
            n1_gat_layers: Number of Graph Attention layers
            n2_gru_layers: Number of GRU layers
            X_core: Core set tensor (number of core patients)
            core_padding_mask: Padding mask for core patients
            num_of_bios: Number of bios features
            num_prescriptions: Number of prescription types
            bios_hidden_dim: Hidden dimension for bios processing
            pres_hidden_dim: Hidden dimension for prescription processing
            num_heads: Number of attention heads for GAT
            dropout: Dropout rate
            seq_len: Sequence length
            k: Number of nearest neighbors for graph construction
            gnn_flag: Whether to use GNN layers
            use_notes: Whether to use notes modality
            use_bios: Whether to use bios modality
            use_prescriptions: Whether to use prescriptions modality
            use_x: Whether to use main sequential data (X) modality
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
        self.gnn_flag = gnn_flag
        self.num_of_bios = num_of_bios
        self.num_prescriptions = num_prescriptions
        self.bios_hidden_dim = bios_hidden_dim if bios_hidden_dim is not None else hidden_dim
        self.pres_hidden_dim = pres_hidden_dim if pres_hidden_dim is not None else hidden_dim
        
        # Modality selection flags for ablation study
        self.use_notes = use_notes
        self.use_bios = use_bios
        self.use_prescriptions = use_prescriptions
        self.use_x = use_x
        
        # Conditionally create modality-specific layers based on selection flags
        if self.use_notes and gnn_flag:
            self.notes_layer = nn.Linear(NOTE_DIM, hidden_dim).to(DEVICE)
        else:
            self.notes_layer = None
            
        if self.use_bios and gnn_flag:
            self.bios_layer = nn.Linear(num_of_bios, self.bios_hidden_dim).to(DEVICE)
        else:
            self.bios_layer = None
            
        if self.use_prescriptions and gnn_flag:
            self.pres_layer = nn.EmbeddingBag(num_prescriptions, self.pres_hidden_dim, mode='mean').to(DEVICE)
        else:
            self.pres_layer = None
        self.gat_layers = nn.ModuleList().to(DEVICE)
        
        self.gat_layers.append(
            TransformerConv(input_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, concat=True)
        )
        
        for _ in range(n1_gat_layers - 1):
            self.gat_layers.append(
                TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, concat=True)
            )
        
        if self.gnn_flag:
            self.gru = nn.GRU(hidden_dim, hidden_dim, n2_gru_layers, batch_first=True, dropout=dropout)
        else:
            self.gru = nn.GRU(input_dim, hidden_dim, n2_gru_layers, batch_first=True, dropout=dropout)
        
        clf_input_dim = 0
        
        if self.use_x:
            clf_input_dim += 3 * hidden_dim
        
        if self.use_notes:
            clf_input_dim += hidden_dim
        if self.use_bios:
            clf_input_dim += self.bios_hidden_dim
        if self.use_prescriptions:
            clf_input_dim += self.pres_hidden_dim
            
        if not self.gnn_flag:
            clf_input_dim = hidden_dim 

        self.classifier_mort = nn.Sequential(
            nn.Linear(clf_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.classifier_re = nn.Sequential(
            nn.Linear(clf_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.classifier_pro = nn.Sequential(
            nn.Linear(clf_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        
        self.dropout = nn.Dropout(dropout).to(DEVICE)
        self.k = k

        self.best_model = None

        # initialize weights
        for layer in self.gat_layers:
            if isinstance(layer, GATv2Conv):
                nn.init.xavier_uniform_(layer.lin_l.weight)
                nn.init.xavier_uniform_(layer.lin_r.weight) 
        nn.init.xavier_uniform_(self.gru.weight_ih_l0)
        nn.init.xavier_uniform_(self.gru.weight_hh_l0)
        nn.init.constant_(self.gru.bias_ih_l0, 0.0)
        nn.init.constant_(self.gru.bias_hh_l0, 0.0)
        for layer in self.classifier_mort:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        for layer in self.classifier_re:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        for layer in self.classifier_pro:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        self.calibrator = None

    def initialize_calibrator(self, true_labels = None, raw_predictions = None):
        print("initializing calibrator")
        self.calibrator = [LogisticRegression(solver="lbfgs") for _ in range(len(raw_predictions))]
        for i in range(len(raw_predictions)):
            self.calibrator[i].fit(raw_predictions[i].reshape(-1, 1), true_labels[i])

    def calibrate_predictions(self, true_labels = None, raw_predictions = None, calibrator = 0):
        if self.calibrator is None:
            return raw_predictions
        calibrated_predictions = self.calibrator[calibrator].predict_proba(raw_predictions.reshape(-1, 1))[:, 1]
        return calibrated_predictions

        
    @staticmethod
    def collect_bags(batch):
        """
        batch: list of (subject_id, LongTensor[Ni])
        returns:
        subject_ids: LongTensor[B]
        drug_ids:    LongTensor[sum(Ni)]
        offsets:     LongTensor[B]  (start index of each bag in drug_ids)
        """
        subject_ids, bags = zip(*batch)
        lens = [len(b) for b in bags]
        offsets = torch.zeros(len(bags), dtype=torch.long)
        if lens:
            offsets[1:] = torch.tensor(lens[:-1]).cumsum(dim=0)
            drug_ids = torch.cat(bags).to(torch.long) if sum(lens) > 0 else torch.empty(0, dtype=torch.long)
        else:
            drug_ids = torch.empty(0, dtype=torch.long)
        return torch.tensor(subject_ids, dtype=torch.long), drug_ids, offsets
        
    def forward(self, x ,padding_mask, edge_index, nots, bios, prescriptions):
        """
        Forward pass
        
        Args:
            core: Core patients tensor (X_core_dim, seq_len, input_dim)
            x: Batch patients tensor (batch_size, seq_len, input_dim)
            y: Target labels (batch_size, seq_len)
            padding_mask: Padding mask (batch_size, seq_len)
            edge_index: Graph edge indices (2, num_edges)
            nots: Notes tensor (batch_size, NOTE_DIM)
            bios: Bios tensor (batch_size, num_of_bios)
        
        Returns:
            predictions: Mortality predictions (batch_size, seq_len, 1)
        """
        batch_size = x.size(0)

        if self.gnn_flag:
            all_patients = torch.cat([x, self.X_core], dim=0) 
            total_patients = batch_size + self.X_core.shape[0]
            
            graph_input = all_patients.view(total_patients * self.seq_len, -1)
            for gat_layer in self.gat_layers:
                graph_input = F.leaky_relu(gat_layer(graph_input, edge_index), negative_slope=0.2)
            
            graph_output = graph_input.view(total_patients, self.seq_len, -1)
            batch_output = graph_output[:batch_size]  
        else:
            batch_output = x
        modality_outputs = []
        if self.use_x:
            lengths = (padding_mask.to(bool)).sum(dim=1)                       # (batch,)
            lengths = lengths.clamp(min=1).cpu()

            mask_index = padding_mask.sum(dim=1).long() - 1 
            mask_expanded = padding_mask.unsqueeze(-1)    
            gru_output, _ = self.gru(batch_output)

            if not self.gnn_flag:
                out = gru_output[torch.arange(gru_output.size(0)), mask_index, :]
            else:
                out = torch.cat([
                    gru_output[torch.arange(gru_output.size(0)), mask_index, :],
                    (gru_output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1),
                    (gru_output * mask_expanded).max(dim=1)[0]
                ], dim=-1)
            
            modality_outputs.append(out)

        if self.use_notes and self.notes_layer is not None:
            nots_processed = F.relu(self.notes_layer(nots))
            modality_outputs.append(nots_processed)
            
        if self.use_bios and self.bios_layer is not None:
            bios_processed = F.relu(self.bios_layer(bios))
            modality_outputs.append(bios_processed)
            
        if self.use_prescriptions and self.pres_layer is not None:
            _, drug_ids, offsets = self.collect_bags(list(enumerate(prescriptions)))
            pres_processed = F.relu(self.pres_layer(drug_ids.to(DEVICE), offsets.to(DEVICE)))
            modality_outputs.append(pres_processed)
        
        # Concatenate all selected modality outputs
        X_concat = torch.cat(modality_outputs, dim=-1)
        predictions_mort = self.classifier_mort(X_concat)  # (batch_size, 1)
        predictions_re = self.classifier_re(X_concat)  # (batch_size, 1)
        predictions_pro = self.classifier_pro(X_concat)  # (batch_size, 1)
        predictions = torch.concat([predictions_mort, predictions_pro, predictions_re], dim=1)
        return predictions


    def train_all(self, dataloaders, datasets, epochs: int = 10, learning_rate: float = 1e-3, pos_lambda : float = 1):
        self.train()
        optim = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        # Add a learning rate scheduler (ReduceLROnPlateau based on validation AP)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.33, patience=3, min_lr=1e-6)
        best_validation_ap = - float('inf')
        losses = []
        for i in range(datasets['train'].y.shape[1]):
            targets = datasets['train'].y[:,i] # Get the max target for each patien
            pos_weight = (targets == 0).sum() / (targets == 1).sum()  # Adjust pos_weight as needed
            losses.append(nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight*pos_lambda))
            print(f'Pos weight {i}: {pos_weight:.4f}')


        for epoch in range(epochs):
            print(f'Starting epoch {epoch + 1}/{epochs}')
            total = 0
            for x, y, padding_mask, idx, notes, bios, prescriptions in tqdm(dataloaders['train'], file=sys.stdout):
                optim.zero_grad()

                x, padding_mask, y, notes, bios = x.to(DEVICE), padding_mask.to(DEVICE), y.to(DEVICE), notes.to(DEVICE), bios.to(DEVICE)
                edge_index = datasets['train'].get_edge_index(x, padding_mask, idx).to(DEVICE)
                predictions = self.forward(x, padding_mask, edge_index, notes, bios, prescriptions)
                loss = losses[0](predictions[:, 0], y[:, 0]) + \
                       losses[1](predictions[:, 1], y[:, 1]) + \
                       losses[2](predictions[:, 2], y[:, 2])
                loss.backward()
                optim.step()
                
                total += loss.item()
            avg_loss = total / len(dataloaders['train'])
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
            validation_results = self.validate(dataloaders['val'], datasets['val'], calibrate=False)
            print(f'Val Mortality - AUC: {validation_results[0]:.4f} | AP: {validation_results[1]:.4f}')
            print(f'Val Prolonged LOS - AUC: {validation_results[2]:.4f} | AP: {validation_results[3]:.4f}')
            print(f'Val Readmission - AUC: {validation_results[4]:.4f} | AP: {validation_results[5]:.4f}')

            scheduler.step(validation_results[1])

            if validation_results[1] > best_validation_ap:
                best_validation_ap = validation_results[1]
                self.best_model = self.state_dict().copy()
                print("Best model updated")
        self.load_state_dict(self.best_model)
        val_results = self.validate(dataloaders['val'], datasets['val'], return_predictions=True, calibrate=False)
        self.initialize_calibrator(val_results[0], val_results[1])

    def validate(self, dataloader, dataset, return_predictions=False, labels_exist=True, calibrate=True):
        is_train = self.training
        if is_train:
            self.eval()
        with torch.no_grad():
            all_true_labels = {i: [] for i in range(3)}
            all_predicted_labels = {i: [] for i in range(3)}
            if labels_exist:
                for x, y, padding_mask, idx, notes, bios, prescriptions in tqdm(dataloader, file=sys.stdout):
                    x, padding_mask, y, notes, bios = x.to(DEVICE), padding_mask.to(DEVICE), y.to(DEVICE), notes.to(DEVICE), bios.to(DEVICE)
                    edge_index = dataset.get_edge_index(x, padding_mask, idx).to(DEVICE)
                    predictions = self.forward(x, padding_mask, edge_index, notes, bios, prescriptions)
                    for i in range(3):
                        all_true_labels[i].extend(y[:, i].cpu().numpy())
                        all_predicted_labels[i].extend(torch.sigmoid(predictions[:, i]).cpu().numpy().flatten())
            else:
                for x, padding_mask, idx, notes, bios, prescriptions in tqdm(dataloader, file=sys.stdout):
                    x, padding_mask, notes, bios = x.to(DEVICE), padding_mask.to(DEVICE), notes.to(DEVICE), bios.to(DEVICE)
                    edge_index = dataset.get_edge_index(x, padding_mask, idx).to(DEVICE)
                    predictions = self.forward(x, padding_mask, edge_index, notes, bios, prescriptions)
                    for i in range(3):
                        all_predicted_labels[i].extend(torch.sigmoid(predictions[:, i]).cpu().numpy().flatten())
                   
        calibrated_preds = {}
        for i in range(3):
            raw_preds = np.array(all_predicted_labels[i])
            labels = np.array(all_true_labels[i]) if labels_exist else None
            calibrated_preds[i] = self.calibrate_predictions(labels, raw_preds, i) if calibrate else raw_preds
        
        if is_train:
            self.train()
        if return_predictions:
            if labels_exist:
                return all_true_labels, calibrated_preds
            else:
                return calibrated_preds
        
        return (
            roc_auc_score(all_true_labels[0], calibrated_preds[0]), average_precision_score(all_true_labels[0], calibrated_preds[0]),
            roc_auc_score(all_true_labels[1], calibrated_preds[1]), average_precision_score(all_true_labels[1], calibrated_preds[1]),
            roc_auc_score(all_true_labels[2], calibrated_preds[2]), average_precision_score(all_true_labels[2], calibrated_preds[2]),
        )
      
        
    def save_model(self, filepath: str, ):
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__.__name__,
            'model_config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'n1_gat_layers': self.n1_gat_layers,
                'n2_gru_layers': self.n2_gru_layers,
                'num_heads': self.num_heads,
                'seq_len': self.seq_len,
                'gnn_flag': self.gnn_flag,
                'num_of_bios': self.num_of_bios,
                'num_prescriptions': self.num_prescriptions,
                'bios_hidden_dim': self.bios_hidden_dim,
                'pres_hidden_dim': self.pres_hidden_dim,
                'k': self.k,
                'X_core': self.X_core,
                'core_padding_mask': self.core_padding_mask,
                'use_notes': self.use_notes,
                'use_bios': self.use_bios,
                'use_prescriptions': self.use_prescriptions,
                'use_x': self.use_x
            }
        }
    
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath, device=None):
        if device is None:
            device = DEVICE
            
        checkpoint = torch.load(filepath, map_location=device)
        
        config = checkpoint['model_config']
        
        model = cls(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            n1_gat_layers=config['n1_gat_layers'],
            n2_gru_layers=config['n2_gru_layers'],
            X_core=config['X_core'],
            core_padding_mask=config['core_padding_mask'],
            num_of_bios=config['num_of_bios'],
            num_prescriptions=config['num_prescriptions'],
            bios_hidden_dim=config['bios_hidden_dim'],
            pres_hidden_dim=config['pres_hidden_dim'],
            num_heads=config['num_heads'],
            seq_len=config['seq_len'],
            k=config['k'],
            gnn_flag=config['gnn_flag'],
            use_notes=config.get('use_notes', True),
            use_bios=config.get('use_bios', True),
            use_prescriptions=config.get('use_prescriptions', True),
            use_x=config.get('use_x', True)
        )
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
            
        return model
    