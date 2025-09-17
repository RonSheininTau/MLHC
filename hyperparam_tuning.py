import optuna
from preprocess import preprocess_pipeline
import torch
from torch.utils.data import DataLoader
from Model import GraphGRUMortalityModel
from Dataset import PatientDataset

# Define device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def objective(trial):
    # Suggest hyperparameters
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    n1_gat_layers = trial.suggest_int('n1_gat_layers', 1, 2)
    n2_gru_layers = trial.suggest_int('n2_gru_layers', 1, 3)
    num_heads = trial.suggest_categorical('num_heads', [1, 2, 4])
    dropout = trial.suggest_float('dropout', 0.0, 0.2)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    # pos_lambda = trial.suggest_float('pos_lambda', 0.5, 2.0)
    pos_lambda = 1
    bios_hidden_dim = trial.suggest_categorical('bios_hidden_dim', [32, 64])
    pres_hidden_dim = trial.suggest_categorical('pres_hidden_dim', [32, 64])
    k = trial.suggest_int('k', 3, 11)
    num_clusters = trial.suggest_int('num_clusters', 100, 300)
    # gnn_flag = trial.suggest_categorical('gnn_flag', [True, False])

    data = preprocess_pipeline(num_clusters=num_clusters)
    train_labels = torch.tensor(data["y_train"][['mort_30day', 'prolonged_stay', 'readmission_30day']].values, dtype=torch.float32).to(DEVICE)
    val_labels = torch.tensor(data["y_val"][['mort_30day', 'prolonged_stay', 'readmission_30day']].values, dtype=torch.float32).to(DEVICE)

    core_padding_mask = data["padding_mask_core"]
    X_core = data["padded_tensor_core"]

    train_labels = torch.tensor(data["y_train"][['mort_30day', 'prolonged_stay', 'readmission_30day']].values, dtype=torch.float32).to(DEVICE)
    val_labels = torch.tensor(data["y_val"][['mort_30day', 'prolonged_stay', 'readmission_30day']].values, dtype=torch.float32).to(DEVICE)
    test_labels = torch.tensor(data["y_test"][['mort_30day', 'prolonged_stay', 'readmission_30day']].values, dtype=torch.float32).to(DEVICE)

    datasets = {x: PatientDataset(d, y, core=data["padded_tensor_core"], padding_mask=m, padding_mask_core=data["padding_mask_core"], k=k ,notes=n, bios=b, prescriptions=p) for x, d, y, m, n, b, p in
            zip(['train', 'val', 'test'], [data["padded_tensor_train"], data["padded_tensor_val"]],
                [train_labels, val_labels],
                [data["padding_mask_train"], data["padding_mask_val"]],
                [torch.stack(data["notes_df_train"].embeddings.values.tolist()),
                torch.stack(data["notes_df_val"].embeddings.values.tolist())],
                [torch.tensor(data["bio_train"].values >= 1, dtype=torch.float32).to(DEVICE),
                torch.tensor(data["bio_val"].values >= 1, dtype=torch.float32).to(DEVICE)],
                [torch.tensor(data["prescriptions_train"].values, dtype=torch.float32).to(DEVICE), 
                torch.tensor(data["prescriptions_val"].values, dtype=torch.float32).to(DEVICE)])}
    
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=(x == 'train')) for x in ['train', 'val']}
    torch.manual_seed(42)
    # Model instantiation
    model = GraphGRUMortalityModel(
        input_dim=datasets['train'].X.shape[-1],
        hidden_dim=hidden_dim,
        n1_gat_layers=n1_gat_layers,
        n2_gru_layers=n2_gru_layers,
        X_core=X_core,
        core_padding_mask=core_padding_mask,
        num_of_bios=data["bio_train"].shape[1],
        num_prescriptions=data["prescriptions_train"].shape[1],
        bios_hidden_dim=bios_hidden_dim if bios_hidden_dim is not None else hidden_dim,
        pres_hidden_dim=pres_hidden_dim if pres_hidden_dim is not None else hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        k=k,
        gnn_flag=True
    )
    model = model.to(DEVICE)

    # Train the model
    model.train_all(
        dataloaders,
        datasets,
        epochs=10,
        learning_rate=learning_rate,
        pos_lambda=pos_lambda
    )

    # Validate and return the negative AP (since Optuna minimizes)
    val_results = model.validate(dataloaders['val'], datasets['val'])
    val_ap = (val_results[1] + val_results[3] + val_results[5]) / 3
    return val_ap

def run_optuna_tuning(n_trials=20):
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial),
        n_trials=n_trials
    )
    print("Best trial:")
    print(study.best_trial)
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")
    return study

def main():

    # Uncomment and use after implementing data loading:
    study = run_optuna_tuning(
        n_trials=150
    )

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()

