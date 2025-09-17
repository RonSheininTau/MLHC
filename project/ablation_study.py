import os
import json
import preprocess
import torch
from Model import GraphGRUMortalityModel
from torch.utils.data import DataLoader
import Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model_for_ablation(datasets, model_config, modality_config):
    """
    Create a model with specific modality selection for ablation study.
    
    Args:
        model_config: Dictionary with model hyperparameters
        modality_config: Dictionary with modality configuration
        
    Returns:
        GraphGRUMortalityModel: Model configured with selected modalities
    """
    
    model = GraphGRUMortalityModel(
        input_dim=datasets['train'].X.shape[2],
        hidden_dim=model_config['hidden_dim'],
        n1_gat_layers=model_config['n1_gat_layers'],
        n2_gru_layers=model_config['n2_gru_layers'],
        X_core=datasets['train'].core,
        core_padding_mask=datasets['train'].padding_mask_core,
        num_of_bios=datasets['train'].bios.shape[1],
        num_prescriptions=datasets['train'].prescriptions.shape[1],
        use_notes=modality_config['use_notes'],    
        use_bios=modality_config['use_bios'],
        use_prescriptions=modality_config['use_prescriptions'],
        use_x=modality_config['use_x'],
        gnn_flag=True,
        k=model_config['k'],
        dropout=model_config['dropout'],    
        num_heads=model_config['num_heads'],
        seq_len=datasets['train'].X.shape[1],
    ).to(DEVICE)
    
    return model

def run_ablation_study(dataloaders, datasets, model_config):
    """
    Run a complete ablation study comparing different modality combinations.
    
    Args:
        dataloaders: Dictionary containing train/val/test dataloaders
        datasets: Dictionary containing train/val/test datasets
        model_config: Dictionary with model hyperparameters
        
    Returns:
        dict: Results for each configuration
    """
    
    results = {}

    ablation_configs = {
        'all_modalities': {
            'use_notes': True,
            'use_bios': True,
            'use_prescriptions': True,
            'use_x': True,
            'description': 'All modalities (baseline)'
        },
        'no_notes': {
            'use_notes': False,
            'use_bios': True,
            'use_prescriptions': True,
            'use_x': True,
            'description': 'Without notes'
        },
        'no_bios': {
            'use_notes': True,
            'use_bios': False,
            'use_prescriptions': True,
            'use_x': True,
            'description': 'Without bios'
        },
        'no_prescriptions': {
            'use_notes': True,
            'use_bios': True,
            'use_prescriptions': False,
            'use_x': True,
            'description': 'Without prescriptions'
        },
        'no_x': {
            'use_notes': True,
            'use_bios': True,
            'use_prescriptions': True,
            'use_x': False,
            'description': 'Without main sequential data'
        }
    }
    os.makedirs('ablation_models', exist_ok=True)
    for config_name, modality_config in ablation_configs.items():
        print(f"\n{'='*60}")
        print(f"TRAINING MODEL: {config_name.upper()}")
        print(f"Description: {modality_config['description']}")
        print(f"Modalities: Notes={modality_config['use_notes']}, "
              f"Bios={modality_config['use_bios']}, "
              f"Prescriptions={modality_config['use_prescriptions']}, "
              f"X={modality_config['use_x']}")
        print(f"{'='*60}")
        
        # Create model with specific modality configuration
        model = create_model_for_ablation(
            datasets=datasets,
            model_config=hyperparameters,
            modality_config=modality_config
        )
        
        # Train the model
        print("Training model...")
        model.train_all(
            dataloaders=dataloaders,
            datasets=datasets,
            epochs=10,
            learning_rate=hyperparameters['learning_rate'],
            pos_lambda=1
        )
        
        print("Evaluating model...")
        test_results = model.validate(
            dataloader=dataloaders['test'],
            dataset=datasets['test'],
            calibrate=True
        )
        
        # Store results
        results[config_name] = {
            'description': modality_config['description'],
            'modality_config': modality_config,
            'test_results': test_results
        }
        
        # Save model
        
        model_path = f"ablation_models/ablation_model_{config_name}.pth"
        model.save_model(model_path)
        
        # Print results
        print(f"\nTest Results for {config_name}:")
        print(f"  Mortality - AUC: {test_results[0]:.4f} | AP: {test_results[1]:.4f}")
        print(f"  Prolonged - AUC: {test_results[2]:.4f} | AP: {test_results[3]:.4f}")
        print(f"  Readmission - AUC: {test_results[4]:.4f} | AP: {test_results[5]:.4f}")
    
    return results


def compare_ablation_results(results):
    """
    Compare and display results from ablation study.
    
    Args:
        results: Dictionary containing results from run_ablation_study
    """
    
    print(f"\n{'='*80}")
    print("ABLATION STUDY RESULTS COMPARISON")
    print(f"{'='*80}")
    
    print(f"{'Configuration':<20} {'Mortality AP':<12} {'Prolong AP':<12} {'Readmission AP':<15}")
    print("-" * 80)
    
    for config_name, result in results.items():
        test_results = result['test_results']
        mortality_ap = test_results[1]
        prolonged_ap = test_results[3] 
        readmission_ap = test_results[5]
        
        print(f"{config_name:<20} {mortality_ap:<12.4f} {prolonged_ap:<12.4f} {readmission_ap:<15.4f}")
    
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    for config_name, result in results.items():
        print(f"\n{config_name.upper()}:")
        print(f"Description: {result['description']}")
        print(f"Modalities: {result['modality_config']}")
        
        test_results = result['test_results']
        print(f"Test Results:")
        print(f"  Mortality - AUC: {test_results[0]:.4f} | AP: {test_results[1]:.4f}")
        print(f"  Prolonged - AUC: {test_results[2]:.4f} | AP: {test_results[3]:.4f}")
        print(f"  Readmission - AUC: {test_results[4]:.4f} | AP: {test_results[5]:.4f}")

if __name__ == "__main__":
    hyperparameters = {'hidden_dim': 128, 'batch_size': 128, 'n1_gat_layers': 1, 'n2_gru_layers': 1, 'num_heads': 4, 
                       'dropout': 0.04033931265087129, 'learning_rate': 0.00015574186652855083, 
                       'pos_lambda': 0.5756532880616873, 'bios_hidden_dim': 32, 'pres_hidden_dim': 64, 'k': 5, 
                       'num_clusters': 240}
    data = preprocess.preprocess_pipeline(num_clusters=hyperparameters['num_clusters'])
    
    k = hyperparameters['k']

    train_labels = torch.tensor(data["y_train"][['mort_30day', 'prolonged_stay', 'readmission_30day']].values, dtype=torch.float32).to(DEVICE)
    val_labels = torch.tensor(data["y_val"][['mort_30day', 'prolonged_stay', 'readmission_30day']].values, dtype=torch.float32).to(DEVICE)
    test_labels = torch.tensor(data["y_test"][['mort_30day', 'prolonged_stay', 'readmission_30day']].values, dtype=torch.float32).to(DEVICE)


    batch_size = hyperparameters['batch_size']
    datasets = {x: Dataset.PatientDataset(d, y, core=data["padded_tensor_core"].to(DEVICE), padding_mask=m, padding_mask_core=data["padding_mask_core"].to(DEVICE), k=k ,notes=n, bios=b, prescriptions=p) for x, d, y, m, n, b, p in
            zip(['train', 'val', 'test'], [data["padded_tensor_train"].to(DEVICE), data["padded_tensor_val"].to(DEVICE), data["padded_tensor_test"].to(DEVICE)],
                [train_labels, val_labels, test_labels],
                [data["padding_mask_train"].to(DEVICE), data["padding_mask_val"].to(DEVICE), data["padding_mask_test"].to(DEVICE)],
                [data["notes_df_train"].embeddings.values.tolist(),
                data["notes_df_val"].embeddings.values.tolist(),
                data["notes_df_test"].embeddings.values.tolist()],
                [torch.tensor(data["bio_train"].values >= 1, dtype=torch.float32).to(DEVICE),
                torch.tensor(data["bio_val"].values >= 1, dtype=torch.float32).to(DEVICE),
                torch.tensor(data["bio_test"].values >= 1, dtype=torch.float32).to(DEVICE)],
                [torch.tensor(data["prescriptions_train"].values, dtype=torch.float32).to(DEVICE), 
                torch.tensor(data["prescriptions_val"].values, dtype=torch.float32).to(DEVICE), 
                torch.tensor(data["prescriptions_test"].values, dtype=torch.float32).to(DEVICE)])}
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'val', 'test']}
    print("Running ablation study...")
    
    # Run ablation study
    results = run_ablation_study(
        dataloaders=dataloaders,
        datasets=datasets,
        model_config=hyperparameters
    )
    
    compare_ablation_results(results)
    with open('ablation_results.json', 'w') as f:
        json.dump(results, f)
    

