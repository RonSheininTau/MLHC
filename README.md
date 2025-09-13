#CONTACT: Contextual and Temporal attention based framework for predicting clinical Trajectory

![Model Architecture](Model.png)

## Overview

This repository contains a comprehensive machine learning system for predicting critical healthcare outcomes in ICU patients using multi-modal data. The system combines Graph Neural Networks (GNNs) with Gated Recurrent Units (GRUs) to process temporal clinical data, patient notes, biosignals, and prescription information for mortality, prolonged length of stay, and readmission prediction.

## Key Features

- **Multi-Modal Architecture**: Integrates temporal clinical data, clinical notes, biosignals, and prescription data
- **Graph Neural Networks**: Uses Graph Attention Networks (GAT) to model patient relationships
- **Temporal Modeling**: Employs GRU layers for sequential data processing
- **Multi-Task Learning**: Simultaneously predicts mortality, prolonged stay, and readmission
- **Calibration**: Includes prediction calibration for reliable probability estimates
- **Ablation Studies**: Comprehensive analysis of different modality contributions

## Model Architecture

The `GraphGRUMortalityModel` combines several key components:

1. **Graph Attention Layers**: Process patient relationships using TransformerConv layers
2. **GRU Layers**: Handle temporal sequences of clinical measurements
3. **Multi-Modal Fusion**: Integrates:
   - Clinical time series data (vitals, lab values)
   - Clinical notes (using pre-trained embeddings)
   - Biosignals (binary features)
   - Prescription data (embedding bag representation)
4. **Multi-Task Classifiers**: Separate prediction heads for each outcome

## Dataset

The model is trained on MIMIC-IV data with the following preprocessing:

- **Patient Selection**: First admissions, age 18-89, minimum 24h stay
- **Temporal Resolution**: 4-hour prediction windows
- **Features**: 18 time steps of clinical measurements
- **Targets**: 30-day mortality, prolonged stay (>7 days), 30-day readmission

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Access to MIMIC-IV database

```

## Usage

### Training and Evaluation

For training and evaluating the model, use the `Project_Evaluation.ipynb` notebook:

```bash
jupyter notebook Project_Evaluation.ipynb
```

This notebook contains:
- Complete data preprocessing pipeline
- Model training with hyperparameter optimization
- Comprehensive evaluation on test data
- Performance analysis and visualization

### Inference on New Data

For inference on new patient datasets, use the `unseen_data_evaluation.py` script:

```python
from unseen_data_evaluation import evaluate_unseen_data

# Evaluate on new patient data
results = evaluate_unseen_data(model_path, new_patient_data)
```

This script handles:
- Loading pre-trained models
- Preprocessing new patient data
- Generating predictions for mortality, prolonged stay, and readmission
- Outputting calibrated probability estimates

## Key Files

- `Model.py`: Main model architecture implementation
- `preprocess.py`: Data preprocessing pipeline
- `Dataset.py`: PyTorch dataset classes
- `unseen_data_evaluation.py`: Evaluation on new patient data
- `NoteEmbedder.py`: Clinical note embedding generation
- `Project_Evaluation.ipynb`: Model evaluation and analysis

## Model Performance

The model achieves strong performance across all three prediction tasks:

- **Mortality Prediction**: High AUC and precision-recall performance
- **Prolonged Stay**: Effective identification of patients requiring extended care
- **Readmission Risk**: Accurate prediction of 30-day readmission probability

## Ablation Study Results

The ablation study demonstrates the contribution of each modality:

- **Clinical Time Series**: Core temporal patterns
- **Clinical Notes**: Rich semantic information
- **Biosignals**: Additional physiological indicators
- **Prescriptions**: Medication-based insights

## Calibration

The model includes prediction calibration using logistic regression to ensure reliable probability estimates for clinical decision-making.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mlhc2024,
  title={Multi-Modal Graph Neural Networks for ICU Outcome Prediction},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MIMIC-III database for providing the clinical data
- PyTorch Geometric for GNN implementations
- The healthcare AI research community

## Contact

For questions or collaborations, please contact [your-email@domain.com].

---

**Note**: This model is for research purposes only and should not be used for direct clinical decision-making without proper validation and regulatory approval.
