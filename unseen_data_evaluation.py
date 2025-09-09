import sys
from pathlib import Path
from collections import Counter
import re
import numpy as np
import pandas as pd
import math
from NoteEmbedder import run_embeeding
from Model import GraphGRUMortalityModel
from Dataset import PatientDataset, PatientDatasetUnseen
import preprocess
import os 
import tempfile
from collections import Counter
import re
import numpy as np
import pandas as pd
import math
import gdown
import pickle 
import gc
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transform_unseen_data(merged, scaler, baseline_df):
    """
    Transform unseen patient data using the same preprocessing pipeline as training data.
    
    This function applies the same scaling and normalization transformations that were used
    during model training to ensure consistency between training and inference data.
    
    Args:
        merged (pandas.DataFrame): Raw merged patient data containing clinical measurements,
                                  demographics, and temporal information.
        scaler (sklearn.preprocessing.StandardScaler): Fitted scaler from training data
        baseline_df (pandas.Series): Baseline values for filling missing data
        
    Returns:
        pandas.DataFrame: Preprocessed and scaled data ready for model inference
        
    Note:
        - Sets random seed for reproducibility
        - Removes target variables and metadata columns
        - Applies the same feature scaling as training data
        - Fills missing values with baseline statistics
    """
    np.random.seed(0)
    merged_clean = merged.reset_index(drop=True)

    # Split to train & test (all data of a single patient needs to be in the same group)
    X = merged_clean
    X["icustay_id"] = X["icustay_id"].astype(float)
    num_cols = X.select_dtypes(include='float').columns
    X.loc[:, num_cols] = scaler.transform(X[num_cols])
    X.loc[:, num_cols] = X[num_cols].fillna(baseline_df)

    # Remove target variables and metadata columns
    to_drop = ['hadm_id','icustay_id','intime','outtime','admittime', 'dischtime', 'dod','dob', 'mort', 'los_hosp_hr', 'charttime','adm_to_death']
    to_keep = ~X.columns.isin(to_drop)
    to_keep = X.columns[to_keep]
    X = X[to_keep]

    return X


def download_data():
    """
    Download the trained model and preprocessing data from Google Drive.
    
    This function downloads the pre-trained GraphGRUMortalityModel and associated
    preprocessing artifacts (scaler, baseline values, core patient data) from
    Google Drive storage. The downloads are performed in a temporary directory
    to avoid cluttering the local filesystem.
    
    Returns:
        tuple: A tuple containing:
            - model (GraphGRUMortalityModel): Loaded and ready-to-use trained model
            - data (dict): Dictionary containing preprocessing artifacts:
                - 'scaler': Fitted StandardScaler for feature normalization
                - 'baseline_df': Baseline values for missing data imputation
                - 'X_core': Core patient data for graph construction
                - 'padding_mask_core': Padding masks for core patients
                - 'orgs': List of organism names for biomarker processing
                - 'drugs': List of drug names for prescription processing
                
    Note:
        - Uses temporary directory for downloads to avoid local file pollution
        - Downloads are performed from Google Drive using gdown
        - Model is automatically loaded onto the appropriate device (CPU/GPU)
        - Memory cleanup is performed after loading
    """
    with tempfile.TemporaryDirectory() as td:
        print(f"Downloading data to temporary directory: {td}")
        
        # Download necessary data from traninig
        download_url = f'https://drive.google.com/uc?id=1ocAMGK0ppVJqKClASCkmuRZdqY3IQBEV'
        output_path = f'{td}/data.pkl'
        gdown.download(download_url, output_path, quiet=False)
      
        # Download trained model
        download_url = f'https://drive.google.com/uc?id=1zZ0pAJ9ASexmNS43WVKuHiQQTyVNnVe0'
        output_path = f'{td}/graph_gru_mortality_model.pt'
        gdown.download(download_url, output_path, quiet=False)

        download_url = f'https://drive.google.com/uc?id=1pr3APIiwTALAA5jSMyFgkzG5GOYwJ8DE'
        output_path = f'{td}/lab_metadata.csv'
        gdown.download(download_url, output_path, quiet=False)

        download_url = f'https://drive.google.com/uc?id=11Jq0OrfC8JQou3ngA0puMptUHYqw546I'
        output_path = f'{td}/vital_metadata.csv'
        gdown.download(download_url, output_path, quiet=False)

        # Load model
        model = GraphGRUMortalityModel.load_model(f'{td}/graph_gru_mortality_model.pt', device)

        # Load preprocessing data
        with open(f"{td}/data.pkl", "rb") as f:
            data = pickle.load(f)

        data['lavbevent_meatdata'] = pd.read_csv(f'{td}/lab_metadata.csv')
        data['vital_meatdata'] = pd.read_csv(f'{td}/vital_metadata.csv')
        
        return model, data


def exeute_basic_query(subject_ids, lavbevent_meatdata, vital_meatdata, con):
    """
    Execute basic data queries to extract core clinical data for unseen patients.
    
    This function retrieves the fundamental clinical data required for model inference,
    including hospital admissions, laboratory results, and vital signs. It applies
    the same preprocessing steps used during training to ensure data consistency.
    
    Args:
        subject_ids (list): List of subject IDs for the unseen patients
        lavbevent_meatdata (pandas.DataFrame): Metadata for laboratory events
        vital_meatdata (pandas.DataFrame): Metadata for vital sign events
        con: Database connection object (e.g., BigQuery client)
        
    Returns:
        pandas.DataFrame: Merged and preprocessed clinical data containing:
            - Demographics and admission information
            - Laboratory test results
            - Vital sign measurements
            - One-hot encoded ethnicity information
            
    Note:
        - Applies the same SQL queries used during training
        - Converts column names to lowercase for consistency
        - Merges data from multiple sources (hospital, lab, vital signs)
        - Applies ethnicity one-hot encoding
    """
    # Execute hospital admission queries
    hosps = con.execute(preprocess.ICUQ, [subject_ids]).fetchdf().rename(str.lower, axis='columns')
    
    # Execute laboratory result queries
    lab = con.execute(preprocess.LABQUERY, [lavbevent_meatdata['itemid'].tolist()]).fetchdf().rename(str.lower, axis='columns')
    
    # Execute vital signs queries
    vit = (con.execute(preprocess.VITQUERY, [vital_meatdata['itemid'].tolist(), subject_ids]).fetchdf().rename(str.lower, axis='columns'))
    
    # Apply preprocessing transformations
    hosps = preprocess.ethnicity_to_ohe(hosps)
    merged = preprocess.exclude_and_merge(hosps, lab, vit, lavbevent_meatdata, vital_meatdata)
    
    return merged


def execute_extra_modalities_query(subject_ids, merged, con):
    """
    Execute queries to extract additional multi-modal data for unseen patients.
    
    This function retrieves supplementary data modalities that enhance model predictions,
    including clinical notes, biomarker results, and prescription information. It processes
    temporal data to ensure proper alignment with the main clinical timeline.
    
    Args:
        subject_ids (list): List of subject IDs for the unseen patients
        merged (pandas.DataFrame): Previously merged clinical data for temporal reference
        con: Database connection object (e.g., BigQuery client)
        
    Returns:
        tuple: A tuple containing three DataFrames:
            - notes (pandas.DataFrame): Clinical notes data with concatenated text per patient
            - bios (pandas.DataFrame): Biomarker/microbiology results
            - meds (pandas.DataFrame): Prescription/medication data
            
    Note:
        - Calculates temporal bounds from existing clinical data
        - Concatenates multiple clinical notes per patient into single text
        - Applies consistent column naming conventions
        - Handles missing and empty text entries
    """
    # Calculate temporal bounds for each patient
    min_max_df = merged.groupby('subject_id')[["charttime","admittime"]].agg({"admittime":"min","charttime":'max'}).reset_index()
    min_max_df.columns = ['subject_id', 'min_charttime', 'max_charttime']

    min_max_df["min_charttime"] = pd.to_datetime(min_max_df["min_charttime"])
    min_max_df["max_charttime"] = pd.to_datetime(min_max_df["max_charttime"])
    con.register("time_windows", min_max_df)

    # Execute clinical notes query
    notes = (
        con.execute(preprocess.NOTES, [subject_ids])  
          .fetchdf()
          .rename(str.lower, axis="columns")
    )

    # Process and concatenate clinical notes per patient
    notes = (
          notes.sort_values(['subject_id', 'charttime'])
          .groupby('subject_id', sort=False)['text']
          .apply(lambda s: '\n'.join(x.strip() for x in s.dropna().astype(str) if x.strip()))
          .reset_index()
      )
    
    # Execute biomarker/microbiology query
    bios = (
      con.execute(preprocess.BIOQUERY, [subject_ids])
        .fetchdf()
        .rename(str.lower, axis="columns")
    )
    
    # Execute medication/prescription query
    meds = (
      con.execute(preprocess.MEDS, [subject_ids])
        .fetchdf()
        .rename(str.lower, axis="columns")
    )
    
    return notes, bios, meds


def inferance_query(subject_ids, con):
    """
    Execute the complete inference pipeline for unseen patient data.
    
    This function orchestrates the entire inference process, from data extraction
    to prediction generation. It handles all aspects of the pipeline including
    model loading, data preprocessing, feature extraction, and batch inference.
    
    Args:
        subject_ids (list): List of subject IDs for the unseen patients
        con: Database connection object (e.g., BigQuery client)
        
    Returns:
        dict: Dictionary containing prediction probabilities for each outcome:
            - Key 0: Mortality prediction probabilities
            - Key 1: Prolonged length of stay prediction probabilities  
            - Key 2: Readmission prediction probabilities
            
    Note:
        - Downloads model and preprocessing artifacts automatically
        - Performs memory cleanup after embedding generation
        - Uses batch processing for efficient inference
        - Applies the same preprocessing pipeline as training data
    """
    model, data = download_data() 


    merged = exeute_basic_query(subject_ids, data['lavbevent_meatdata'], data['vital_meatdata'], con)

    notes, bios, meds = execute_extra_modalities_query(subject_ids, merged, con)

    # Generate clinical note embeddings
    notes["embeddings"] = run_embeeding(notes)
    gc.collect()  # Clean up memory
    torch.cuda.empty_cache()  # Clear GPU cache

    scaler = data["scaler"]
    baseline_df = data["baseline_df"]
    X = transform_unseen_data(merged, scaler, baseline_df)
    padded_tensor, padding_mask = preprocess.generate_series_data(X, group_col="subject_id", maxlen=18)

    notes["subject_id"] = notes["subject_id"].astype(int)
    bios["subject_id"] = bios["subject_id"].astype(int)
    meds["subject_id"] = meds["subject_id"].astype(int)
    meds["hadm_id"] = meds["hadm_id"].astype(int)

    orgs = data["orgs"][:-1]
    orgs = list(map(lambda x: x.replace('org_', '').replace('.0',''), orgs))
    drugs = data["drugs"]
    drugs = list(map(lambda x: x.replace('drug_', ''), drugs))

    notes_df = preprocess.load_notes_embeddings(merged, notes=notes, path=None)
    bio_df = preprocess.process_bios(merged, bios=bios, path=None, orgs=orgs)
    pres_df = preprocess.process_prescriptions(merged, prescriptions=meds, path=None,drugs=drugs)

    # Create dataset and dataloader for inference
    dataset = PatientDatasetUnseen(padded_tensor, data["X_core"], padding_mask, data["padding_mask_core"], 
                                 notes_df.embeddings.values.tolist(), bio_df.values >= 1, pres_df.values)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    # Perform inference
    res = model.inference(dataloader, dataset)
    return res


def run_pipeline_on_unseen_data(subject_ids ,client):
  """
  Run your full pipeline, from data loading to prediction.

    :param subject_ids: A list of subject IDs of an unseen test set.
    :type subject_ids: List[int]

  :param client: A BigQuery client object for accessing the MIMIC-III dataset.
  :type client: google.cloud.bigquery.client.Client

    :return: DataFrame with the following columns:
                - subject_id: Subject IDs, which in some cases can be different due to your analysis.
                - mortality_proba: Prediction probabilities for mortality.
                - prolonged_LOS_proba: Prediction probabilities for prolonged length of stay.
                - readmission_proba: Prediction probabilities for readmission.
    :rtype: pandas.DataFrame
    """
  try:
      predictions = inferance_query(subject_ids, client)
      
      results_df = pd.DataFrame({
          'subject_id': subject_ids[:len(predictions[0])],  # Ensure we have matching lengths
          'mortality_proba': predictions[0],  # Mortality predictions
          'prolonged_LOS_proba': predictions[1],  # Prolonged LOS predictions
          'readmission_proba': predictions[2]  # Readmission predictions
      })
      
      return results_df
        
  except Exception as e:
        print(f"Error in run_pipeline_on_unseen_data: {str(e)}")
