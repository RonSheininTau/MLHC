import sys
from pathlib import Path
from collections import Counter
import re
import numpy as np
import pandas as pd
import math
from NoteEmbedder import run_embeeding
from Model import GraphGRUMortalityMode
from Dataset import PatientDataset
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

def transform_unseen_data(merged, scaler,baseline_df):

    np.random.seed(0)
    merged_clean = merged.reset_index(drop=True)

    #Split to train & test (all data of a single patient needs to be in the same group)
    X = merged_clean
    X["icustay_id"] = X["icustay_id"].astype(float)
    num_cols = X.select_dtypes(include='float').columns
    X.loc[:, num_cols] = scaler.transform(X[num_cols])
    X.loc[:, num_cols] = X[num_cols].fillna(baseline_df)


    to_drop = ['hadm_id','icustay_id','intime','outtime','admittime', 'dischtime', 'dod','dob', 'mort', 'los_hosp_hr', 'charttime','adm_to_death']
    to_keep = ~X.columns.isin(to_drop)
    to_keep = X.columns[to_keep]
    X = X[to_keep]

    
    return X

def inferance_query(subject_ids, lavbevent_meatdata, vital_meatdata, con):
  with tempfile.TemporaryDirectory() as td:
    print(td)
    download_url = f'https://drive.google.com/uc?id=1ocAMGK0ppVJqKClASCkmuRZdqY3IQBEV'
    output_path = f'{td}/data.pkl'
    gdown.download(download_url, output_path, quiet=False)
  
    download_url = f'https://drive.google.com/uc?id=1zZ0pAJ9ASexmNS43WVKuHiQQTyVNnVe0'
    output_path = f'{td}/graph_gru_mortality_model.pt'
    gdown.download(download_url, output_path, quiet=False)
  
    model = GraphGRUMortalityModel.load_model(f'{td}/graph_gru_mortality_model.pt',device)

    with open(f"{td}/data.pkl", "rb") as f:
      data = pickle.load(f)


    hosps =  con.execute(preprocess.ICUQ, [subject_ids]).fetchdf().rename(str.lower, axis='columns')
    lab = con.execute(preprocess.LABQUERY, [lavbevent_meatdata['itemid'].tolist()]).fetchdf().rename(str.lower, axis='columns')

    vit = (con.execute(
            preprocess.VITQUERY,
            [vital_meatdata['itemid'].tolist(), subject_ids]
          ).fetchdf().rename(str.lower, axis='columns'))
    
    hosps = preprocess.ethnicity_to_ohe(hosps)
    merged = preprocess.exclude_and_merge(hosps, lab, vit, lavbevent_meatdata, vital_meatdata)
    min_max_df = merged.groupby('subject_id')[["charttime","admittime"]].agg({"admittime":"min","charttime":'max'}).reset_index()
    min_max_df.columns = ['subject_id', 'min_charttime', 'max_charttime']

    min_max_df["min_charttime"] = pd.to_datetime(min_max_df["min_charttime"])
    min_max_df["max_charttime"] = pd.to_datetime(min_max_df["max_charttime"])
    con.register("time_windows", min_max_df)

    notes = (
      con.execute(preprocess.NOTES, [subject_ids])  
        .fetchdf()
        .rename(str.lower, axis="columns")
    )

    notes = (
        notes.sort_values(['subject_id', 'charttime'])
        .groupby('subject_id', sort=False)['text']
        .apply(lambda s: '\n'.join(x.strip() for x in s.dropna().astype(str) if x.strip()))
        .reset_index()
    )

    bios = (
      con.execute(preprocess.BIOQUERY, [subject_ids])   # list of ints, e.g. [3,4,9,11,12]
        .fetchdf()
        .rename(str.lower, axis="columns")
    )

    meds = (
    con.execute(preprocess.MEDS, [subject_ids])   # list of ints, e.g. [3,4,9,11,12]
      .fetchdf()
      .rename(str.lower, axis="columns")
    )

    notes["embeddings"] = run_embeeding(notes)
    gc.collect()
    torch.cuda.empty_cache()

    scaler = data["scaler"]
    baseline_df = data["baseline_df"]
    X = preprocess.transform_unseen_data(merged, scaler,baseline_df)
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


    y_test = pd.read_csv("./Models/y_test.csv")


    labels = torch.tensor(y_test[['mort_30day', 'prolonged_stay', 'readmission_30day']].values, dtype=torch.float32).to(device)
    dataset=PatientDataset(padded_tensor,labels,data["X_core"],padding_mask, data["padding_mask_core"],notes_df.embeddings.values.tolist(), bio_df.values >= 1, pres_df.values)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    res = model.validate(dataloader, dataset)
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
  raise NotImplementedError('You need to implement this function')
