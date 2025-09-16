import sys
import pandas as pd
import numpy as np
import pickle
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
import duckdb
import gdown
# from NoteEmbedder import embed_long_texts
sys.modules["numpy._core.numeric"] = np.core.numeric  
os.environ['OPENBLAS_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '16'


MIN_TARGET_ONSET = 2*24  # minimal time of target since admission (hours)
MIN_LOS = 24  # minimal length of stay (hours)
PREDICT_FREQ = '4H'  # frequency of prediction (1 hour)
GAP_HOURS = 6  # gap hours for prediction
SCALE_META_FEATURES = True
ICUQ = \
"""--sql
    SELECT admissions.subject_id::INTEGER AS subject_id, admissions.hadm_id::INTEGER AS hadm_id
    , admissions.admittime::DATE AS admittime, admissions.dischtime::DATE AS dischtime
    , admissions.ethnicity, admissions.deathtime::DATE AS deathtime
    , patients.gender, patients.dob::DATE AS dob, icustays.icustay_id::INTEGER AS icustay_id, patients.dod::DATE as dod,
    icustays.intime::DATE AS intime,icustays.outtime::DATE AS outtime
    FROM admissions
    INNER JOIN patients
        ON admissions.subject_id = patients.subject_id
    LEFT JOIN icustays
        ON admissions.hadm_id = icustays.hadm_id

    WHERE admissions.has_chartevents_data = 1
    AND admissions.subject_id::INTEGER IN ?
    ORDER BY admissions.subject_id, admissions.hadm_id, admissions.admittime;
"""


LABQUERY = \
f"""--sql
    SELECT labevents.subject_id::INTEGER AS subject_id\
        , labevents.hadm_id::INTEGER AS hadm_id\
        , labevents.charttime::DATE AS charttime
        , labevents.itemid::INTEGER AS itemid\
        , labevents.valuenum::DOUBLE AS valuenum
        , admissions.admittime::DATE AS admittime
    FROM labevents
            INNER JOIN admissions
                        ON labevents.subject_id = admissions.subject_id
                            AND labevents.hadm_id = admissions.hadm_id
                            AND labevents.charttime::DATE between
                                (admissions.admittime::DATE)
                                AND (admissions.admittime::DATE + interval 48 hour)
                            AND itemid::INTEGER IN ? \
                            """

VITQUERY = f"""
  --sql
  SELECT
    ce.subject_id::INTEGER AS subject_id,
    ce.hadm_id::INTEGER    AS hadm_id,
    ce.charttime           AS charttime,
    ce.itemid::INTEGER     AS itemid,
    ce.valuenum::DOUBLE    AS valuenum,
    a.admittime::DATE      AS admittime
  FROM chartevents AS ce
    JOIN admissions AS a
      ON ce.subject_id = a.subject_id
      AND ce.hadm_id    = a.hadm_id
  WHERE ce.charttime::DATE BETWEEN a.admittime::DATE
                              AND a.admittime::DATE + INTERVAL 48 HOUR
    AND ce.itemid::INTEGER     IN (SELECT * FROM UNNEST(?))
    AND ce.subject_id::INTEGER IN (SELECT * FROM UNNEST(?))
    AND ce.error::INTEGER IS DISTINCT FROM 1
  """

NOTES = """
SELECT n.subject_id, n.charttime, n.text
FROM noteevents AS n
JOIN admissions  AS a ON n.subject_id = a.subject_id        -- or: ON n.hadm_id = a.hadm_id (safer, avoids dupes)
JOIN time_windows AS w ON w.subject_id = a.subject_id
WHERE a.subject_id::INTEGER IN (SELECT * FROM UNNEST(?))     -- or drop this line if w already filters subjects
  AND n.charttime::DATE BETWEEN w.min_charttime::DATE AND w.max_charttime::DATE
ORDER BY n.subject_id, n.charttime
"""

MEDS = """
SELECT n.subject_id,n.hadm_id, n.startdate, n.enddate, n.drug, n.drug_name_poe, n.drug_name_generic, n.dose_val_rx, n.dose_unit_rx
FROM prescriptions AS n
JOIN admissions AS a ON n.subject_id = a.subject_id        -- or: ON n.hadm_id = a.hadm_id (safer, avoids dupes)
JOIN time_windows AS w ON w.subject_id = a.subject_id
WHERE a.subject_id::INTEGER IN (SELECT * FROM UNNEST(?))     -- or drop this line if w already filters subjects
  AND n.startdate::DATE BETWEEN w.min_charttime::DATE AND w.max_charttime::DATE
ORDER BY n.subject_id, n.startdate
"""

BIOQUERY = \
f"""--sql
SELECT microbiologyevents.subject_id::INTEGER AS subject_id\
      , microbiologyevents.hadm_id::INTEGER AS hadm_id\
      , microbiologyevents.charttime::DATE AS charttime
      , microbiologyevents.spec_itemid::INTEGER AS spec_itemid\
      , microbiologyevents.org_itemid::INTEGER AS org_itemid\

FROM microbiologyevents INNER JOIN admissions
                    ON microbiologyevents.subject_id = admissions.subject_id
                        AND microbiologyevents.hadm_id = admissions.hadm_id
                        AND microbiologyevents.charttime::DATE between
                            (admissions.admittime::DATE)
                            AND (admissions.admittime::DATE + interval 48 hour)
                        AND admissions.subject_id::INTEGER IN ? \
                        """

def load_data(path = r"./data"):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    subject_ids = pd.read_csv(os.path.join(path, 'initial_cohort.csv'))['subject_id'].to_list()
    lab_event_metadata = pd.read_csv(os.path.join(path, 'labs_metadata.csv'))
    vital_metadata = pd.read_csv(os.path.join(path, 'vital_metadata.csv'))
    labs = pd.read_csv('data/labs.csv')
    vits = pd.read_csv('data/vits.csv')
    hosps = pd.read_csv('data/icu.csv')

    for col in ['admittime', 'dischtime', 'dob', 'dod', 'intime', 'outtime']:
        hosps[col] = pd.to_datetime(hosps[col].str.strip(), errors='coerce')
    
    return subject_ids, lab_event_metadata, vital_metadata, labs, vits, hosps

def donwload_data(path=r'./data'):
    if not os.path.exists(path):
        try:
            print(f"Downloading data to {path}")
            os.makedirs(path, exist_ok=True)
            download_url = f'https://drive.google.com/uc?id=1pT9iDdDWP1PbOEt1I2xtBF8oOlZQLtDq'
            output_path = os.path.join(path, 'icu.csv')

            gdown.download(download_url, output_path, quiet=False)
            download_url = f'https://drive.google.com/uc?id=18t5EUNKtMtfz9YMdF1xkFfRxuUXpPlNb'
            output_path = os.path.join(path, 'labs.csv')
            gdown.download(download_url, output_path, quiet=False)

            download_url = f'https://drive.google.com/uc?id=1zyF_FavDyTYu1URNf81d6-aVcrJmj_lT'
            output_path = os.path.join(path, 'vits.csv')
            gdown.download(download_url, output_path, quiet=False)

            download_url = f'https://drive.google.com/uc?id=1Q4uCqN4XjoqAp5wnpLsTsIrayLSrGPVZ'
            output_path = os.path.join(path, 'notes_with_embeddings_fast.pkl')
            gdown.download(download_url, output_path, quiet=False)

            download_url = f'https://drive.google.com/uc?id=1Le3FtXqfiWi03CRe8dEH2ySolkYQkLQ1'
            output_path = os.path.join(path, 'prescriptions.csv')
            gdown.download(download_url, output_path, quiet=False)

            download_url = f'https://drive.google.com/uc?id=1wrlUCGZr8Gib17CbC4nLRG42xCtE783o'
            output_path = os.path.join(path, 'bios.csv')
            gdown.download(download_url, output_path, quiet=False)

            download_url = f'https://drive.google.com/uc?id=1pr3APIiwTALAA5jSMyFgkzG5GOYwJ8DE'
            output_path = f'{td}/lab_metadata.csv'
            gdown.download(download_url, output_path, quiet=False)

            download_url = f'https://drive.google.com/uc?id=11Jq0OrfC8JQou3ngA0puMptUHYqw546I'
            output_path = f'{td}/vital_metadata.csv'
            gdown.download(download_url, output_path, quiet=False)

        except Exception as e:
            print(f"Error downloading data: {e}")


def create_labels(hosps):
    """
    Create three labels for each subject_id and hadm_id:
    1. 30-day mortality (died within 30 days of discharge or during admission)
    2. Prolonged stay (length of stay > 7 days)
    3. 30-day readmission (readmitted within 30 days of discharge)
    """
    
    hosps_sorted = hosps.sort_values(['subject_id', 'admittime'])[['subject_id','hadm_id', 'admittime','dischtime','dod']].drop_duplicates().copy()
    hosps_sorted['los_hosp_hr'] = (hosps_sorted['dischtime'] - hosps_sorted['admittime']).dt.total_seconds()/3600
    hosps_sorted['mort_30day'] = 0
    
    died_during_admission = (~hosps_sorted['dod'].isna()) & (hosps_sorted['dod'] <= hosps_sorted['dischtime'])
    hosps_sorted.loc[died_during_admission, 'mort_30day'] = 1
    days_to_death_post_discharge = (hosps_sorted['dod'] - hosps_sorted['dischtime']).dt.total_seconds() / (24 * 3600)
    died_within_30_days = (~hosps_sorted['dod'].isna()) & (days_to_death_post_discharge <= 30) & (days_to_death_post_discharge >= 0)
    hosps_sorted.loc[died_within_30_days, 'mort_30day'] = 1
    
    hosps_sorted['prolonged_stay'] = (hosps_sorted['los_hosp_hr'] > 7 * 24).astype(int)
    hosps_sorted['readmission_30day'] = 0
    #hosps_sorted = hosps_sorted.sort_values(["subject_id", "admittime"])

    # next admission time per subject
    hosps_sorted["next_admittime"] = hosps_sorted.groupby("subject_id")["admittime"].shift(-1)

    # time until next admission
    delta = hosps_sorted["next_admittime"] - hosps_sorted["dischtime"]

    # 30-day readmission label on the *current* stay (1 if patient returns within 30 days after this discharge)
    hosps_sorted["readmission_30day"] = delta.between(pd.Timedelta(days=1), pd.Timedelta(days=30), inclusive="right").astype(int)

    return hosps_sorted[['subject_id', 'hadm_id', 'mort_30day', 'prolonged_stay', 'readmission_30day']]


def ethnicity_to_ohe(hosps):
    # ethnicity  - to category
    hosps.ethnicity = hosps.ethnicity.str.lower()
    hosps.loc[(hosps.ethnicity.str.contains('^white')),'ethnicity'] = 'white'
    hosps.loc[(hosps.ethnicity.str.contains('^black')),'ethnicity'] = 'black'
    hosps.loc[(hosps.ethnicity.str.contains('^hisp')) | (hosps.ethnicity.str.contains('^latin')),'ethnicity'] = 'hispanic'
    hosps.loc[(hosps.ethnicity.str.contains('^asia')),'ethnicity'] = 'asian'
    hosps.loc[~(hosps.ethnicity.str.contains('|'.join(['white', 'black', 'hispanic', 'asian']))),'ethnicity'] = 'other'

    # ethnicity - one hot encoding
    hosps['eth_white'] = (hosps['ethnicity'] == 'white').astype(int)
    hosps['eth_black'] = (hosps['ethnicity'] == 'black').astype(int)
    hosps['eth_hispanic'] = (hosps['ethnicity'] == 'hispanic').astype(int)
    hosps['eth_asian'] = (hosps['ethnicity'] == 'asian').astype(int)
    hosps['eth_other'] = (hosps['ethnicity'] == 'other').astype(int)
    hosps.drop(['ethnicity', 'deathtime'], inplace=True, axis=1)
    return hosps

def age(admittime, dob):
    if admittime < dob:
      return 0
    return admittime.year - dob.year - ((admittime.month, admittime.day) < (dob.month, dob.day))


def exclude_and_merge(hosps, labs, vits, lavbevent_meatdata, vital_meatdata):

    """ Exclude patients based on criteria and merge lab and vital data with hospital data.
    Args:
        hosps (pd.DataFrame): Hospital data containing patient admissions.
        labs (pd.DataFrame): Laboratory data.
        vits (pd.DataFrame): Vital signs data.
        lavbevent_meatdata (pd.DataFrame): Metadata for lab events.
        vital_meatdata (pd.DataFrame): Metadata for vital signs.
    Returns:
        pd.DataFrame: Merged DataFrame with hospital, lab, and vital data.
    """

    hosps['age'] = hosps.apply(lambda row: age(row['admittime'], row['dob']), axis=1)
    hosps['los_hosp_hr'] = (hosps.dischtime - hosps.admittime).dt.total_seconds()/3600
    hosps['mort'] = np.where(~np.isnat(hosps.dod),1,0)

    # Gender to binary
    hosps['gender'] = np.where(hosps['gender']=="M", 1, 0)

    # @title Q1.1 - Patient Exclusion Criteria
    hosps = hosps.sort_values('admittime').groupby('subject_id').first().reset_index()
    print(f"1. Include only first admissions: N={hosps.shape[0]}")

    hosps = hosps[hosps.age.between(18,90)]
    print(f"2. Exclusion by ages: N={hosps.shape[0]}")

    # Exclude patients hospitalized for less than 24 hours
    hosps = hosps[hosps['los_hosp_hr'] >= MIN_LOS]
    print(f"3. Include only patients who admitted for at least {MIN_LOS} hours: N={hosps.shape[0]}")

    # Exclude patients that died in the first 48 hours of admission
    hours_to_death = (hosps['dod'] - hosps['admittime']).dt.total_seconds() / 3600
    hosps = hosps[~((hosps['mort'].astype(bool)) & (hours_to_death < MIN_TARGET_ONSET + GAP_HOURS))]
    print(f"4. Exclude patients who died within {MIN_TARGET_ONSET + GAP_HOURS}-hours of admission: N={hosps.shape[0]}")
    labs = labs[labs['hadm_id'].isin(hosps['hadm_id'])]

    labs = pd.merge(labs,lavbevent_meatdata,on='itemid')
    labs = labs[labs['valuenum'].between(labs['min'],labs['max'],  inclusive='both')]

    vits = vits[vits['hadm_id'].isin(hosps['hadm_id'])]
    vits = pd.merge(vits,vital_meatdata,on='itemid')
    vits = vits[vits['valuenum'].between(vits['min'],vits['max'], inclusive='both')]

    vits.loc[(vits['feature name'] == 'TempF'),'valuenum'] = (vits[vits['feature name'] == 'TempF']['valuenum']-32)/1.8
    vits.loc[vits['feature name'] == 'TempF','feature name'] = 'TempC'

    merged = pd.concat([vits, labs])
    merged['charttime'] = pd.to_datetime(merged['charttime'], errors='coerce')

    pivot = pd.pivot_table(merged, index=['subject_id', 'hadm_id', pd.Grouper(key='charttime', freq=PREDICT_FREQ)],
                        columns=['feature name'], values='valuenum', aggfunc=['mean', 'max', 'min', 'std'])
    pivot.columns = [f'{c[1]}_{c[0]}' for c in pivot.columns.to_flat_index()]

    # temp = merged.copy()

    merged = pd.merge(hosps, pivot.reset_index(), on=['subject_id', 'hadm_id'])
    merged[pivot.columns] = merged.groupby(['subject_id', 'hadm_id'])[pivot.columns].ffill()

    merged = merged.sort_values(['subject_id', 'hadm_id', 'charttime'])
    labs_features_names = set(labs['feature name'])
    vits_features_names = set(vits['feature name'])
    labs_features = [col for col in merged.columns if col.split('_')[0] in labs_features_names]
    vits_features = [col for col in merged.columns if col.split('_')[0] in vits_features_names]

    lab_diff_cols = {}
    for col in labs_features:
        if col.find("mean") >= 0:
            base = merged.groupby(['subject_id', 'hadm_id'])[col].transform('first')
            lab_diff_cols[f'{col}_diff'] = merged[col] - base

    lab_diff_df = pd.DataFrame(lab_diff_cols)

    vital_diff_cols = {}
    for col in vits_features:
        if col.find("mean") >= 0:
            diff_series = merged.groupby(['subject_id', 'hadm_id'])[col].diff()
            vital_diff_cols[f'{col}_diff'] = diff_series

        vital_diff_df = pd.DataFrame(vital_diff_cols)

        # Concatenate back to original DataFrame
    merged = pd.concat([merged, lab_diff_df, vital_diff_df], axis=1)

    merged['charttime'] = pd.to_datetime(merged['charttime'], errors='coerce')

    return merged


def train_test_split(merged, labels_df, scale_meta_features=SCALE_META_FEATURES):

    np.random.seed(0)
    merged_clean = merged.reset_index(drop=True)

    #Split to train & test (all data of a single patient needs to be in the same group)
    X = merged_clean
    X = X.merge(labels_df, on=['subject_id', 'hadm_id'], how='inner')
    groups = merged_clean['subject_id']

    gss = GroupShuffleSplit(n_splits=1, train_size=.8, test_size=0.1)
    train_index, test_index = next(gss.split(X, groups=groups))
    val_index = list(set(X.index.to_list()) - (set(train_index.tolist()) | set(test_index.tolist())))

    X_train = X.iloc[train_index]
    X_val = X.iloc[val_index]
    X_test = X.iloc[test_index]

    y_train = X_train[["subject_id","mort_30day","prolonged_stay","readmission_30day"]].drop_duplicates()
    y_train = y_train.groupby('subject_id',as_index=False).max()

    X_train.drop(columns=["mort_30day","prolonged_stay","readmission_30day"], axis=1, inplace=True)

    y_val = X_val[["subject_id","mort_30day","prolonged_stay","readmission_30day"]].drop_duplicates()
    y_val = y_val.groupby('subject_id',as_index=False).max()
    X_val.drop(columns=["mort_30day","prolonged_stay","readmission_30day"], axis=1, inplace=True)

    y_test = X_test[["subject_id","mort_30day","prolonged_stay","readmission_30day"]].drop_duplicates()
    y_test = y_test.groupby('subject_id',as_index=False).max()
    X_test.drop(columns=["mort_30day","prolonged_stay","readmission_30day"], axis=1, inplace=True)

    if scale_meta_features:
        num_cols = X_train.select_dtypes(include=['float','int']).columns.drop(["subject_id","hadm_id","icustay_id","mort"])
    else:
        num_cols = X_train.select_dtypes(include='float').columns
    scaler = StandardScaler()

    X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val.loc[:, num_cols] = scaler.transform(X_val[num_cols])
    X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

    baseline_df = X_train[X_train.charttime.dt.date == X_train.admittime.dt.date].mean(axis=0).fillna(0)
    X_train.loc[:, num_cols] = X_train[num_cols].fillna(baseline_df)
    X_val.loc[:, num_cols] = X_val[num_cols].fillna(baseline_df)
    X_test.loc[:, num_cols] = X_test[num_cols].fillna(baseline_df)

    to_drop = ['hadm_id','icustay_id','intime','outtime','admittime', 'dischtime', 'dod','dob', 'mort', 'los_hosp_hr', 'charttime','adm_to_death']
    to_keep = ~X_train.columns.isin(to_drop)
    to_keep = X_train.columns[to_keep]
    X_train = X_train[to_keep]
    X_test = X_test[to_keep]
    X_val = X_val[to_keep]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, baseline_df


    
def cluster_and_select_subjects(X_train, num_clusters=10, random_state=42):
    """
    Calculate the first row of each subject_id in X_train, cluster it to num_clusters 
    and choose the subject_id closest to each cluster centroid.
    
    Parameters:
    X_train: DataFrame with subject_id column
    num_clusters: int, number of clusters to create
    random_state: int, for reproducibility
    
    Returns:
    list: selected subject_ids, one closest to each cluster centroid
    """
    first_rows = X_train.groupby('subject_id').first().reset_index()
    
    features_for_clustering = first_rows.drop('subject_id', axis=1)
    
    # Normalize features to ensure each index has equal effect on distance calculation

    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(features_for_clustering)
    first_rows['cluster'] = cluster_labels
    

    selected_subjects = []
    
    for cluster_id in range(num_clusters):
        cluster_mask = first_rows['cluster'] == cluster_id
        cluster_subjects = first_rows[cluster_mask]['subject_id'].values
        cluster_features = features_for_clustering[cluster_mask]
        
        if len(cluster_subjects) > 0:
            # Calculate distances from each point in cluster to the centroid
            centroid = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            
            # Select the subject with minimum distance to centroid
            closest_idx = np.argmin(distances)
            selected_subject = cluster_subjects[closest_idx]
            selected_subjects.append(selected_subject)
    
    return selected_subjects



def load_notes_embeddings(merged,notes=None, path='data/notes_with_embeddings.pkl'):
# Create an alias so pickle can find the old path

    if notes is None:
      with open(path, "rb") as f:
          notes = pickle.load(f)
    notes_ordered = merged[['subject_id']].drop_duplicates().merge(
    notes[["subject_id","embeddings"]], 
      on='subject_id', 
      how='left'
    )

    embeddings_dict = {}

    for idx, row in notes_ordered.iterrows():
        subject_id = row['subject_id']
        embeddings = row['embeddings']
        try:
            if isinstance(embeddings, np.ndarray) :
                # Convert to tensor if it's not already
                if not isinstance(embeddings, torch.Tensor):
                    embeddings = torch.tensor(embeddings, dtype=torch.float32)
                
                # Perform average pooling across the sequence dimension
                # Assuming embeddings shape is (sequence_length, embedding_dim)
                pooled_embedding = torch.mean(embeddings, dim=0)  # Shape: (embedding_dim,)
                embeddings_dict[subject_id] = pooled_embedding
            elif isinstance(embeddings, list):
                embeddings_dict[subject_id] = torch.tensor(embeddings, dtype=torch.float32)
            else:
                # Handle missing embeddings with zero vector
                # Assuming embedding dimension is 768 (common for transformers)
                embeddings_dict[subject_id] = torch.zeros(768, dtype=torch.float32)
        except Exception as e:
            flag = 1

    # Convert to a tensor where each row corresponds to a subject_id
    subject_ids_list = notes_ordered['subject_id'].tolist()
    pooled_embeddings = [embeddings_dict[subject_id] for subject_id in subject_ids_list]

    print(f"Pooled embeddings shape: {len(pooled_embeddings)}")
    print(f"Number of subjects: {len(subject_ids_list)}")

    notes_df = pd.DataFrame({
        'subject_id': subject_ids_list,
        'embeddings': pooled_embeddings}).set_index('subject_id')
    
    return notes_df


def process_bios(merged, bios=None, path='data/bios.csv', threshold=240, orgs=None):
    if bios is None:
      bios = pd.read_csv(path)
    bios = bios.loc[bios.hadm_id.isin(merged.hadm_id) & bios.subject_id.isin(merged.subject_id)]
    bios["org_itemid"] = bios["org_itemid"].astype(str)
    if orgs is None:
      item_counts = bios[["subject_id", "org_itemid"]].drop_duplicates()["org_itemid"].value_counts()
      valid_items = item_counts[item_counts > threshold].index
      bios_filtered = bios[bios["org_itemid"].isin(valid_items)]
    else:
      bios_filtered = bios[bios["org_itemid"].isin(orgs)]
    bios_merge = merged[['subject_id', 'hadm_id']].drop_duplicates().merge(
        bios_filtered,
        on=['subject_id', 'hadm_id'],
        how='left'
    ).fillna("None")

    bios_onehot = pd.get_dummies(bios_merge, columns=['org_itemid'], prefix='org')
    groupby_cols = ['subject_id', 'hadm_id']
    onehot_cols = [col for col in bios_onehot.columns if col.startswith('org_')]
    bios_table = bios_onehot.groupby(groupby_cols)[onehot_cols].sum().reset_index().drop("hadm_id", axis=1).set_index("subject_id")

    return bios_table

def process_prescriptions(merged, prescriptions=None, path='data/prescriptions.csv', threshold=240, drugs=None):
    if prescriptions is None:
          prescriptions = pd.read_csv(path, index_col=0)      
    prescriptions = prescriptions.loc[prescriptions.hadm_id.isin(merged.hadm_id) & prescriptions.subject_id.isin(merged.subject_id)]
    if drugs is None:
      item_counts = prescriptions[["subject_id", "drug"]].drop_duplicates()["drug"].value_counts()
      valid_items = item_counts[item_counts > threshold].index
      prescriptions_filtered = prescriptions[prescriptions["drug"].isin(valid_items)][['subject_id', "hadm_id", 'drug']].drop_duplicates()
    else:
      prescriptions_filtered = prescriptions[prescriptions["drug"].isin(drugs)][['subject_id', "hadm_id", 'drug']].drop_duplicates()
    prescriptions_merge = merged[['subject_id', 'hadm_id']].drop_duplicates().merge(
            prescriptions_filtered,
            on=['subject_id', 'hadm_id'],
            how='left'
        ).fillna("None")

    prescriptions_onehot = pd.get_dummies(prescriptions_merge, columns=['drug'], prefix='drug')
    groupby_cols = ['subject_id', 'hadm_id']
    onehot_cols = [col for col in prescriptions_onehot.columns if col.startswith('drug_')]
    prescriptions_table = prescriptions_onehot.groupby(groupby_cols)[onehot_cols].sum().reset_index().drop("hadm_id", axis=1).set_index("subject_id")


    return prescriptions_table

def generate_series_data(df, group_col="subject_id", maxlen=18):
  grouped = df.groupby(group_col)
  subject_sequences = [group.values[:, 1:] for _, group in grouped]
  padded_tensor = pad_sequences(subject_sequences, padding='post', dtype='float32')
  sequence_lengths = [len(seq) for seq in subject_sequences]
  padding_mask = np.zeros((len(sequence_lengths), maxlen), dtype=np.float32)
  for i, length in enumerate(sequence_lengths):
      padding_mask[i, :length] = 1.0
  padded_tensor = torch.tensor(padded_tensor, dtype=torch.float32)
  padding_mask = torch.tensor(padding_mask, dtype=torch.float32)
  return padded_tensor, padding_mask

        
def preprocess_pipeline(path=r'./data', num_clusters=240, scale_meta_features=SCALE_META_FEATURES):

    donwload_data(path)
    subject_ids, lab_event_metadata, vital_metadata, labs, vits, hosps = load_data(path)
    labels_df = create_labels(hosps)
    hosps = ethnicity_to_ohe(hosps)
    merged = exclude_and_merge(hosps, labs, vits, lab_event_metadata, vital_metadata)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, baseline_df = train_test_split(merged, labels_df, scale_meta_features=scale_meta_features)
    # con = duckdb.connect(f'./data/mimiciii.duckdb')

    notes_df = load_notes_embeddings(merged,path=os.path.join(path, 'notes_with_embeddings_fast.pkl'))

    selected_subjects = cluster_and_select_subjects(X_train, num_clusters=num_clusters, random_state=42)

    X_core = X_train[X_train['subject_id'].isin(selected_subjects)]
    y_core = y_train[y_train['subject_id'].isin(selected_subjects)]

    # Update X_train to exclude the selected subjects
    X_train = X_train[~X_train['subject_id'].isin(selected_subjects)]
    y_train = y_train[~y_train['subject_id'].isin(selected_subjects)]

    prescriptions_table = process_prescriptions(merged, path=os.path.join(path, 'prescriptions.csv'), threshold=240)
    pre_train = prescriptions_table.loc[X_train['subject_id'].drop_duplicates()]
    pre_val = prescriptions_table.loc[X_val['subject_id'].drop_duplicates()]
    pre_test = prescriptions_table.loc[X_test['subject_id'].drop_duplicates()]

    bio_df = process_bios(merged, path=os.path.join(path, 'bios.csv'), threshold=240)
    bio_train = bio_df.loc[X_train['subject_id'].drop_duplicates()]
    bio_val = bio_df.loc[X_val['subject_id'].drop_duplicates()]
    bio_test = bio_df.loc[X_test['subject_id'].drop_duplicates()]

    padded_tensor_train, padding_mask_train = generate_series_data(X_train, group_col="subject_id", maxlen=18)
    padded_tensor_core, padding_mask_core = generate_series_data(X_core, group_col="subject_id", maxlen=18)
    padded_tensor_val, padding_mask_val = generate_series_data(X_val, group_col="subject_id", maxlen=18)
    padded_tensor_test, padding_mask_test = generate_series_data(X_test, group_col="subject_id", maxlen=18)

    notes_df_train = notes_df.loc[X_train.subject_id.unique()]
    notes_df_val = notes_df.loc[X_val.subject_id.unique()]
    notes_df_test = notes_df.loc[X_test.subject_id.unique()]

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_core': X_core,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'baseline_df': baseline_df,
        'orgs': bio_df.columns.tolist(),
        'drugs': prescriptions_table.columns.tolist(),
        'y_core': y_core,
        'selected_subjects': selected_subjects,
        'notes_df_train': notes_df_train,
        'notes_df_val': notes_df_val,
        'notes_df_test': notes_df_test,
        'padded_tensor_train': padded_tensor_train,
        'padding_mask_train': padding_mask_train,
        'padded_tensor_val': padded_tensor_val,
        'padding_mask_val': padding_mask_val,
        'padded_tensor_test': padded_tensor_test,
        'padding_mask_test': padding_mask_test,
        'padded_tensor_core': padded_tensor_core,
        'padding_mask_core': padding_mask_core,
        'bio_train': bio_train,
        'bio_val': bio_val,
        'bio_test': bio_test,
        'prescriptions_train': pre_train,
        'prescriptions_val': pre_val,
        'prescriptions_test': pre_test,
    }

if __name__ == "__main__":
    preprocess_pipeline(num_clusters=100)
