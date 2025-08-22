import sys
import pandas as pd
import numpy as np
from tqdm import tqdm 
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

sys.modules["numpy._core.numeric"] = np.core.numeric  

MIN_TARGET_ONSET = 2*24  # minimal time of target since admission (hours)
MIN_LOS = 24  # minimal length of stay (hours)
PREDICT_FREQ = '4H'  # frequency of prediction (1 hour)

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

VITQUERY = f"""--sql
        SELECT chartevents.subject_id::INTEGER AS subject_id\
             , chartevents.hadm_id::INTEGER AS hadm_id\
             , chartevents.charttime::DATE AS charttime\
             , chartevents.itemid::INTEGER AS itemid\
             , chartevents.valuenum::DOUBLE AS valuenum\
             , admissions.admittime::DATE AS admittime\
        FROM chartevents
                 INNER JOIN admissions
                            ON chartevents.subject_id = admissions.subject_id
                                AND chartevents.hadm_id = admissions.hadm_id
                                AND chartevents.charttime::DATE between
                                   (admissions.admittime::DATE)
                                   AND (admissions.admittime::DATE + interval 48 hour)
                                AND itemid::INTEGER in ?
      -- exclude rows marked as error
      AND chartevents.error::INTEGER IS DISTINCT \
        FROM 1 \
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
    
    hosps_sorted['next_admittime'] = hosps_sorted.groupby('subject_id')['admittime'].shift(-1)
    days_between = (hosps_sorted['next_admittime'] - hosps_sorted['dischtime']).dt.total_seconds() / (24 * 3600)
    hosps_sorted['readmission_30day'] = ((days_between > 0) & (days_between <= 30)).astype(int)
    
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
    hosps = hosps[~((hosps['mort'].astype(bool)) & (hours_to_death < MIN_TARGET_ONSET))]
    print(f"4. Exclude patients who died within {MIN_TARGET_ONSET}-hours of admission: N={hosps.shape[0]}")
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


def train_test_split(merged, labels_df):

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
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def cluster_and_select_subjects(X_train, num_clusters=10, random_state=42):
    """
    Calculate the first row of each subject_id in X_train, cluster it to num_clusters 
    and choose one subject_id from each cluster.
    
    Parameters:
    X_train: DataFrame with subject_id column
    num_clusters: int, number of clusters to create
    random_state: int, for reproducibility
    
    Returns:
    list: selected subject_ids, one from each cluster
    """
    first_rows = X_train.groupby('subject_id').first().reset_index()
    
    features_for_clustering = first_rows.drop('subject_id', axis=1)
    

    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(features_for_clustering)
    

    first_rows['cluster'] = cluster_labels
    

    np.random.seed(random_state)
    selected_subjects = []
    
    for cluster_id in range(num_clusters):
        cluster_subjects = first_rows[first_rows['cluster'] == cluster_id]['subject_id'].values
        if len(cluster_subjects) > 0:
            selected_subject = np.random.choice(cluster_subjects)
            selected_subjects.append(selected_subject)
    
    return selected_subjects


def load_notes_embeddings(merged, path='data/notes_with_embeddings.pkl'):
# Create an alias so pickle can find the old path

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


def process_bios(merged, path='data/bios.csv', threshold=240):
    bios = pd.read_csv(path)
    bios = bios.loc[bios.hadm_id.isin(merged.hadm_id) & bios.subject_id.isin(merged.subject_id)]
    item_counts = bios[["subject_id", "org_itemid"]].drop_duplicates()["org_itemid"].value_counts()
    valid_items = item_counts[item_counts > 240].index
    bios_filtered = bios[bios["org_itemid"].isin(valid_items)]
    bios_merge = merged[['subject_id', 'hadm_id']].drop_duplicates().merge(
        bios_filtered,
        on=['subject_id', 'hadm_id'],
        how='left'
    ).fillna("Nonn")

    bios_onehot = pd.get_dummies(bios_merge, columns=['org_itemid'], prefix='org')
    groupby_cols = ['subject_id', 'hadm_id']
    onehot_cols = [col for col in bios_onehot.columns if col.startswith('org_')]
    bios_table = bios_onehot.groupby(groupby_cols)[onehot_cols].sum().reset_index().drop("hadm_id", axis=1).set_index("subject_id")

    return bios_table



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


def preprocess_pipeline(path=r'./data'):

    subject_ids, lab_event_metadata, vital_metadata, labs, vits, hosps = load_data(path)
    labels_df = create_labels(hosps)
    hosps = ethnicity_to_ohe(hosps)
    merged = exclude_and_merge(hosps, labs, vits, lab_event_metadata, vital_metadata)
    X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(merged, labels_df)
    selected_subjects = cluster_and_select_subjects(X_train, num_clusters=100, random_state=42)

    notes_df = load_notes_embeddings(merged, path=os.path.join(path, 'notes_with_embeddings.pkl'))

    selected_subjects = cluster_and_select_subjects(X_train, num_clusters=240, random_state=42)

    X_core = X_train[X_train['subject_id'].isin(selected_subjects)]
    y_core = y_train[y_train['subject_id'].isin(selected_subjects)]

    # Update X_train to exclude the selected subjects
    X_train = X_train[~X_train['subject_id'].isin(selected_subjects)]
    y_train = y_train[~y_train['subject_id'].isin(selected_subjects)]




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
    }


if __name__ == "__main__":
    preprocess_pipeline()
