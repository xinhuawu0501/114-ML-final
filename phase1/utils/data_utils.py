import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datetime import datetime
import math
import os
import matplotlib.pyplot as plt
import re
from datetime import timedelta

import pandas as pd
import re

def filter_notes(notes_df: pd.DataFrame, admission_text_only=False) -> pd.DataFrame:
    """
    Keep only Discharge Summaries and filter out Newborn admissions. Replace duplicates and join reports with
    their addendums. If admission_text_only is True, filter all sections that are not known at admission time.
    """
    # strip texts from leading and trailing and white spaces
    notes_df["text"] = notes_df["text"].str.strip()

    # remove entries without subject id or text
    notes_df = notes_df.dropna(subset=["subject_id", "text"])

    if admission_text_only:
        # reduce text to admission-only text
        notes_df = filter_admission_text(notes_df)

    return notes_df


def filter_admission_text(notes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter text information by section and only keep sections that are known on admission time.
    """
    admission_sections = {
        "chief_complaint": "chief complaint:",
        "present_illness": "present illness:",
        "medical_history": "medical history:",
        "medication_adm": "medications on admission:",
        "allergies": "allergies:",
        "physical_exam": "physical exam:",
        "family_history": "family history:",
        "social_history": "social history:"
    }

    # replace linebreak indicators
    # notes_df['text'] = notes_df['text'].str.replace("\n", "\\n")
    notes_df['text'] = notes_df['text'].str.replace("___\nFamily History:", "___\n\nFamily History:", flags=re.IGNORECASE)

    # extract each section by regex
    for key, section in admission_sections.items():
        notes_df[key] = notes_df.text.str.extract('{}([\\s\\S]+?)\n\\s*?\n[^(\\\\|\\d|\\.)]+?:'.format(section), flags=re.IGNORECASE)

        notes_df[key] = notes_df[key].str.replace('\n', ' ')
        notes_df[key] = notes_df[key].str.strip()
        notes_df[key] = notes_df[key].fillna("")
        notes_df.loc[notes_df[key].str.startswith("[]"), key] = ""

    # filter notes with missing main information
    notes_df = notes_df[(notes_df.chief_complaint != "") | (notes_df.present_illness != "") |
                        (notes_df.medical_history != "")]

    # add section headers and combine into TEXT_ADMISSION
    notes_df = notes_df.assign(text="CHIEF COMPLAINT: " + notes_df.chief_complaint.astype(str)
                                    + '\n\n' +
                                    "PRESENT ILLNESS: " + notes_df.present_illness.astype(str)
                                    + '\n\n' +
                                    "MEDICAL HISTORY: " + notes_df.medical_history.astype(str)
                                    + '\n\n' +
                                    "MEDICATION ON ADMISSION: " + notes_df.medication_adm.astype(str)
                                    + '\n\n' +
                                    "ALLERGIES: " + notes_df.allergies.astype(str)
                                    + '\n\n' +
                                    "PHYSICAL EXAM: " + notes_df.physical_exam.astype(str)
                                    + '\n\n' +
                                    "FAMILY HISTORY: " + notes_df.family_history.astype(str)
                                    + '\n\n' +
                                    "SOCIAL HISTORY: " + notes_df.social_history.astype(str))['text']

    return notes_df


icu_pattern = r"icu|ccu"

def is_icu(careunit):
    return re.search(icu_pattern, careunit, re.IGNORECASE)

def map_careunit(careunit):
    surgery = ['Med/Surg', 'Surgery', 'Medical/Surgical (Gynecology)','Med/Surg/GYN', 'Surgery/Pancreatic/Biliary/Bariatric', 'Surgery/Trauma', 'Med/Surg/Trauma','Cardiology Surgery Intermediate', 'Cardiac Surgery', 'Thoracic Surgery', 'PACU']
    labor_delivery = ['Labor & Delivery', 'Obstetrics (Postpartum & Antepartum)', 'Obstetrics Postpartum' ,'Obstetrics Antepartum']
    cardiology = [ 'Cardiology', 'Medicine/Cardiology Intermediate', 'Medicine/Cardiology']
    oncology = ['Hematology/Oncology', 'Hematology/Oncology Intermediate']
    observation = ['Emergency Department Observation', 'Observation']
    neurology = ['Neurology', 'Neuro Intermediate','Neuro Stepdown']

    if is_icu(careunit):
        careunit = 'icu'
    if careunit in surgery:
        careunit = "surgery"
    elif careunit in labor_delivery:
        careunit = 'obstetrics'
    elif careunit in cardiology:
        careunit = 'cardiology'
    elif careunit in oncology:
        careunit = 'oncology'
    elif careunit in observation:
        careunit = 'observation'
    elif careunit in neurology:
        careunit = 'neurology'
    else:
        careunit = careunit.lower()
    return careunit

def map_pr_class(row: pd.Series) -> int:
    if is_icu(row['careunit']):
        return 1
    elif row['icu_in_24h']:
        return 2
    else:
        # assigned to other wards within 24h 
        if not row['death_in_24h']:
            return 3
        else: 
            return 4
    

def icu_in_24h_after_other_wards(transfer_seq: list[tuple]) -> bool:
    ed_admission_time = transfer_seq[0][1]
    first_cu_after_ed = transfer_seq[1][0]
    
    # Skip if first transfer is already ICU
    if is_icu(first_cu_after_ed):
        return False

    return any(
        is_icu(cu) and (time <= ed_admission_time + timedelta(hours=24))
        for cu, time in transfer_seq[1:]  # skip first transfer
    )

def label_transfers(transfer_seq, admissions: pd.DataFrame) -> pd.DataFrame:
    '''
        add label:
        - ICU within 24 hr: 0 or 1
        - mortality within 24 hr: 0 or 1
    '''
    results = []
    admissions['deathtime'] = pd.to_datetime(admissions['deathtime'], format='%Y-%m-%d %H:%M:%S')

    for hadm_id, transfers in transfer_seq.items():
        went_icu_24h = icu_in_24h_after_other_wards(transfers)
        ed_admission_time = transfers[0][1]
        first_careunit_after_ed = transfers[1][0]
        deathtime = admissions.loc[hadm_id]['deathtime']
        death_in_24h = deathtime <= ed_admission_time + timedelta(hours=24)
        
        results.append({
            "hadm_id": hadm_id,
            "careunit": first_careunit_after_ed,
            "icu_in_24h": int(went_icu_24h),
            "death_in_24h": int(death_in_24h)
        })


    result_df = pd.DataFrame(results).set_index("hadm_id")
    result_df.to_csv("./labeled_transfer_w_mort.csv")

    return result_df

def log_statistic(result_df):
    for col in result_df.columns:
        print(f"\nColumn: {col}")
        print(result_df[col].value_counts(normalize=True) * 100)


def create_patient_routing(transfers, notes, output_path):
    admit_events = transfers.copy()
    admit_events.index = admit_events.index.astype(int)
    
    log_statistic(admit_events)
    if isinstance(notes, pd.Series):
        notes = notes.reset_index()
        notes.columns = ['hadm_id', 'text']

    admit_events['class'] = admit_events.apply(map_pr_class, axis=1)
    admit_events.to_csv('./map_classes.csv')
    
    log_statistic(admit_events)
    # Merge: hadm_id -> admit_events.index
    routing_notes = notes.merge(
        admit_events,
        how="inner", # only keep hadm_id present in BOTH tables
        left_on="hadm_id",
        right_index=True
    )

    routing_notes = routing_notes.set_index("hadm_id")

    # --- Extract X and y with indices preserved ---
    X = routing_notes["text"]
    y = routing_notes["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    train_df = pd.DataFrame({"text": X_train, "class": y_train})
    test_df = pd.DataFrame({"text": X_test, "class": y_test})

    test_df, val_df = np.array_split(test_df, 2)

    os.makedirs(output_path, exist_ok=True)
    train_df.to_csv(os.path.join(output_path, "train.csv"), index=True)
    test_df.to_csv(os.path.join(output_path, "test.csv"), index=True)
    val_df.to_csv(os.path.join(output_path, "val.csv"), index=True)

    return 

def get_transfer_seq_after_ed(transfer):
    sorted_transfers = transfer.copy()
    sorted_transfers = transfer.sort_values(by=['hadm_id', 'intime'])
    is_discharge_event = sorted_transfers['eventtype'] == 'discharge'
    sorted_transfers.loc[is_discharge_event, 'careunit'] = 'Discharge'

    first_transfer = sorted_transfers.groupby('hadm_id').head(1)
    ED_UNITS = ['Emergency Department'] # Adjust this list if other ED units exist in your data

    ed_admissions_df = first_transfer[first_transfer['careunit'].isin(ED_UNITS)]

    ed_hadm_ids = ed_admissions_df.index.unique() # Since transfers is indexed by hadm_id
    ed_patient_transfers = sorted_transfers[sorted_transfers.index.isin(ed_hadm_ids)]

    ed_patient_transfers.to_csv('./ed_patient_transfers.csv')
    ed_patient_transfers['careunit'] = ed_patient_transfers['careunit'].apply(map_careunit)

    transfer_sequences_with_times = (
        ed_patient_transfers
        .groupby('hadm_id')[['careunit', 'intime']]  # <-- Select BOTH columns here
        .apply(lambda x: list(zip(x['careunit'], pd.to_datetime(x['intime'], format='%Y-%m-%d %H:%M:%S'))))
    )

    return transfer_sequences_with_times


def create_pr_data_csv():
    '''
    load:
        - `transfers.csv.gz` for transfer record
        - `discharge.csv` for discharge summary
    merge 2 table and export to csv file under dataset/pr 
    '''
    data_dir = "../../dataset"
    split = ["hosp", "note"]

    transfer = pd.read_csv(data_dir + f"/{split[0]}/transfers.csv.gz", index_col="hadm_id")
    discharge = pd.read_csv(data_dir + f"/{split[1]}/discharge.csv", index_col="hadm_id")
    admissions = pd.read_csv(data_dir + f"/{split[0]}/admissions.csv.gz", index_col="hadm_id")
    notes = filter_notes(discharge, True)
    
    notes = notes.reset_index()  # index becomes a column
    notes.columns = ['hadm_id', 'text']  # rename columns properly

    transfer_seq = get_transfer_seq_after_ed(transfer)
    labeled_t = label_transfers(transfer_seq=transfer_seq, admissions=admissions)

    create_patient_routing(labeled_t, notes, os.path.join(data_dir, 'pr'))

    return


# create_pr_data_csv()

#===================== Analysis utility =================================================
def misclassified_ratio(prediction: pd.DataFrame, num_classes=4):
    counts = prediction['true_label'].value_counts()
    class_cols = [f'class_{i}_prob' for i in range(num_classes)]

    pred_class = prediction[class_cols].idxmax(axis=1).str.extract("(\d+)").astype(int)[0]

    for i in range(num_classes):
        misclassified = prediction[(prediction['true_label'] == i) & (pred_class != i)]
        total = counts.get(i, 0)
        
        print(f'class {i} has {misclassified.shape[0]} misclassified out of {total} samples')

        if total > 0:
            print(f'misclassified ratio: {misclassified.shape[0] / total:.4f}')
        else:
            print("misclassified ratio: N/A (class not present)")

