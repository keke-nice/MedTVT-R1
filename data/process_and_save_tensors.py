"""
This script loads and preprocesses the Symile-MIMIC dataset splits, saving the
resulting tensors to split-specific directories in `data_dir`.
"""
import json
import os
import time

import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as t
from tqdm import tqdm
import wfdb
from scipy.signal import resample
import pdb
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_cxr(cxr_path, split='train'):
    """
    Loads and preprocesses a chest X-ray (CXR) image, including resizing and converting to tensor.
    """
    img = Image.open(cxr_path).convert('RGB')  
    # square crop
    if split == "train":
        crop = t.RandomCrop((224, 224))
    else:
        crop = t.CenterCrop((224, 224))

    transform = t.Compose([
        t.Resize(400),
        crop,
        t.ToTensor(),
        t.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform(img)


def get_ecg(ecg_path):
    """
    Loads and preprocesses an ECG signal.
    """
    ecg_signal = wfdb.rdsamp(ecg_path)  # files/p1205/p12054137/s40989841/40989841
    ecg = ecg_signal[0].T
    ecg = torch.from_numpy(resample(ecg, int(ecg.shape[1] / 5), axis=1))
    if torch.any(torch.isnan(ecg)):
        isnan = torch.isnan(ecg)
        ecg = torch.where(isnan, torch.zeros_like(ecg), ecg)
    return ecg.to(torch.float32)


def get_lab(lab_row, lab_means_path="./QA/labs_means.json"):
    """
    Processes laboratory data for a given row, handling missing values and
    ensuring consistent order.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two tensors, one for the lab percentiles
                                           and one for the missing indicators.
    """
    percentiles = []
    missing_indicators = []

    with open(lab_means_path, 'r') as f:
        labs_means = json.load(f)


    for col_p in sorted(labs_means.keys()): # sort to ensure order is consistent
        col = col_p.replace("_percentile", "")

        if pd.isna(lab_row.get(col)):
            # lab is missing
            percentiles.append(labs_means[col_p])
            missing_indicators.append(0)
        else:
            # lab is not missing
            percentiles.append(lab_row[col_p])
            missing_indicators.append(1)

    assert len(percentiles) == len(missing_indicators), \
        "Lengths of percentiles and missing indicators must match."
    assert len(percentiles) == 50, "There should be 50 labs."
    lab_percentiles_tensors = torch.tensor(percentiles, dtype=torch.float32)
    labs_missingness_tensors = torch.tensor(missing_indicators, dtype=torch.int64)
    labs_tensor = torch.cat([lab_percentiles_tensors, labs_missingness_tensors], dim=0)

    return labs_tensor


def process_and_save_tensors(args, df, split):
    """
    Processes the data from df and saves the resulting tensors to the specified
    directory.
    """
    save_dir = args.data_dir / split
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cxr_list = []
    ecg_list = []
    labs_percentiles_list = []
    labs_missingness_list = []
    hadm_id_list = []
    label_hadm_id_list = []
    label_list = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        cxr = get_cxr(args, row["cxr_path"], split)
        ecg = get_ecg(args, row["ecg_path"])
        (labs_percentiles, labs_missingness) = get_labs(args, row)

        cxr_list.append(cxr)
        ecg_list.append(ecg)
        labs_percentiles_list.append(labs_percentiles)
        labs_missingness_list.append(labs_missingness)
        hadm_id_list.append(row["hadm_id"])

        if split in ["val_retrieval", "test"]:
            label_hadm_id_list.append(row["label_hadm_id"])
            label_list.append(row["label"])

    cxr_tensor = torch.stack(cxr_list) # (n, 3, cxr_crop, cxr_crop)
    ecg_tensor = torch.stack(ecg_list) # (n, 1, 5000, 12)
    labs_percentiles_tensor = torch.stack(labs_percentiles_list) # (n, 50)
    labs_missingness_tensor = torch.stack(labs_missingness_list) # (n, 50)
    hadm_id_tensor = torch.tensor(hadm_id_list) # (n,)

    torch.save(cxr_tensor, save_dir / f"cxr_{split}.pt")
    torch.save(ecg_tensor, save_dir / f"ecg_{split}.pt")
    torch.save(labs_percentiles_tensor, save_dir / f"labs_percentiles_{split}.pt")
    torch.save(labs_missingness_tensor, save_dir / f"labs_missingness_{split}.pt")
    torch.save(hadm_id_tensor, save_dir / f"hadm_id_{split}.pt")

    if split in ["val_retrieval", "test"]:
        label_hadm_id_tensor = torch.tensor(label_hadm_id_list)
        torch.save(label_hadm_id_tensor, save_dir / f"label_hadm_id_{split}.pt")

        label_tensor = torch.tensor(label_list)
        torch.save(label_tensor, save_dir / f"label_{split}.pt")


