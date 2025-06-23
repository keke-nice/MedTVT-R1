from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score as meteor_scorer
from nltk.tokenize import wordpunct_tokenize
import json
from bert_score import score
from tqdm.auto import tqdm
import nltk
import re
from pycocoevalcap.cider.cider import Cider

nltk.download('wordnet')

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

predicted_data = json.load(open("./results/infer_GRPO2500_retrain_llama3.2_1B_8331_fusion.json", "r"))
GT_data = json.load(open("/remote-home/hao.lu/Code/Yuting/ECG/ECG_pretrain/Dataset/symile-mimic/QA/test_dig_qa_dataset_gpt_7_category_add_spec_token.json", "r"))

label_mapping = {
    "E785": "Hyperlipidemia, Pure Hypercholesterolemia",
    "I2510": "Atherosclerotic Heart Disease Without Angina Pectoris",
    "N179": "Acute Kidney Failure, Unspecified",
    "I10": "Essential (Primary) Hypertension",
    "I4891": "Atrial Fibrillation",
    "E039": "Hypothyroidism, Unspecified",
    "I5033": "Chronic Systolic Heart Failure",
    "J189": "Pneumonia, Unspecified Organism",
    "J449": "Chronic Obstructive Pulmonary Disease (COPD), Unspecified",
    "I214": "Non-ST-Elevation Myocardial Infarction (NSTEMI)",
    "E119": "Type 2 diabetes mellitus without complications",
    "A419": "Sepsis, unspecified organism",
    "Other": "The conditions such as Hyperlipidemia, Atherosclerotic Heart Disease, Acute Kidney Failure, Essential Hypertension, Atrial Fibrillation, Hypothyroidism, Chronic Systolic Heart Failure, Pneumonia, COPD, NSTEMI, Type 2 Diabetes, and Sepsis were not identified. Please consider other diagnoses."
}

icd_mapping = {
    "Coronary Artery Disease": ["I250", "I251", "I252", "I253", "I254", "I255", "I256", "I257", "I258", "I259", "I2510"],
    "Acute Renal Failure": ["N170", "N171", "N172", "N178", "N179"],
    "Hypertension": ["I10", "I110", "I119", "I120", "I129", "I130", "I131", "I132", "I139", "I150", "I151", "I152", "I158", "I159"],
    "Atrial Fibrillation": ["I480", "I481", "I482", "I483", "I484", "I489", "I4891", "I4892", "I4890"],
    "Pneumonia": ["J180", "J181", "J182", "J188", "J189"],
    "Diabetes Mellitus": ["E100", "E101", "E102", "E103", "E104", "E105", "E106", "E107", "E108", "E109", "E110", "E111", "E112", "E113", "E114", "E115", "E116", "E117", "E118", "E119", "E130", "E131", "E132", "E133", "E134", "E135", "E136", "E137", "E138", "E139", "E140", "E141", "E142", "E143", "E144", "E145", "E146", "E147", "E148", "E149", "E190", "E191", "E192", "E193", "E194", "E195", "E196", "E197", "E198", "E199", "E1165", "E1140", "E11319", "E1129"],
    "Sepsis": ["A400", "A401", "A402", "A403", "A408", "A409", "A410", "A411", "A412", "A413", "A414", "A415", "A418", "A419", "R6520", "R6521"]
}

from sklearn.metrics import roc_auc_score
from collections import defaultdict

def evaluate_and_calculate_metrics(model_name, completions, ground_truths):
    rouge_score, bleu_score, bleu4_score, meteor_score, cider_score = 0, 0, 0, 0, 0

    total_precision = 0
    total_recall = 0 
    total_f1 = 0
    total_accuracy = 0
    num_samples = 0
    valid_labels = set(icd_mapping.keys())
    print("valid_labels:", valid_labels)
    gts = {i: [ground_truth] for i, ground_truth in enumerate(ground_truths)}
    res = {i: [completion] for i, completion in enumerate(completions)}

    category_stats = {label: {'TP': 0, 'FP': 0, 'FN': 0} for label in valid_labels}

    auc_data = defaultdict(list)  
    overall_true_labels = []  
    overall_pred_probs = []  

    pattern = r"<answer>(.*?)</answer>"
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    for ground_truth, completion in tqdm(zip(ground_truths, completions), total=len(ground_truths)):
        rouge_score += scorer.score(ground_truth, completion)['rougeL'].recall
        cand_split = wordpunct_tokenize(completion)
        ref_split = wordpunct_tokenize(ground_truth)
        bleu_score += sentence_bleu([ref_split], cand_split)
        meteor_score += meteor_scorer([ref_split], cand_split)

        completion_match = re.search(pattern, completion, re.DOTALL)
        ground_truth_match = re.search(pattern, ground_truth, re.DOTALL)

        if completion_match and ground_truth_match:
            completion_answer = completion_match.group(1).strip()
            ground_truth_answer = ground_truth_match.group(1).strip()
            
            completion_labels = set(completion_answer.split(";"))
            print('completion_labels:', completion_labels)
            ground_truth_labels = set(ground_truth_answer.split(";"))
            print('ground_truth_labels:', ground_truth_labels)
            
            completion_labels = {label.strip() for label in completion_labels if label.strip()}
            ground_truth_labels = {label.strip() for label in ground_truth_labels if label.strip()}
            unmapped_labels = {label for label in completion_labels if label not in valid_labels}
            if unmapped_labels:
                print(f"Unmapped labels in completion: {unmapped_labels}")
            
            true_positives = completion_labels.intersection(ground_truth_labels)
            precision = len(true_positives) / len(completion_labels) if len(completion_labels) > 0 else 0
            recall = len(true_positives) / len(ground_truth_labels) if len(ground_truth_labels) > 0 else 0
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            num_samples += 1

            for label in valid_labels:
                if label in true_positives:
                    category_stats[label]['TP'] += 1
                if label in completion_labels and label not in ground_truth_labels:
                    category_stats[label]['FP'] += 1
                if label in ground_truth_labels and label not in completion_labels:
                    category_stats[label]['FN'] += 1

                auc_data[label].append((1 if label in ground_truth_labels else 0, 
                                        1 if label in completion_labels else 0))
                overall_true_labels.append(1 if label in ground_truth_labels else 0)
                overall_pred_probs.append(1 if label in completion_labels else 0)

    rouge_score /= len(completions)
    bleu_score /= len(completions)
    bleu4_score /= len(completions)
    meteor_score /= len(completions)

    P, R, F1 = score(completions, ground_truths, lang="en", verbose=True)
    bert_score = R.mean().item()

    average_precision = total_precision / num_samples
    average_recall = total_recall / num_samples
    average_f1 = total_f1 / num_samples
  

    category_auc = {}
    for label, data in auc_data.items():
        if len(data) > 0:
            true_labels, pred_probs = zip(*data)
            try:
                category_auc[label] = roc_auc_score(true_labels, pred_probs)
            except ValueError:
                category_auc[label] = None  # 如果数据不足以计算AUC，返回None

    # 计算总体AUC
    try:
        overall_auc = roc_auc_score(overall_true_labels, overall_pred_probs)
    except ValueError:
        overall_auc = None  


    # 输出总体结果
    print(f"Model: {model_name}")
    print(f"BLEU Score: {bleu_score:.4f}")
    print(f"METEOR Score: {meteor_score:.4f}")
    print(f"ROUGE Score: {rouge_score:.4f}")
    print(f"BERT Score: {bert_score:.4f}")
    print(f"Average Precision: {average_precision:.4f}")
    print(f"Average Recall: {average_recall:.4f}")
    print(f"Average F1 Score: {average_f1:.4f}")
    print(f"Overall AUC: {overall_auc:.4f}" if overall_auc is not None else "Overall AUC: N/A")

GTs = []
predicts = []
for i in range(len(GT_data)):
    GTs.append(GT_data[i]["messages"][1]["content"])
    predicts.append(predicted_data[i]["messages"][1]["content"])

evaluate_and_calculate_metrics('Cardio_llama', predicts, GTs)



