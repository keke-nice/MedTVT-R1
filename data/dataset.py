from .instruction_datasets import *
from torch.utils.data import Dataset, ConcatDataset


class FinetuneDataset(Dataset):
    def __init__(self, max_words=30, tokenizer=None, stage=1, llama_type='llama3', add_special_token=True):
        dataset_list = []

        if stage == 1:
            # Encoder Datasets
            if add_special_token:
                ecgqa = ECGQADataset(json_path='./QA/train_ecg_qa_dataset_gpt_add_spec_token.json',
                                    root_path='./Dataset/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0', tokenizer=tokenizer, max_words=max_words, llama_type=llama_type, add_special_token=add_special_token)

                cxrqa = CXRQADataset(json_path='./QA/train_cxr_qa_dataset_gpt_add_spec_token.json',
                                    root_path='./Dataset/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.1.0',
                                    tokenizer=tokenizer, max_words=max_words, llama_type=llama_type,
                                    add_special_token=add_special_token)
                labqa = LABQADataset(json_path='./QA/train_lab_qa_dataset_gpt_add_spec_token.json',
                                    root_path='./Dataset/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0',
                                    tokenizer=tokenizer, max_words=max_words, llama_type=llama_type,
                                    add_special_token=add_special_token)
            else:
                ecgqa = ECGQADataset(json_path='./QA/train_ecg_qa_dataset_gpt.json',
                                    root_path='./Dataset/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0', tokenizer=tokenizer, max_words=max_words, llama_type=llama_type, add_special_token=add_special_token)

                cxrqa = CXRQADataset(json_path='./QA/train_cxr_qa_dataset_gpt.json',
                                    root_path='./Dataset/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.1.0',
                                    tokenizer=tokenizer, max_words=max_words, llama_type=llama_type,
                                    add_special_token=add_special_token)
                labqa = LABQADataset(json_path='./QA/train_lab_qa_dataset_gpt.json',
                                    root_path='./Dataset/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0',
                                    tokenizer=tokenizer, max_words=max_words, llama_type=llama_type,
                                    add_special_token=add_special_token)

            dataset_list.append(ecgqa)
            dataset_list.append(cxrqa)
            dataset_list.append(labqa)

        if stage == 3:
            # QA Dataset
            if add_special_token:
                allqa = All3QADataset(json_path='./QA/train_dig_qa_dataset_gpt_7_category_refine_add_spec_token.json',
                                      ecg_root_path='./Dataset/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0',
                                      cxr_root_path='./Dataset/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.1.0',
                                      tokenizer=tokenizer, max_words=max_words, llama_type=llama_type,
                                      add_special_token=add_special_token)
            else:

                allqa = All3QADataset(json_path='./QA/train_dig_qa_dataset_gpt_7_category_refine.json',
                                      ecg_root_path='./Dataset/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0',
                                      cxr_root_path='./Dataset/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.1.0',
                                      tokenizer=tokenizer, max_words=max_words, llama_type=llama_type,
                                      add_special_token=add_special_token)

            dataset_list.append(allqa)
        self.datasets = ConcatDataset(dataset_list)


    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        return self.datasets[index]