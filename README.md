# StructLLM

This repository contains the data and code for "**Explainable Structure-based Synthesizability of Inorganic Materials using Large Language Models**"

![image](https://github.com/user-attachments/assets/43ac3839-4e65-4f69-8ee6-6de0ecf23287)

# Developer
[Seongmin Kim](https://scholar.google.com/citations?user=HXcbuWQAAAAJ&hl=en&oi=ao),  [Joshua Schrier](https://scholar.google.com/citations?user=zJC_7roAAAAJ&hl=en),  and  [Yousung Jung](https://scholar.google.com/citations?user=y8D-JCAAAAAJ&hl=en&oi=ao)

# Organization
First, download the related data from zenodo (link:   TBD   ) and unzip.

**data** folder : general inorganic material data

**perovskite** folder : perovskite material data

**result** folder : model results for general synthesizability prediction and explanation

**batch_request** folder : batch data and batch result for OpenAI batch api request

Follow the codes in the order indicated by the numbered indexing. (From `00_Data.py` to `07a_batchresult2resultformat_explanation.py`)


# Instructions

Python code (`.py`) uses python 3.8.13 and requires libraries;

Numpy (version == 1.22.3), PyTorch (version == 1.11.0), Pymatgen (version == 2022.9.21), and OpenAI (version == 1.30.1).

For generating structural text description, Robocrystallographer (version == 0.2.8) was used.

- `00_Data.py` : Data preprocessing of General inorganic material for training and finetuning.
- `01_predict_StructSynthGPT-FT_batch.py` : Using fine-tuned StructLLM, predict synthesizability of hold-out-test dataset by OpenAI batch api request.
- `01a_batchresult2resultformat.py` : Convert the batch result ('01_predict_StructSynthGPT-FT_batch.py') to data analysis format.
- `02_get_metrics.py` : See the result of models for general synthesizability prediction
- `03_explain_Synthesizability_GPT4o_4om.py` : Using GPT-4o, explain the reasons of general synthesizability by OpenAI batch api request.
- `03a_batchresult2resultformat_explanation_4om.py` : Convert the batch result ('03_explain_Synthesizability_GPT4o_4om.py') to data analysis format.
- `04_get_explanation_GPT_embedding_4om.py` : Using gpt text-embedding-large model, get embedding vectors of textual explanations ('03a_batchresult2resultformat_explanation_4om.py').
- `04a_make_GPTembedding_json_dictionary_4om.py` : Convert the batch result ('04_get_explanation_GPT_embedding_4om.py') to data analysis format.
- `04b_explanation_analysis_4om.py` : See the explanation result of models for general synthesizability prediction
- `05_predict_PerovStructGPT-FT-TL_batch.py` : Using fine-tuned perovskite StructLLM, predict synthesizability of hold-out-test perovskite dataset by OpenAI batch api request.
- `06_get_metrics_perovskite.py` : See the result of models for perovskite synthesizability prediction



