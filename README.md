# StructLLM

This repository contains the data and code for "**Explainable Synthesizability Prediction of Inorganic Crystal Polymorphs using Large Language Models**"

![image](https://github.com/snu-micc/StructLLM/blob/main/TOC.png)


# Developer
[Seongmin Kim](https://scholar.google.com/citations?user=HXcbuWQAAAAJ&hl=en&oi=ao),  [Joshua Schrier](https://scholar.google.com/citations?user=zJC_7roAAAAJ&hl=en),  and  [Yousung Jung](https://scholar.google.com/citations?user=y8D-JCAAAAAJ&hl=en&oi=ao)

# Organization
First, download the related data from zenodo (link: https://zenodo.org/records/14729225 ) and unzip.

**data** folder : general inorganic material data

**perovskite** folder : perovskite material data

**result** folder : model results for general synthesizability prediction and explanation

**batch_request** folder : batch data and batch result for OpenAI batch api request

Follow the codes in the order indicated by the numbered indexing. (From `00_Data.py` to `06_get_metrics_perovskite.py`)


# Instructions

Python code (`.py`) uses python 3.11.8 and requires libraries;

Numpy (version == 1.26.4), Pymatgen (version == 2024.3.1), and OpenAI (version == 1.30.1).

For generating structural text description, Robocrystallographer (version == 0.2.8) was used.

You can manually install or using `requirements.txt` to use following codes:

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


# Code guideline

Follow the below steps to use the code and to reproduce this method as it is;

- Get your OpenAI api key. (See https://platform.openai.com/)
- Download MP30_description dataset.
- Fix `config.json` file. ( Type your OpenAI api key & your downloaded mp30_description folder path )
- Run `00_Data.py` code. --> `./data/finetuning/train_pu_struct_data.jsonl`, `./data/finetuning/val_pu_struct_data.jsonl`, `./data/finetuning/hold_out_pu_struct_data.jsonl` will be generated.
- Fine-tuning GPT-4o-mini by using the generated `./data/finetuning/train_pu_struct_data.jsonl`, `./data/finetuning/val_pu_struct_data.jsonl` dataset. (See https://platform.openai.com/docs/guides/fine-tuning/)
- After finishing the fine-tuning, copy your fine-tuned model identifier and paste it into `01_predict_StructSynthGPT-FT_batch.py` requesting model argument.
- Run `01_predict_StructSynthGPT-FT_batch.py` code. --> Batch request will be calculated. After batch completed, you can download the output jsonl file.
- Copy your batch identifier and paste it into `01a_batchresult2resultformat.py` at batch output load part.
- Run `01a_batchresult2resultformat.py` to convert batch output to readable json file. --> The test result will be saved as `./result/prediction_StructSynthGPT4om.json`.

If you want to test another inorganic structures, make the same format to the test dataset (`./data/finetuning/hold_out_pu_struct_data.jsonl`) and run `01_predict_StructSynthGPT-FT_batch.py` code.
To obtain the structural text from cif file, use Robocrystallographer tool. (See https://hackingmaterials.lbl.gov/robocrystallographer/index.html#)

# Cite
A publication appears on the [Angewandte Chemie International Edition](https://onlinelibrary.wiley.com/doi/abs/10.1002/anie.202423950) as `doi:10.1002/anie.202423950`  

