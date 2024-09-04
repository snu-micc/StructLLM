# This script can be run in anaconda base environment.
from openai import OpenAI
from tqdm import tqdm
from pymatgen.core import Composition
import re
import math
import jsonlines
import json


with open('./result/prediction_PU_GPTembedding_model.json', 'r') as jsonfile:
    prediction_PU_GPTembedding_model = json.load(jsonfile)

#with open('./result/prediction_StructSynthGPT-FT.json', 'r') as jsonfile:
#    prediction_StructSynthGPT_FT = json.load(jsonfile)

with open('./result/prediction_StructSynthGPT4om-FT.json', 'r') as jsonfile:
    prediction_StructSynthGPT_FT = json.load(jsonfile)

with open('./result/prediction_PU_CGCNN_model.json', 'r') as jsonfile:
    prediction_PU_CGCNN_model = json.load(jsonfile)

with open('./data/mp30s10000_dataset.json', 'r') as jsonfile:
    mp30s10000_dataset = json.load(jsonfile)

hold_out_pu = mp30s10000_dataset['hold-out-test_p'] + mp30s10000_dataset['hold-out-test_u']


# Combine the three result (PU-GPTembedding, PU-CGCNN, and Struct-GPT-FT)
result = []
for dd in tqdm(hold_out_pu):
    check1 = False
    for pp in prediction_PU_GPTembedding_model:
        if pp['mp-id'] == dd['material_id']:
            score_1 = pp['Prediction_Score']
            check1 = True
            break
    check2 = False
    for pp in prediction_PU_CGCNN_model:
        if pp['mp-id'] == dd['material_id']:
            score_2 = pp['Prediction_Score']
            check2 = True
            break
    check3 = False
    for pp in prediction_StructSynthGPT_FT:
        if pp['Prompt'].split('"U" (for unknown or unlikely): ')[-1] == dd['description']:
            p_check = "U"
            if pp["Prediction1"] in ['P', 'Y', ' P', 'Yes', 'p', '"P', 'Possible']:
                p_score = math.exp(pp["Logprobs1"])
                p_check = "P"
            elif pp["Prediction2"] in ['P', 'Y', ' P', 'Yes', 'p', '"P', 'Possible']:
                p_score = math.exp(pp["Logprobs2"])
            elif pp["Prediction3"] in ['P', 'Y', ' P', 'Yes', 'p', '"P', 'Possible']:
                p_score = math.exp(pp["Logprobs3"])
            else:
                p_score = 0
            score_3 = p_score
            check3 = True
            break
    #
    if check1 and check2 and check3:
        rr = {}
        rr['material_id'] = dd['material_id']
        rr['formula'] = dd['formula']
        rr['icsd'] = dd['icsd']
        rr['description'] = dd['description']
        rr['prediction_PU_GPTembedding'] = score_1
        rr['prediction_PU_CGCNN'] = score_2
        rr['prediction_StructSynthGPT_FT'] = score_3
        result.append(rr)


# Select equal prediction
ppp_prediction = []
nnn_prediction = []

pu_gptembedding_thr = 0.8132321113348007
pu_cgcnn_thr = 0.782503445148468
structgpt_thr = 0.777134804575976
for rr in result:
    if (rr['prediction_PU_GPTembedding']>=pu_gptembedding_thr) and (rr['prediction_PU_CGCNN']>=pu_cgcnn_thr) and (rr['prediction_StructSynthGPT_FT']>=structgpt_thr):
        ppp_prediction.append(rr)
    elif (rr['prediction_PU_GPTembedding']<pu_gptembedding_thr) and (rr['prediction_PU_CGCNN']<pu_cgcnn_thr) and (rr['prediction_StructSynthGPT_FT']<structgpt_thr):
        nnn_prediction.append(rr)

u_ppp_prediction = []
p_ppp_prediction = []
u_nnn_prediction = []
for rr in ppp_prediction:
    if rr['icsd'] == False:
        u_ppp_prediction.append(rr)
    elif rr['icsd'] == True:
        p_ppp_prediction.append(rr)
for rr in nnn_prediction:
    if rr['icsd'] == False:
        u_nnn_prediction.append(rr)


# 1. Preparing Your Batch File

# Load prompt text
with open("prompts.json") as prompt_file:
    prompt_dict = json.load(prompt_file)
    expla_sys_prompt = prompt_dict["expla_sys_prompt"]
    expla_user_prompt1 = prompt_dict["expla_user_prompt1"]
    expla_user_prompt2 = prompt_dict["expla_user_prompt2"]

system_prompt = expla_sys_prompt
user_prompt1 = expla_user_prompt1
user_prompt2 = expla_user_prompt2


batch_data = []
for i in range(len(ppp_prediction)):
    request = {}
    request["custom_id"] = "request-"+str(i+1)+"-ppp"
    request["method"] = "POST"
    request["url"] = "/v1/chat/completions"
    request["body"] = {
                       "model": "gpt-4o-2024-05-13",
                       #"temperature": 0,
                       #"logprobs": True,
                       #"top_logprobs": 3,
                       #"max_tokens": 2,
                       "messages": [
                           {"role":"system", "content": system_prompt},
                           {"role":"user", "content": user_prompt1 + ppp_prediction[i]['description']}
                       ],
                       }
    batch_data.append(request)
for i in range(len(nnn_prediction)):
    request = {}
    request["custom_id"] = "request-"+str(i+1)+"-nnn"
    request["method"] = "POST"
    request["url"] = "/v1/chat/completions"
    request["body"] = {
                       "model": "gpt-4o-2024-05-13",
                       #"temperature": 0,
                       #"logprobs": True,
                       #"top_logprobs": 3,
                       #"max_tokens": 2,
                       "messages": [
                           {"role":"system", "content": system_prompt},
                           {"role":"user", "content": user_prompt2 + nnn_prediction[i]['description']}
                       ],
                       }
    batch_data.append(request)

# Save training data as .jsonl
save_dir = "./data/explanation"
with open(save_dir+"/batch_explanation_4om_ppp_nnn_data.jsonl" , encoding= "utf-8",mode="w") as file:
    for i in batch_data:
        file.write(json.dumps(i) + "\n")


# 2. Uploading Your Batch Input File

# Load your OpenAI api_key
with open("config.json") as config_file:
    config = json.load(config_file)
    api_key = config["api_key"]

client = OpenAI(api_key=api_key)

batch_input_file = client.files.create(
  file=open(save_dir+"/batch_explanation_4om_ppp_nnn_data.jsonl", "rb"),
  purpose="batch"
)


# 3. Creating the Batch
batch_input_file_id = batch_input_file.id

client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
      "description": "nightly eval job"
    }
)










#
