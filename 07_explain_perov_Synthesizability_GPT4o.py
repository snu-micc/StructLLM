# This script can be run in anaconda base environment.
from openai import OpenAI
from tqdm import tqdm
from pymatgen.core import Composition
import re
import math
import jsonlines
import json


with open('./perovskite/result/prediction_PU_gptembedding_TL_model.json', 'r') as jsonfile:
    prediction_PU_gptembedding_TL_model = json.load(jsonfile)

with open('./perovskite/result/prediction_PerovStructGPT-FT-TL-p8uall.json', 'r') as jsonfile:
    prediction_PerovStruct_FT_TL_p8uall = json.load(jsonfile)

with open('./perovskite/result/prediction_PU_GCNN_TL_model.json', 'r') as jsonfile:
    prediction_PU_GCNN_TL_model = json.load(jsonfile)

with open('./perovskite/perov_data/mp30oqmd30s10000_perovskite_dataset.json', 'r') as f:
    mp30oqmd30s10000_perovskite_dataset = json.load(f)

hold_out_pu = mp30oqmd30s10000_perovskite_dataset['hold-out-test_p'] + mp30oqmd30s10000_perovskite_dataset['hold-out-test_u']


# Combine the three result (PU-GPTembedding, PU-CGCNN, and Struct-GPT-FT)
result = []
for dd in tqdm(hold_out_pu):
    check1 = False
    for pp in prediction_PU_gptembedding_TL_model:
        if pp['mp-id'] == dd['material_id'][0]:
            score_1 = pp['Prediction_Score']
            check1 = True
            break
    check2 = False
    for pp in prediction_PU_GCNN_TL_model:
        if pp['mp-id'] == dd['material_id'][0]:
            score_2 = pp['Prediction_Score']
            check2 = True
            break
    check3 = False
    for pp in prediction_PerovStruct_FT_TL_p8uall:
        if pp['Prompt'].split('"U" (for unknown or unlikely): ')[-1] == dd['description']:
            p_check = "U"
            if pp["Prediction1"] in ['P', 'Y', ' P', 'Yes', 'p', '"P', 'Possible', 'Positive', '.P', 'y', ':P', '"P', '\tP']:
                p_score = math.exp(pp["Logprobs1"])
                p_check = "P"
            elif pp["Prediction2"] in ['P', 'Y', ' P', 'Yes', 'p', '"P', 'Possible', 'Positive', '.P', 'y', ':P', '"P', '\tP']:
                p_score = math.exp(pp["Logprobs2"])
            elif pp["Prediction3"] in ['P', 'Y', ' P', 'Yes', 'p', '"P', 'Possible', 'Positive', '.P', 'y', ':P', '"P', '\tP']:
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
    else:
        raise NotImplementedError("Invalid test item")


# Select equal prediction
ppp_prediction = []
nnn_prediction = []
for rr in result:
    if (rr['prediction_PU_GPTembedding']>=0.5) and (rr['prediction_PU_CGCNN']>=0.5) and (rr['prediction_StructSynthGPT_FT']>=0.5):
        ppp_prediction.append(rr)
    elif (rr['prediction_PU_GPTembedding']<0.5) and (rr['prediction_PU_CGCNN']<0.5) and (rr['prediction_StructSynthGPT_FT']<0.5):
        nnn_prediction.append(rr)


# 1. Preparing Your Batch File
system_prompt = "Return only output of the following format for each reason, and no other information: ### Reason 1. **[Keyword of reason]**\n [Detailed description] \n"
user_prompt1 = "Explain why an inorganic compound with the following structural information is synthesizable: "
user_prompt2 = "Explain why an inorganic compound with the following structural information is not synthesizable: "

batch_ppp_data = []
for i in range(len(ppp_prediction)):
    request = {}
    request["custom_id"] = "request-"+str(i+1)
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
    batch_ppp_data.append(request)

batch_nnn_data = []
for i in range(len(nnn_prediction)):
    request = {}
    request["custom_id"] = "request-"+str(i+1)
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
    batch_nnn_data.append(request)

with open('./perovskite/perov_data/explanation/ppp_prediction_for_hold-out-pu.json', 'w') as f:
    json.dump(ppp_prediction, f, indent=4)

with open('./perovskite/perov_data/explanation/nnn_prediction_for_hold-out-pu.json', 'w') as f:
    json.dump(nnn_prediction, f, indent=4)


# Save training data as .jsonl
save_dir = "./perovskite/perov_data/explanation"

with open(save_dir+"/batch_explanation_ppp_data.jsonl" , encoding= "utf-8",mode="w") as file:
    for i in batch_ppp_data:
        file.write(json.dumps(i) + "\n")

with open(save_dir+"/batch_explanation_nnn_data.jsonl" , encoding= "utf-8",mode="w") as file:
    for i in batch_nnn_data:
        file.write(json.dumps(i) + "\n")


# 2. Uploading Your Batch Input File
client = OpenAI(api_key="[   TYPE YOUR OPENAI API KEY   ]")


batch_input_file1 = client.files.create(
  file=open(save_dir+"/batch_explanation_ppp_data.jsonl", "rb"),
  purpose="batch"
)

batch_input_file2 = client.files.create(
  file=open(save_dir+"/batch_explanation_nnn_data.jsonl", "rb"),
  purpose="batch"
)


# 3. Creating the Batch
batch_input_file_id1 = batch_input_file1.id

client.batches.create(
    input_file_id=batch_input_file_id1,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
      "description": "nightly eval job"
    }
)

batch_input_file_id2 = batch_input_file2.id

client.batches.create(
    input_file_id=batch_input_file_id2,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
      "description": "nightly eval job"
    }
)










#
