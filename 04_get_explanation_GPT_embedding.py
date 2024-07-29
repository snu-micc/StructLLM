# This script can be run in anaconda base environment.
import re
import json
import matplotlib.pyplot as plt
from openai import OpenAI
import jsonlines


with open('./result/explanation_ppp.json', 'r') as jsonfile:
    explanation_ppp = json.load(jsonfile)

with open('./result/explanation_nnn.json', 'r') as jsonfile:
    explanation_nnn = json.load(jsonfile)


total_data = []

reason_cluster = []
reason_count_dict ={}
for dd in explanation_ppp:
    explanation = re.sub('\n', ' ', dd['Explanation'])
    reasons = explanation.split('### ')
    while '' in reasons:
        reasons.remove('')
    while ' ' in reasons:
        reasons.remove(' ')
    while '  ' in reasons:
        reasons.remove('  ')
    #
    r_len = len(reasons)
    if r_len not in reason_count_dict.keys():
        reason_count_dict[r_len] = 1
    else:
        reason_count_dict[r_len] += 1
    #
    formatted_reasons = []
    for rr in reasons:
        r_idx = rr.split('**')[0]
        r_description = rr[len(r_idx):]
        formatted_reasons.append(r_description)
    #
    reason_cluster += formatted_reasons
    data = {}
    data['Label'] = 'ppp'
    data['Prompt'] = dd['Prompt']
    data['Explanation'] = dd['Explanation']
    data['formatted_Explanation'] = formatted_reasons
    total_data.append(data)

for dd in explanation_nnn:
    explanation = re.sub('\n', ' ', dd['Explanation'])
    reasons = explanation.split('### ')
    while '' in reasons:
        reasons.remove('')
    while ' ' in reasons:
        reasons.remove(' ')
    while '  ' in reasons:
        reasons.remove('  ')
    #
    r_len = len(reasons)
    if r_len not in reason_count_dict.keys():
        reason_count_dict[r_len] = 1
    else:
        reason_count_dict[r_len] += 1
    #
    formatted_reasons = []
    for rr in reasons:
        r_idx = rr.split('**')[0]
        r_description = rr[len(r_idx):]
        formatted_reasons.append(r_description)
    #
    reason_cluster += formatted_reasons
    data = {}
    data['Label'] = 'nnn'
    data['Prompt'] = dd['Prompt']
    data['Explanation'] = dd['Explanation']
    data['formatted_Explanation'] = formatted_reasons
    total_data.append(data)


with open('./result/explanation_ppp_nnn_formatted.json', 'w') as outfile:
    json.dump(total_data, outfile, indent=4)


### Get GPTembedding vector
# 1. Preparing Your Batch File
batch_data = []
i = 0
for dd in total_data:
    for expl in dd['formatted_Explanation']:
        request = {}
        request["custom_id"] = 'data-'+str(i+1)
        request["method"] = "POST"
        #request["url"] = "/v1/chat/completions"
        request["url"] = "/v1/embeddings"
        request["body"] = {"model": "text-embedding-3-large",
                           "input": expl
                           }
        batch_data.append(request)
        i += 1

batch_data1 = batch_data[:40000]
batch_data2 = batch_data[40000:]

# Save batch data as .jsonl
save_dir = "./batch_request"
with open(save_dir+"/batch_data_for_explanation_embedding1.jsonl" , encoding= "utf-8",mode="w") as file:
    for i in batch_data1:
        file.write(json.dumps(i) + "\n")
with open(save_dir+"/batch_data_for_explanation_embedding2.jsonl" , encoding= "utf-8",mode="w") as file:
    for i in batch_data2:
        file.write(json.dumps(i) + "\n")


# 2. Uploading Your Batch Input File
client = OpenAI(api_key="[   TYPE YOUR OPENAI API KEY   ]")

batch_input_file1 = client.files.create(
  file=open(save_dir+"/batch_data_for_explanation_embedding1.jsonl", "rb"),
  purpose="batch"
)
batch_input_file2 = client.files.create(
  file=open(save_dir+"/batch_data_for_explanation_embedding2.jsonl", "rb"),
  purpose="batch"
)


# 3. Creating the Batch
batch_input_file_id1 = batch_input_file1.id
batch_input_file_id2 = batch_input_file2.id

client.batches.create(
    input_file_id=batch_input_file_id1,
    #endpoint="/v1/chat/completions",
    endpoint="/v1/embeddings",
    completion_window="24h",
    metadata={
      "description": "nightly eval job"
    }
)
client.batches.create(
    input_file_id=batch_input_file_id2,
    #endpoint="/v1/chat/completions",
    endpoint="/v1/embeddings",
    completion_window="24h",
    metadata={
      "description": "nightly eval job"
    }
)








#
