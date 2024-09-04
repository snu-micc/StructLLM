# This script can be run in anaconda base environment.
import re
import json
import matplotlib.pyplot as plt
from openai import OpenAI
import jsonlines


with open('./result/explanation_ppp_nnn_4om.json', 'r') as jsonfile:
    explanation_ppp_nnn = json.load(jsonfile)


with open("prompts.json") as prompt_file:
    prompt_dict = json.load(prompt_file)
    expla_sys_prompt = prompt_dict["expla_sys_prompt"]
    expla_user_prompt1 = prompt_dict["expla_user_prompt1"]
    expla_user_prompt2 = prompt_dict["expla_user_prompt2"]

total_data = []

reason_cluster = []
reason_count_dict ={}
for dd in explanation_ppp_nnn:
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
    if expla_user_prompt1 in dd['Prompt'][1]['content']:
        pn_label = 'ppp'
    elif expla_user_prompt2 in dd['Prompt'][1]['content']:
        pn_label = 'nnn'
    else:
        aaaaaaaa
    data['Label'] = pn_label
    data['Prompt'] = dd['Prompt']
    data['Explanation'] = dd['Explanation']
    data['formatted_Explanation'] = formatted_reasons
    total_data.append(data)


with open('./result/explanation_ppp_nnn_formatted_4om.json', 'w') as outfile:
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
with open(save_dir+"/batch_data_for_explanation_embedding1_4om.jsonl" , encoding= "utf-8",mode="w") as file:
    for i in batch_data1:
        file.write(json.dumps(i) + "\n")
with open(save_dir+"/batch_data_for_explanation_embedding2_4om.jsonl" , encoding= "utf-8",mode="w") as file:
    for i in batch_data2:
        file.write(json.dumps(i) + "\n")


# 2. Uploading Your Batch Input File

# Load your OpenAI api_key
with open("config.json") as config_file:
    config = json.load(config_file)
    api_key = config["api_key"]

client = OpenAI(api_key=api_key)

batch_input_file1 = client.files.create(
  file=open(save_dir+"/batch_data_for_explanation_embedding1_4om.jsonl", "rb"),
  purpose="batch"
)
batch_input_file2 = client.files.create(
  file=open(save_dir+"/batch_data_for_explanation_embedding2_4om.jsonl", "rb"),
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
