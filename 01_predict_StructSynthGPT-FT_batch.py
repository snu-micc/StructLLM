# This script can be run in anaconda base environment.
from openai import OpenAI
from tqdm import tqdm
from pymatgen.core import Composition
import re
import jsonlines
import json


# Road hold-out-test set
test_set = []
with jsonlines.open("./data/finetuning/hold_out_pu_struct_data.jsonl") as f:
    for line in f.iter():
        test_set.append(line)


user_prompt_list = []
answer_list = []
for dd in test_set:
    user_prompt_list.append(dd['messages'][0]['content'])
    answer_list.append(dd['messages'][1]['content'])

# Load prompt text
with open("prompts.json") as prompt_file:
    prompt_dict = json.load(prompt_file)
    synth_sys_prompt = prompt_dict["synth_sys_prompt"]
    synth_user_prompt = prompt_dict["synth_user_prompt"]

# 1. Preparing Your Batch File
batch_data = []
for i in range(len(user_prompt_list)):
    request = {}
    request["custom_id"] = "request-"+str(i+1)
    request["method"] = "POST"
    request["url"] = "/v1/chat/completions"
    request["body"] = {#"model": "ft:gpt-3.5-turbo-0125:micc:stoisynthgpt:9KISV3XX",
                       #"model": "ft:gpt-3.5-turbo-0125:micc:structsynth:9F0E7WaE", # ft model by train/val_pu_struct (epoch 1, batch 32, lr 2) - 240417ver
                       #"model": "ft:gpt-3.5-turbo-0125:micc:structgpt5:9dqKRDHk", # ft model by train/val_pu_struct_5 (epoch 1, batch 32, lr 2) - 240625ver
                       #"model": "ft:gpt-3.5-turbo-0125:micc:structgpt10:9dqZaBid", # ft model by train/val_pu_struct_10 (epoch 1, batch 32, lr 2) - 240625ver
                       #"model": "ft:gpt-3.5-turbo-0125:micc:structgpt20:9dushatj", # ft model by train/val_pu_struct_20 (epoch 1, batch 32, lr 2) - 240625ver
                       #"model": "ft:gpt-3.5-turbo-0125:micc:structgpt50:9dwfS3BZ", # ft model by train/val_pu_struct_50 (epoch 1, batch 32, lr 2) - 240625ver
                       #"model": "ft:gpt-3.5-turbo-0125:micc:structgpt100:9dtetYhd", # ft model by train/val_pu_struct (epoch 1, batch 32, lr 2) - 240625ver
                       "model": "ft:gpt-4o-mini-2024-07-18:micc:structsynth:9r2PkOol",
                       "temperature": 0,
                       "logprobs": True,
                       "top_logprobs": 3,
                       "max_tokens": 2,
                       "messages": [
                           {"role":"system", "content": synth_sys_prompt},
                           {"role":"user", "content": synth_user_prompt+ user_prompt_list[i].split("(for unknown or unlikely):")[1]}
                       ],
                       }
    batch_data.append(request)

# Save training data as .jsonl
save_dir = "./batch_request"
with open(save_dir+"/batch_data_struct_4omini.jsonl" , encoding= "utf-8",mode="w") as file:
    for i in batch_data:
        file.write(json.dumps(i) + "\n")


# 2. Uploading Your Batch Input File

# Load your OpenAI api_key
with open("config.json") as config_file:
    config = json.load(config_file)
    api_key = config["api_key"]

client = OpenAI(api_key=api_key)

batch_input_file = client.files.create(
  file=open(save_dir+"/batch_data_struct_4omini.jsonl", "rb"),
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


# 4. Checking the Status of a Batch
#client.batches.retrieve('batch_Ww7PvtVH2vjMqX1d5IxhB64d')


# 5. Retrieving the Results
#content = client.files.content("file-4NDdfUAoMuhhG4wreddPqSFE")

"""
Note that the output line order may not match the input line order.
Instead of relying on order to process your results,
use the custom_id field which will be present in each line of your output file
and allow you to map requests in your input to results in your output.
"""


# 6. Download the batch_result (e.g. batch_Ww7PvtVH2vjMqX1d5IxhB64d_output.jsonl)
# 7. using "batchresult2resultformat.py" to convert batch result to our result formmat.


#
