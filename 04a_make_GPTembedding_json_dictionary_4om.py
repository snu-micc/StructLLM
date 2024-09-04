# This script can be run in anaconda base environment.
from tqdm import tqdm
from pymatgen.core import Composition
import numpy as np
import re
import jsonlines
import json
import pickle


with open('./result/explanation_ppp_nnn_formatted_4om.json', "r") as json_file:
    total_data = json.load(json_file)

# Type your batch result files identifier of (batch_data_for_explanation_embedding1 and batch_data_for_explanation_embedding2)
batch1_result = []
with jsonlines.open("./batch_request/batch_data_for_explanation_embedding1_4om_output.jsonl") as f:
    for line in f.iter():
        batch1_result.append(line)
batch2_result = []
with jsonlines.open("./batch_request/batch_data_for_explanation_embedding2_4om_output.jsonl") as f:
    for line in f.iter():
        batch2_result.append(line)

batch_result = batch1_result + batch2_result

# check the batch calculated response sequence
idx_sequence = []
for i, rr in enumerate(batch_result):
    idx = int(rr['custom_id'].split('-')[-1])-1
    idx_sequence.append(idx)

if idx_sequence == np.arange(len(batch_result)).tolist():
    print("Correct idx order")
else:
    aaaaaaaaaaaaaa


embedded_data = []
b_idx = 0
for idx in range(len(total_data)):
    e_data = {}
    e_data['Label'] = total_data[idx]['Label']
    e_data['Prompt'] = total_data[idx]['Prompt']
    e_data['Explanation'] = total_data[idx]['Explanation']
    e_data['formatted_Explanation'] = total_data[idx]['formatted_Explanation']
    e_data['GPTembedding'] = []
    for j in range(len(total_data[idx]['formatted_Explanation'])):
        e_data['GPTembedding'].append( batch_result[b_idx]['response']['body']['data'][0]['embedding'] )
        b_idx += 1
    embedded_data.append(e_data)


# save
with open('./result/explanation_ppp_nnn_formatted_with_embedding_4om.pickle', 'wb') as f:
    pickle.dump(embedded_data, f, pickle.HIGHEST_PROTOCOL)

# load
#with open('./result/explanation_ppp_nnn_formatted_with_embedding.pickle', 'rb') as f:
#    embedded_data1 = pickle.load(f)

#
