# This script can be run in anaconda base environment.
import jsonlines
import json


# For ppp_nnn data (3886 / 11969)
test_set = []
with jsonlines.open("./data/explanation/batch_explanation_4om_ppp_nnn_data.jsonl") as f:
    for line in f.iter():
        test_set.append(line)

user_prompt_list = []
for dd in test_set:
    user_prompt_list.append(dd['body']['messages'])

batch_test = []
with jsonlines.open("./batch_request/batch_explanation_4om_ppp_nnn_data_output.jsonl") as f:
    for line in f.iter():
        batch_test.append(line)

p_data_count = 0
for i in range(len(batch_test)):
    d_idx = int(batch_test[i]['custom_id'].split('request-')[-1].split('-')[0])-1
    if 'ppp' in batch_test[i]['custom_id'].split('request-')[-1]:
        p_data_count += 1

explanation_result = []
d_idx_list = []
for i in range(len(batch_test)):
    d_idx = int(batch_test[i]['custom_id'].split('request-')[-1].split('-')[0])-1
    if 'ppp' in batch_test[i]['custom_id'].split('request-')[-1]:
        pass
    elif 'nnn' in batch_test[i]['custom_id'].split('request-')[-1]:
        d_idx += p_data_count
    d_idx_list.append(d_idx)
    response = batch_test[i]['response']
    expla_result = {}
    expla_result['Model'] = response['body']['model']
    expla_result['Prompt'] = user_prompt_list[d_idx]
    expla_result['Explanation'] = response['body']['choices'][0]['message']['content']
    explanation_result.append(expla_result)

with open('./result/explanation_ppp_nnn_4om.json', 'w') as f:
    json.dump(explanation_result, f, indent=4)


#
