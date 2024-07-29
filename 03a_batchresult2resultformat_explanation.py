import jsonlines
import json


# For ppp data (7054)
test_set = []
with jsonlines.open("./data/explanation/batch_explanation_ppp_data.jsonl") as f:
    for line in f.iter():
        test_set.append(line)

user_prompt_list = []
for dd in test_set:
    user_prompt_list.append(dd['body']['messages'])

batch_test = []
with jsonlines.open("./batch_request/batch_explanation_ppp_data_output.jsonl") as f:
    for line in f.iter():
        batch_test.append(line)

explanation_result = []
for i in range(len(batch_test)):
    d_idx = int(batch_test[i]['custom_id'].split('request-')[-1])-1
    response = batch_test[i]['response']
    expla_result = {}
    expla_result['Model'] = response['body']['model']
    expla_result['Prompt'] = user_prompt_list[d_idx]
    expla_result['Explanation'] = response['body']['choices'][0]['message']['content']
    explanation_result.append(expla_result)

with open('./result/explanation_ppp.json', 'w') as f:
    json.dump(explanation_result, f, indent=4)


# For nnn data (8586)
test_set = []
with jsonlines.open("./data/explanation/batch_explanation_nnn_data.jsonl") as f:
    for line in f.iter():
        test_set.append(line)

user_prompt_list = []
for dd in test_set:
    user_prompt_list.append(dd['body']['messages'])

batch_test = []
with jsonlines.open("./batch_request/batch_explanation_nnn_data_output.jsonl") as f:
    for line in f.iter():
        batch_test.append(line)

explanation_result = []
for i in range(len(batch_test)):
    d_idx = int(batch_test[i]['custom_id'].split('request-')[-1])-1
    response = batch_test[i]['response']
    expla_result = {}
    expla_result['Model'] = response['body']['model']
    expla_result['Prompt'] = user_prompt_list[d_idx]
    expla_result['Explanation'] = response['body']['choices'][0]['message']['content']
    explanation_result.append(expla_result)

with open('./result/explanation_nnn.json', 'w') as f:
    json.dump(explanation_result, f, indent=4)
