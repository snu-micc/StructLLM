# This script can be run in anaconda base environment.
import jsonlines
import json


# For ppp data (480)
with open('./perovskite/perov_data/explanation/ppp_prediction_for_hold-out-pu.json', 'r') as jsonfile:
    ppp_prediction = json.load(jsonfile)

batch_test = []
with jsonlines.open("./perovskite/batch_request/batch_ppp_prediction_for_hold-out-pu_output.jsonl") as f:
    for line in f.iter():
        batch_test.append(line)

ppp_prediction_explanation = []
for i in range(len(batch_test)):
    d_idx = int(batch_test[i]['custom_id'].split('request-')[-1])-1
    response = batch_test[i]['response']
    expla = response['body']['choices'][0]['message']['content']
    ppp_prediction[d_idx]['Explanation'] = expla
    ppp_prediction_explanation.append(ppp_prediction[d_idx])

with open('./perovskite/result/explanation_ppp.json', 'w') as f:
    json.dump(ppp_prediction_explanation, f, indent=4)


# For nnn data (1897)
with open('./perovskite/perov_data/explanation/nnn_prediction_for_hold-out-pu.json', 'r') as jsonfile:
    nnn_prediction = json.load(jsonfile)

batch_test = []
with jsonlines.open("./perovskite/batch_request/batch_nnn_prediction_for_hold-out-pu_output.jsonl") as f:
    for line in f.iter():
        batch_test.append(line)

nnn_prediction_explanation = []
for i in range(len(batch_test)):
    d_idx = int(batch_test[i]['custom_id'].split('request-')[-1])-1
    response = batch_test[i]['response']
    expla = response['body']['choices'][0]['message']['content']
    nnn_prediction[d_idx]['Explanation'] = expla
    nnn_prediction_explanation.append(nnn_prediction[d_idx])

with open('./perovskite/result/explanation_nnn.json', 'w') as f:
    json.dump(nnn_prediction_explanation, f, indent=4)





#
