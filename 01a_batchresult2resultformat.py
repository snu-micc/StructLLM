import jsonlines
import json



test_set = []
with jsonlines.open("./data/finetuning/hold_out_pu_struct_data.jsonl") as f:
    for line in f.iter():
        test_set.append(line)

user_prompt_list = []
answer_list = []
for dd in test_set:
    user_prompt_list.append(dd['messages'][0]['content'])
    answer_list.append(dd['messages'][1]['content'])

batch_test = []
with jsonlines.open("./batch_request/batch_[  TYPE YOUR BATCH RESULT IDENTIFIER  ]_output.jsonl") as f:
    for line in f.iter():
        batch_test.append(line)


prediction_result = []
for i in range(len(batch_test)):
    d_idx = int(batch_test[i]['custom_id'].split('request-')[-1])-1
    response = batch_test[i]['response']

    pred_result = {}
    pred_result['Model'] = response['body']['model']
    pred_result['Prompt'] = user_prompt_list[d_idx]
    pred_result['Answer'] = answer_list[d_idx]
    pred_result['Prediction1'] = response['body']['choices'][0]['logprobs']['content'][0]['top_logprobs'][0]['token']
    pred_result['Prediction2'] = response['body']['choices'][0]['logprobs']['content'][0]['top_logprobs'][1]['token']
    pred_result['Prediction3'] = response['body']['choices'][0]['logprobs']['content'][0]['top_logprobs'][2]['token']
    pred_result['Logprobs1'] = response['body']['choices'][0]['logprobs']['content'][0]['top_logprobs'][0]['logprob']
    pred_result['Logprobs2'] = response['body']['choices'][0]['logprobs']['content'][0]['top_logprobs'][1]['logprob']
    pred_result['Logprobs3'] = response['body']['choices'][0]['logprobs']['content'][0]['top_logprobs'][2]['logprob']
    prediction_result.append(pred_result)


with open('./result/prediction_StructSynthGPT-FT.json', 'w') as f:
    json.dump(prediction_result, f, indent=4)
