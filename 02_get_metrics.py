import json
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import glob


with open('./result/prediction_StructSynthGPT-FT.json', 'r') as jsonfile:
    prediction_StructSynthGPT_FT = json.load(jsonfile)

with open('./result/prediction_PU_CGCNN_model.json', 'r') as jsonfile:
    prediction_PU_CGCNN_model = json.load(jsonfile)

with open('./result/prediction_PU_GPTembedding_model.json', 'r') as jsonfile:
    prediction_PU_GPTembedding_model = json.load(jsonfile)


prediction_Model = prediction_StructSynthGPT_FT


p_comp_list = []
p_score_list = []
p_pred_list = []
u_comp_list = []
u_score_list = []
u_pred_list = []


prediction_dict = {}
for pred in prediction_StructSynthGPT_FT:
    pred_list = [pred["Prediction1"],pred["Prediction2"],pred["Prediction3"]]
    for pp in pred_list:
        if pp not in prediction_dict.keys():
            prediction_dict[pp] = 1
        else:
            prediction_dict[pp] += 1
#print(prediction_dict)

error_count = 0
if 'gpt' in prediction_Model[0]['Model']:
    for i, pred in enumerate(prediction_Model):
        try:
            p_check = "U"
            if pred["Prediction1"] in ['P', 'Y', ' P', 'Yes', 'p', '"P', 'Possible']:
                p_score = math.exp(pred["Logprobs1"])
                p_check = "P"
            elif pred["Prediction2"] in ['P', 'Y', ' P', 'Yes', 'p', '"P', 'Possible']:
                p_score = math.exp(pred["Logprobs2"])
            elif pred["Prediction3"] in ['P', 'Y', ' P', 'Yes', 'p', '"P', 'Possible']:
                p_score = math.exp(pred["Logprobs3"])
            else:
                p_score = 0
            #
            if pred['Answer'] == "P":
                p_comp_list.append(pred["Prompt"])
                p_score_list.append(p_score)
                p_pred_list.append(p_check=="P")
            elif pred['Answer'] == "U":
                u_comp_list.append(pred["Prompt"])
                u_score_list.append(p_score)
                u_pred_list.append(p_check=="P")
            else:
                print("Invalid answer! (0 or 1)")
                aaaaaaaaa
        except:
            error_count += 1
else:  # PU-CGCNN model case
    for i, pred in enumerate(prediction_Model):
        if pred['Answer'] == "P":
            p_comp_list.append(pred["mp-id"])
            p_score_list.append(pred["Prediction_Score"])
            p_pred_list.append(pred["Prediction_Score"]>=0.5)
        elif pred['Answer'] == "U":
            u_comp_list.append(pred["mp-id"])
            u_score_list.append(pred["Prediction_Score"])
            u_pred_list.append(pred["Prediction_Score"]>=0.5)
        else:
            print("Invalid answer! (0 or 1)")
            aaaaaaaaa


print(len(p_comp_list), np.sum((np.array(p_score_list)>=0.5) == np.array(p_pred_list)))
print(len(u_comp_list), np.sum((np.array(u_score_list)>=0.5) == np.array(u_pred_list)))


# Draw score distribution

plt.figure(figsize=(8,5))
plt.hist(p_score_list, histtype='bar', bins=40, edgecolor='darkcyan', color='lightseagreen')
plt.title("Positive composition", fontsize=25)
plt.xlabel('Score', fontsize=25)
plt.ylabel('Counts', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim([0,1])
plt.show()


plt.figure(figsize=(8,5))
plt.hist(u_score_list, histtype='bar', bins=40, edgecolor='dimgray', color='silver')
plt.title("Unlabeled composition", fontsize=25)
plt.xlabel('Score', fontsize=25)
plt.ylabel('Counts', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim([0,1])
plt.show()






#
