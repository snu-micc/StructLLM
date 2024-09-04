import json
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import math
from tqdm import tqdm



# bespoke ML
with open('./perovskite/result/prediction_PU_GCNN_TL_model.json', 'r') as jsonfile:
    prediction_PU_GCNN_TL_model = json.load(jsonfile)
with open('./perovskite/result/prediction_PU_gptembedding_TL_model.json', 'r') as jsonfile:
    prediction_PU_gptembedding_TL_model = json.load(jsonfile)


# GPT-4o-mini
with open('./perovskite/result/prediction_PerovStructGPT4om.json', 'r') as jsonfile:
    prediction_PerovStruct4om = json.load(jsonfile)
with open('./perovskite/result/prediction_PerovStructGPT4om-base.json', 'r') as jsonfile:
    prediction_PerovStruct4om_base = json.load(jsonfile)
with open('./perovskite/result/prediction_PerovStructGPT4om-FT-p8uall.json', 'r') as jsonfile:
    prediction_PerovStruct4om_FT_p8uall = json.load(jsonfile)
with open('./perovskite/result/prediction_PerovStructGPT4om-FT-TL-p8uall.json', 'r') as jsonfile:
    prediction_PerovStruct4om_FT_TL_p8uall = json.load(jsonfile)


prediction_Model = prediction_PerovStruct4om_FT_TL_p8uall


p_comp_list = []
p_score_list = []
p_pred_list = []
u_comp_list = []
u_score_list = []
u_pred_list = []


prediction_dict = {}
for pred in prediction_PerovStruct4om_FT_TL_p8uall:
    pred_list = [pred["Prediction1"],pred["Prediction2"],pred["Prediction3"]]
    for pp in pred_list:
        if pp not in prediction_dict.keys():
            prediction_dict[pp] = 1
        else:
            prediction_dict[pp] += 1
#print(prediction_dict)


if ('structperov' in prediction_Model[0]['Model']) or ('gpt' in prediction_Model[0]['Model']):
    for i, pred in enumerate(prediction_Model):
        p_check = "U"
        if pred["Prediction1"] in ['P', 'Y', ' P', 'Yes', 'p', '"P', 'Possible', 'Positive', '.P', 'y', ':P', '"P', '\tP']:
            p_score = math.exp(pred["Logprobs1"])
            p_check = "P"
        elif pred["Prediction2"] in ['P', 'Y', ' P', 'Yes', 'p', '"P', 'Possible', 'Positive', '.P', 'y', ':P', '"P', '\tP']:
            p_score = math.exp(pred["Logprobs2"])
        elif pred["Prediction3"] in ['P', 'Y', ' P', 'Yes', 'p', '"P', 'Possible', 'Positive', '.P', 'y', ':P', '"P', '\tP']:
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



# Get model metrics

p_score_list = np.array(p_score_list)
#n_score_list = np.array(n_score_list)
u_score_list = np.array(u_score_list)


alpha = 0.06322916 # PU-GTP-embedding-TL derived alpha
ids = np.argsort(-u_score_list)
Ntot = len(u_score_list)
Nn = int(Ntot*alpha)
cutoff = u_score_list[ids[Nn]]#0.741 #u_score_list[ids[Nn]]


FPR = []
TPR = []
PREC = []
#AUC = []
N = len(p_score_list)

FPR.append([])
TPR.append([])
PREC.append([])
#AUC.append([])
np.random.seed(0)
for _ in tqdm(range(1000)):
    np.random.shuffle(p_score_list)
    x1 = p_score_list[:N]
    #
    np.random.shuffle(u_score_list)
    x = u_score_list[:N]
    #
    tpr_pu = np.sum(x1>=cutoff)/len(x1)
    fpr_pu = np.sum(x>=cutoff)/len(x)
    prec_pu = np.sum(x1>=cutoff)/(np.sum(x1>=cutoff)+np.sum(x>=cutoff))
    #
    #p,q,ths = roc_curve([1]*len(x1)+[0]*len(x),x1.tolist()+x.tolist())
    #auc_pu = auc(p,q)
    #
    prec = alpha*tpr_pu/fpr_pu
    fpr = (fpr_pu-alpha*tpr_pu)/(1-alpha)
    #aa = (auc_pu-0.5*alpha)/(1-alpha)
    #
    FPR[-1].append(fpr)
    TPR[-1].append(tpr_pu)
    PREC[-1].append(prec)
    #AUC[-1].append(aa)

print(alpha,cutoff,np.mean(FPR[-1]),np.mean(TPR[-1]),np.mean(PREC[-1]))#,np.mean(AUC[-1]))
print(np.sum(u_score_list>=cutoff)/len(u_score_list))



#
