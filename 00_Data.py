import glob
import json
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Composition


with open("config.json") as config_file:
    config = json.load(config_file)
    mp30_description_path = config["mp30_description_path"]

dir = mp30_description_path
mp_list = glob.glob(dir+"m*.json")

data = []
for mp_dir in tqdm(mp_list):
    with open(mp_dir, "r") as jsonfile:
        d = json.load(jsonfile)
        data.append(d)

print(d.keys()) # dict_keys(['material_id', 'formula', 'icsd', 'description'])


string_length = []
data_10000 = []
for d in data:
    string_length.append(len(d["description"]))
    if len(d["description"]) < 10000: # less than 10000 chracters
        data_10000.append(d)


plt.figure(figsize=(12,8))
plt.hist(string_length, bins=40, ec='k', color='gray')
plt.xlabel('String length', fontsize=40, labelpad=15)
plt.ylabel('Count', fontsize=40, labelpad=15)
plt.title('MP30 description length', size=35)
plt.xticks(size=30, rotation=30)
plt.yticks(size=30)
plt.xlim([-1000,max(string_length)])
#plt.ylim([0,1])
plt.show()

plt.figure(figsize=(12,8))
plt.hist(string_length, bins=130, ec='k', color='gray')
plt.xlabel('String length', fontsize=40, labelpad=15)
plt.ylabel('Count', fontsize=40, labelpad=15)
plt.title('MP30 description length', size=35)
plt.xticks(size=30, rotation=30)
plt.yticks(size=30)
plt.xlim([0,8579])
#plt.ylim([0,1])
plt.show()


comp_count_dict = {}
for dd in data_10000:
    comp = Composition(dd['formula']).reduced_composition
    if comp not in comp_count_dict:
        comp_count_dict[comp] = 1
    else:
        comp_count_dict[comp] += 1


count_ary = np.array(list(comp_count_dict.values()))
x_list = [1, 2, 3, 4, 5]
y_list = [sum(count_ary==1), sum(count_ary==2), sum(count_ary==3), sum(count_ary==4), sum(count_ary>=5)]

plt.figure(figsize=(8,8))
plt.bar(x_list, y_list, width=0.8, ec='k', color='gray')
plt.xlabel('Polymorph number', fontsize=40, labelpad=15)
plt.ylabel('Count', fontsize=40, labelpad=15)
#plt.title('Polymorph number', size=35)
plt.xticks(x_list, ['1','2','3','4','>5'], size=30, rotation=0)
plt.yticks(size=30)
plt.xlim([0.5,5.5])
#plt.ylim([0,1])
plt.show()



p_count = 0
p_data = []
u_count = 0
u_data = []
for d in data_10000:
    if d['icsd'] == False:
        u_count += 1
        u_data.append(d)
    elif d['icsd'] == True:
        p_count += 1
        p_data.append(d)

print(p_count/u_count)


train_p, hold_out_p = train_test_split(p_data, test_size=0.2, random_state=7)
train_p, val_p = train_test_split(train_p, test_size=0.2, random_state=7)
train_u, hold_out_u = train_test_split(u_data, test_size=0.2, random_state=7)
train_u, val_u = train_test_split(train_u, test_size=0.2, random_state=7)

random.shuffle(train_p)
random.shuffle(val_p)
random.shuffle(hold_out_p)
random.shuffle(train_u)
random.shuffle(val_u)
random.shuffle(hold_out_u)

mp30s10000_dataset = {}
mp30s10000_dataset['train_p'] = train_p
mp30s10000_dataset['train_u'] = train_u
mp30s10000_dataset['val_p'] = val_p
mp30s10000_dataset['val_u'] = val_u
mp30s10000_dataset['hold-out-test_p'] = hold_out_p
mp30s10000_dataset['hold-out-test_u'] = hold_out_u

with open('./data/mp30s10000_dataset.json', 'w') as f:
    json.dump(mp30s10000_dataset, f)


# Load prompt text
with open("prompts.json") as prompt_file:
    prompt_dict = json.load(prompt_file)
    synth_sys_prompt = prompt_dict["synth_sys_prompt"]


train_data = []
val_data = []
hold_out_data = []
for i, d in enumerate(train_p):
    request = {}
    request["messages"] = [
        {"role":"user", "content": synth_sys_prompt+ d["description"]},
        {"role":"assistant", "content": "P"}
    ]
    train_data.append(request)

for i, d in enumerate(train_u):
    request = {}
    request["messages"] = [
        {"role":"user", "content": synth_sys_prompt+ d["description"]},
        {"role":"assistant", "content": "U"}
    ]
    train_data.append(request)
    if i == len(train_p) -1:
        break

for i, d in enumerate(val_p):
    request = {}
    request["messages"] = [
        {"role":"user", "content": synth_sys_prompt+ d["description"]},
        {"role":"assistant", "content": "P"}
    ]
    val_data.append(request)

for i, d in enumerate(val_u):
    request = {}
    request["messages"] = [
        {"role":"user", "content": synth_sys_prompt+ d["description"]},
        {"role":"assistant", "content": "U"}
    ]
    val_data.append(request)
    if i == len(val_p) -1:
        break

for i, d in enumerate(hold_out_p):
    request = {}
    request["messages"] = [
        {"role":"user", "content": synth_sys_prompt+ d["description"]},
        {"role":"assistant", "content": "P"}
    ]
    hold_out_data.append(request)

for i, d in enumerate(hold_out_u):
    request = {}
    request["messages"] = [
        {"role":"user", "content": synth_sys_prompt+ d["description"]},
        {"role":"assistant", "content": "U"}
    ]
    hold_out_data.append(request)


print(len(train_data), len(val_data), len(hold_out_data)) # 49082 12272 20040


random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(hold_out_data)

save_dir = "./data/finetuning"
with open(save_dir+"/train_pu_struct_data.jsonl" , encoding= "utf-8",mode="w") as file:
    for i in train_data:
        file.write(json.dumps(i) + "\n")
with open(save_dir+"/val_pu_struct_data.jsonl" , encoding= "utf-8",mode="w") as file:
    for i in val_data:
        file.write(json.dumps(i) + "\n")
with open(save_dir+"/hold_out_pu_struct_data.jsonl" , encoding= "utf-8",mode="w") as file:
    for i in hold_out_data:
        file.write(json.dumps(i) + "\n")

#
