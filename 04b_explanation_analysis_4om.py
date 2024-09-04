# This script can be run in anaconda base environment.
from tqdm import tqdm
import numpy as np
import pandas as pd
import re
import jsonlines
import json
import pickle
import matplotlib.pyplot as plt
from colour import Color
from pymatgen.core import Composition
from collections import Counter


def grouping_similar_keyword(keyword):
    if keyword in ['Coordination Geometry', 'Coordination Environment', 'Bonding Geometry',
                     'Bond Lengths and Angles', 'Geometric Constraints', 'Geometric Stability',
                     'Coordination Complexity', 'Geometric Coordination', 'Stable Coordination Geometry',
                     'Geometric Complexity', 'Complex Coordination Environment', 'Geometrical Constraints',
                     'Bond Geometry', 'Geometric Distortion', 'Geometric Strain', 'Bonding Environment',
                     'Geometric Compatibility', 'Unusual Coordination Environment', 'Bond Coordination',
                     'High Coordination Numbers', 'Steric Hindrance', 'Steric hindrance', 'Lattice Strain',
                     'Bond Lengths and Coordination', 'Bond Lengths and Geometries', 'Complexity',
                     'Structural Complexity', 'Crystal Structure',
                     'Octahedral Tilt Angles', 'Octahedral Tilting', 'Octahedral Distortion',
                     ]:
        keyword = 'Coordination Geometry'
        class_idx = 0

    elif keyword in ['Bond Lengths', 'Bond Length Discrepancies', 'Bond Length Discrepancy',
                   'Bond Length Variability', 'Bond Length Consistency', 'Stable Bond Lengths',
                   'Unstable Bond Lengths', 'Bond Length Variations', 'Bond Length Mismatch',
                   'Bond Length Inconsistencies', 'Bond Length Disparities', 'Bond Length Compatibility',
                   'Bond Length', 'Unfavorable Bond Lengths', 'Bond Distances', 'Consistent Bond Lengths',
                   'Bond Length Variation', 'Interatomic Distances', 'Bonding Interactions',
                   'Bond Length Disparity', 'Inconsistent Bond Lengths', 'Unusual Bond Lengths',
                   'Bond length discrepancies']:
        keyword = 'Bond Lengths'
        class_idx = 1

    elif keyword in ['Symmetry and Space Group', 'Space Group Symmetry', 'Symmetry', 'Crystallographic Symmetry',
                     'Crystal Symmetry', 'Space Group Constraints', 'Crystallographic Stability',
                     'Space Group', 'Symmetry Constraints', 'Crystallographic Space Group',
                     'Crystallographic Constraints', 'Symmetry and Stability', 'Crystallography',
                     'Symmetry and Coordination']:
        keyword = 'Symmetry and Space Group'
        class_idx = 2

    elif keyword in ['Chemical Compatibility', 'Charge Imbalance', 'Electronic Configuration',
                     'Charge Balance', 'Chemical Incompatibility', 'Chemical Composition',
                     'Elemental Compatibility', 'Electrostatic Imbalance', 'Stoichiometric Imbalance',
                     'Charge Balance Issues']:
        keyword = 'Compositional Compatibility'
        class_idx = 3

    elif keyword in ['Thermodynamic Stability', 'Thermodynamic Instability', 'Phase Stability',
                     'Electronic Structure', 'Electronic Instability',
                     'Structural Stability', 'Structural Instability',
                     'Crystal Structure Stability', 'Stable Crystal Structure','Crystalline Structure',
                     ]:
        keyword = 'Thermodynamic Stability'
        class_idx = 4

    elif keyword in ['Size Mismatch', 'Atomic Size Mismatch', 'Ionic Size Mismatch', 'Ionic Radius Mismatch']:
        keyword = 'Atomic Size Mismatch'
        class_idx = 5

    elif keyword in ['Synthesis Conditions', 'Kinetic Barriers', 'Chemical Reactivity']:
        keyword = 'Synthetic Aspect'
        class_idx = 6

    elif keyword in ['Polyhedral Connectivity', 'Octahedral Coordination', 'Corner-Sharing Polyhedra',
                     'Coordination Polyhedra', 'Corner-Sharing Octahedra', 'Interatomic Interactions']:
        keyword = 'Polyhedral Connectivity'
        class_idx = 7

    elif keyword in ['Inequivalent Sites', 'Multiple Inequivalent Sites']:
        keyword = 'Inequivalent Sites'
        class_idx = 8

    elif keyword in ['Dimensionality']:
        keyword = 'Plane-orientation Stability' # Convert into more intuitive expression
        class_idx = 9
    else:
        class_idx = 10

    return keyword, class_idx



# load
with open('./result/explanation_ppp_nnn_formatted_with_embedding_4om.pickle', 'rb') as f:
    embedded_data = pickle.load(f)
# embedded_data[0].keys() == {'Label', 'Prompt', 'Explanation', 'formatted_Explanation', 'GPTembedding'}


examples = []
example1 = {}
example1['Prompt'] = embedded_data[0]['Prompt']
example1['Explanation'] = embedded_data[0]['Explanation']
example2 = {}
example2['Prompt'] = embedded_data[-1]['Prompt']
example2['Explanation'] = embedded_data[-1]['Explanation']
examples = [example1, example2]

# save two examples
with open("./result/two_explanation_examples_4om.json", encoding= "utf-8",mode="w") as jsonfile:
    json.dump(examples, jsonfile, indent=4)



label_list = []
embedding_list = []
reason_count_dict = {}
for dd in embedded_data:
    if dd['Label'] == 'ppp':
        label = 1
    elif dd['Label'] == 'nnn':
        label = 0
    else:
        raise NotImplementedError("Invalid label")
    for embed in dd['GPTembedding']:
        label_list.append(label)
        embedding_list.append(embed)

    number_of_reasons = len(dd['GPTembedding'])
    if number_of_reasons not in reason_count_dict.keys():
        reason_count_dict[number_of_reasons] = 1
    else:
        reason_count_dict[number_of_reasons] += 1


# Draw # of reasons distribution
plt.figure(figsize=(8,5))
for key, value in reason_count_dict.items():
    plt.bar(key, value, edgecolor='dimgray', color='dimgray')
#plt.title("Number of reasons distribution", fontsize=25)
plt.xlabel('Number of reasons', fontsize=25)
plt.ylabel('Counts', fontsize=25)
plt.xticks(range(0,max(list(reason_count_dict.keys()))+1,1),fontsize=20)
plt.yticks(fontsize=20)
plt.xlim([0,max(list(reason_count_dict.keys()))+1])
plt.show()


# t-SNE analysis
from sklearn.manifold import TSNE

X = embedding_list
y = label_list
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(np.array(X))

plt.figure(figsize=(8,8))
plt.scatter(X_tsne[np.array(y)==1][:,0], X_tsne[np.array(y)==1][:,1], color='lightseagreen', alpha=0.4, lw=2)
plt.scatter(X_tsne[np.array(y)==0][:,0], X_tsne[np.array(y)==0][:,1], color='lightcoral' , alpha=0.4, lw=2)
plt.scatter(X_tsne[np.array(y)==1][:,0], X_tsne[np.array(y)==1][:,1], color='lightseagreen', alpha=0.2, lw=2)
#plt.title("t-SNE", fontsize=25)
plt.legend(['Synthesizable', 'Not-synthesizable'], loc='upper left', shadow=False, scatterpoints=1, fontsize=20)
plt.xlabel('t-SNE feature 1', fontsize=25)
plt.ylabel('t-SNE feature 2', fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()



# Draw 10 most relevant reasons for synthesizability
keywords_dict = {}
class_idx_list = []
for dd in embedded_data:
    for expl in dd['formatted_Explanation']:
        try:
            keyword = expl.split('**')[1]
            t_keyword, class_idx = grouping_similar_keyword(keyword)

            if t_keyword not in keywords_dict.keys():
                keywords_dict[t_keyword] = 1
            else:
                keywords_dict[t_keyword] += 1
            class_idx_list.append(class_idx)
        except:
            t_keyword = 'Invalid answer'
            if t_keyword not in keywords_dict.keys():
                keywords_dict[t_keyword] = 1
            else:
                keywords_dict[t_keyword] += 1
            class_idx_list.append(10)

sorted_dict = sorted(keywords_dict.items(), key= lambda item:item[1], reverse=True)

reasons_list = []
counts_list = []
for i in range(len(sorted_dict[:10])):
    print(sorted_dict[i])
    reasons_list.append(sorted_dict[i][0])
    counts_list.append(sorted_dict[i][1])
reasons_list.append('⋮              ')
counts_list.append(0)

red = Color("red")
colors = list(red.range_to(Color("purple"),10))
colors = [color.rgb for color in colors] + ['dimgrey']

plt.figure(figsize=(10,7))
plt.barh(reasons_list[::-1], counts_list[::-1], color=colors[::-1])
#plt.title("10 most relevant reasons for synthesizability", fontsize=35)
plt.xlabel('Counts', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=25, fontstyle='italic')
plt.ylim([0,11])
plt.show()



# Specify the regions of class_idx in t-SNE plot
class_idx_ary = np.array(class_idx_list)

plt.figure(figsize=(8,8))
plt.scatter(X_tsne[class_idx_ary==10][:,0], X_tsne[class_idx_ary==10][:,1], color=colors[10], alpha=0.4, lw=2)
for idx in range(10):
    plt.scatter(X_tsne[class_idx_ary==idx][:,0], X_tsne[class_idx_ary==idx][:,1], color=colors[idx], alpha=0.4, lw=2)
#plt.title("t-SNE", fontsize=25)
#plt.legend(['Synthesizable', 'Not-synthesizable'], loc='best', shadow=False, scatterpoints=1, fontsize=20)
plt.xlabel('t-SNE feature 1', fontsize=25)
plt.ylabel('t-SNE feature 2', fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()




#
