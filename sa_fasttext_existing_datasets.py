# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 03:35:44 2022

@author: G_A.Papadakis
"""

import numpy as np
import pandas as pd
import time
from gensim.models.fasttext import load_facebook_model
from scipy import spatial, stats

def get_similarities(d1_cols, d2_cols, row):
    entity1 = ''
    for i in d1_cols:
        entity1 = " ".join([entity1, str(row[i])])
        
    entity2 = ''
    for i in d2_cols:
        entity2 = " ".join([entity2, str(row[i])])
        
    if entity1 and entity2:
        vec1 = np.array(voc[entity1])
        vec2 = np.array(voc[entity2])

        cs = 1 - spatial.distance.cosine(vec1, vec2)
        es = 1/(1+spatial.distance.euclidean(vec1, vec2))
        ws = 1/(1+stats.wasserstein_distance(vec1, vec2))
        
        return cs, es, ws
    return 0, 0, 0

def get_similarity(d1_cols, d2_cols, measure_id, row):
    entity1 = ''
    for i in d1_cols:
        entity1 = " ".join([entity1, str(row[i])])

    entity2 = ''
    for i in d2_cols:
        entity2 = " ".join([entity2, str(row[i])])

    if entity1 and entity2:
        vec1 = np.array(voc[entity1])
        vec2 = np.array(voc[entity2])
        
        if (measure_id == 0):
            return 1 - spatial.distance.cosine(vec1, vec2)
        elif (measure_id == 1):
            return 1/(1+spatial.distance.euclidean(vec1, vec2))
        elif (measure_id == 2):    
            return 1/(1+stats.wasserstein_distance(vec1, vec2))
     
model = load_facebook_model('/vectors/wiki.en.bin')
voc = model.wv

similarities = ['CS', 'ES', 'WS']
main_dir = '/data/existingDatasets/'
datasets = ['structured_dblp_acm', 'structured_dblp_scholar', 'structured_itunes_amazon',
'structured_walmart_amazon', 'structured_beer', 'structured_amazon_google',
'dirty_dblp_acm', 'dirty_dblp_scholar', 'dirty_itunes_amazon', 'dirty_walmart_amazon',
'textual_abt_buy', 'textual_company', 'structured_fodors_zagats']

for dataset in datasets:
    current_dir = main_dir + dataset
    print('\n\n' + current_dir)
    
    train = pd.read_csv(current_dir + '/train.csv', na_filter=False)
    valid = pd.read_csv(current_dir + '/valid.csv', na_filter=False)
    test = pd.read_csv(current_dir + '/test.csv', na_filter=False)
    
    d1_cols, d2_cols = [], []
    for index in range(4, len(train.columns)):
        if (train.columns[index].startswith("table1")):
            d1_cols.append(index)
        elif (train.columns[index].startswith("table2")):
            d2_cols.append(index)

    print('D1 Colums', d1_cols)
    print('D2 Colums', d2_cols)

    time_1 = time.time()
    
    best_thresholds = []
    train['CS'], train['ES'], train['WS'] = zip(*train.apply(lambda row : get_similarities(d1_cols, d2_cols, row), axis = 1))
    for measure_id in range(len(similarities)):
        best_F1, bestThr = -1, -1
        for threshold in np.arange(0.01, 1.00, 0.01):                
            train['pred_label'] = threshold <= train[similarities[measure_id]]
            
            tp = len(train[(train['label'] == 1) & train['pred_label']])
            fp = len(train[(train['label'] == 0) & train['pred_label']])
            fn = len(train[(train['label'] == 1) & (train['pred_label'] == False)])
            
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            if ((0 < precision) & (0 < recall)):
                f1 = 2 * precision * recall / (precision + recall)
                if (best_F1 < f1):
                    best_F1 = f1
                    bestThr = threshold
        print(best_F1, bestThr)
        best_thresholds.append(bestThr)
    
    best_F1, best_measure_id = -1, -1
    valid['CS'], valid['ES'], valid['WS'] = zip(*valid.apply(lambda row : get_similarities(d1_cols, d2_cols, row), axis = 1))
    for measure_id in range(len(similarities)):            
        valid['pred_label'] = best_thresholds[measure_id] <= valid[similarities[measure_id]]
        
        tp = len(valid[(valid['label'] == 1) & valid['pred_label']])
        fp = len(valid[(valid['label'] == 0) & valid['pred_label']])
        fn = len(valid[(valid['label'] == 1) & (valid['pred_label'] == False)])
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        if ((0 < precision) & (0 < recall)):
            f1 = 2 * precision * recall / (precision + recall)
            if (best_F1 < f1):
                best_F1 = f1
                best_measure_id = measure_id
    
    time_2 = time.time()
        
    test['sim'] = test.apply(lambda row : get_similarity(d1_cols, d2_cols, best_measure_id, row), axis = 1)

    test['pred_label'] = best_thresholds[best_measure_id] <= test['sim']
    tp = len(test[(test['label'] == 1) & test['pred_label']])
    fp = len(test[(test['label'] == 0) & test['pred_label']])
    fn = len(test[(test['label'] == 1) & (test['pred_label'] == False)])

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    
    time_3 = time.time()
    
    print('Validation results', best_F1, similarities[best_measure_id], best_thresholds[best_measure_id])
    print('Final F1', 2 * precision * recall / (precision + recall))
    print('Training time (sec)', time_2-time_1)
    print('Testing time (sec)', time_3-time_2)