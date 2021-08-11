import pandas as pd
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle
import numpy as np

import json
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np

comb_list = [ ['age'], 
              ['ethnicity'],
              ['target_gender'],
              ['age', 'ethnicity'],
              ['age', 'target_gender'],
              ['ethnicity', 'target_gender'],
              ['age', 'ethnicity', 'target_gender']
            ]

def task_comb_data(df, task_combs, conditions):
    selected_rows = np.array([True]*len(df))
    for task, condition in zip(task_combs, conditions):
        selected_rows = selected_rows & (df[task].to_numpy()==condition)
    return selected_rows

def binary_label_dist(df, col_name="label"):
    label_dist = Counter(df[col_name])
    label_dist = np.array([label_dist[0], label_dist[1]])
    label_dist = label_dist/sum(label_dist)
    return label_dist

def full_label_data(df, tasks):
    selected_rows = np.array([True]*len(df))
    for task in tasks:
        selected_rows = selected_rows & df[task].notnull().to_numpy()
    return selected_rows


def cal_fpr(fp, tn):
    '''False positive rate'''
    return fp/(fp+tn)


def cal_fnr(fn, tp):
    '''False negative rate'''
    return fn/(fn+tp)


def cal_tpr(tp, fn):
    '''True positive rate'''
    return tp/(tp+fn)


def cal_tnr(tn, fp):
    '''True negative rate'''
    return tn/(tn+fp)


def eval(df, pred="pred"):
    # get the task name from the file, gender or ethnicity
    tasks = ['gender', 'age', 'country', 'ethnicity']

    scores = {
        'accuracy': 0.0,
        'f1-macro': 0.0, # macro f1 score
        'f1-weight': 0.0, # weighted f1 score
        # 'auc': 0.0,
    }

    # accuracy, f1, auc
    scores['accuracy'] = metrics.accuracy_score(
        y_true=df.label, y_pred=df[pred]
    )
    scores['f1-macro'] = metrics.f1_score(
        y_true=df.label, y_pred=df[pred],
        average='macro'
    )
    scores['f1-weight'] = metrics.f1_score(
        y_true=df.label, y_pred=df[pred],
        average='weighted'
    )
    # fpr, tpr, _ = metrics.roc_curve(
    #     y_true=df.label, y_score=df["pred"]_prob,
    # )
    # scores['auc'] = metrics.auc(fpr, tpr)

    '''fairness gaps'''
    for task in tasks:

        scores[task] = {
            'fned': 0.0, # gap between fnr
            'fped': 0.0, # gap between fpr
            'tped': 0.0, # gap between tpr
            'tned': 0.0, # gap between tnr
        }
        # filter out the one does not have attributes
        task_df = df[df[task].notnull()]
    
        # get overall confusion matrix
        tn, fp, fn, tp = metrics.confusion_matrix(
            y_true=task_df.label, y_pred=task_df[pred]
        ).ravel()
        # print(cal_fnr(fn, tp), 
        #       cal_fpr(fp, tn),
        #       cal_tpr(tp, fn),
        #       cal_tnr(tn, fp)
        #       )
        # print(tn, fp, fn, tp)

        # get the unique types of demographic groups
        uniq_types = task_df[task].unique()
        for group in uniq_types:
            # calculate group specific confusion matrix
            group_df = task_df[task_df[task] == group]
            
            g_tn, g_fp, g_fn, g_tp = metrics.confusion_matrix(
                y_true=group_df.label, y_pred=group_df[pred]
            ).ravel()

            # calculate and accumulate the gaps
            scores[task]['fned'] = scores[task]['fned'] + abs(
                cal_fnr(fn, tp)-cal_fnr(g_fn, g_tp)
            )
            scores[task]['fped'] = scores[task]['fped'] + abs(
                cal_fpr(fp, tn)-cal_fpr(g_fp, g_tn)
            )
            scores[task]['tped'] = scores[task]['tped'] + abs(
                cal_tpr(tp, fn)-cal_tpr(g_tp, g_fn)
            )
            scores[task]['tned'] = scores[task]['tned'] + abs(
                cal_tnr(tn, fp)-cal_tnr(g_tn, g_fp)
            )

    print(scores)
    return scores

def linear_leakage(train_text_embd, train_author_label, test_text_embd, test_author_label, output=True):
    # leakage
    attack_model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    attack_model.fit(train_text_embd, train_author_label)
    leakage = attack_model.score(test_text_embd, test_author_label)
    Y_test_hat = attack_model.predict(test_text_embd)
    F1 = f1_score(test_author_label, Y_test_hat, average = "macro")
    if output:
        # print("Leakage Acc: {}".format(leakage))
        print("Leakage F1: {}".format(F1))
    return (leakage, F1)

def task_comb_data(df, 
                task_combs, 
                conditions,):
    selected_rows = np.array([True]*len(df))
    for task, condition in zip(task_combs, conditions):
        selected_rows = selected_rows & (df[task].to_numpy()==condition)
    return selected_rows

def task_comb_data(df, 
                task_combs, 
                conditions,):
    selected_rows = np.array([True]*len(df))
    for task, condition in zip(task_combs, conditions):
        selected_rows = selected_rows & (df[task].to_numpy()==condition)
    return selected_rows

def get_all_combs(unique_type_list):
    number_tasks = len(unique_type_list)
    no_unique_types = [len(unique_type) for unique_type in unique_type_list]+[1]
    total_number = np.prod(no_unique_types)
    # print(number_tasks, total_number)
    
    # init 2d matrix
    group_combinations = [[None for j in range(number_tasks)] for i in range(total_number)]

    for single_task_id, single_task_types in enumerate(unique_type_list):
        # print(single_task_id, single_task_types)

        # calculate single repeat time
        single_repeat_time = int(np.prod(no_unique_types[single_task_id+1:]))
        # calculate whole list repeat time
        whole_repeat_time = int(total_number/single_repeat_time/len(single_task_types))
        # print(single_repeat_time, whole_repeat_time)

        # create col number
        task_col = []
        # single repeat
        for t_type in single_task_types:
            task_col = task_col + [t_type]*single_repeat_time
        # whole repeat
        task_col = task_col * whole_repeat_time
        # print(task_col)

        # add to matrix
        for i, v in enumerate(task_col):
            group_combinations[i][single_task_id] = v
    return group_combinations


def combination_eval(in_df,
                     task_combs = comb_list,
                     tasks = ['gender', 'age', 'country', 'ethnicity'],
                     label = "label", 
                     pred="pred",
                     all_labels = False,
                     print_results = True
                     ):
    # save dist at different level
    dist_results = {}
    scores = {} 
    
    if all_labels:
        df = in_df[full_label_data(in_df, tasks)]
    else:
        df = in_df

    overall_dist = binary_label_dist(df)
    dist_results["overall"] = (overall_dist, len(df))

    # overall dist
    overall_label = list(df[label])
    overall_pred = list(df[pred])

    # accuracy, f1, auc
    scores['accuracy'] = metrics.accuracy_score(
        y_true=overall_label, y_pred=overall_pred
    )
    scores['f1-macro'] = metrics.f1_score(
        y_true=overall_label, y_pred=overall_pred,
        average='macro'
    )
    scores['f1-weight'] = metrics.f1_score(
        y_true=overall_label, y_pred=overall_pred,
        average='weighted'
    )

    # get overall confusion matrix
    cnf = metrics.confusion_matrix(
        y_true=overall_label, y_pred=overall_pred
    )
    cnf = cnf/np.sum(cnf,axis=1)[:, np.newaxis]
    t00R, f01R, f10R, t11R = cnf.ravel()
    
    scores['t00'] = t00R
    scores['f01'] = f01R
    scores['f10'] = f10R
    scores['t11'] = t11R

    '''fairness gaps'''
    for task_comb in task_combs:
        # get all group label combinations
        # group_combinations = [p for p in product([0, 1], repeat=len(task_comb))]
        
        comb_uniq_types = [df[~df[t].isnull()][t].unique() for t in task_comb]

        group_combinations = get_all_combs(comb_uniq_types)
        
        scores["-".join(task_comb)] = {}
        
        # a full label subset
        t_df = df[full_label_data(df, task_comb)]

        # tasks rates
        t_cnf = metrics.confusion_matrix(
            y_true=list(t_df[label]), y_pred=list(t_df[pred])
        )

        t_cnf_normalized = t_cnf/np.sum(t_cnf,axis=1)[:, np.newaxis]

        t_t00R, t_f01R, t_f10R, t_t11R = t_cnf_normalized.ravel()
        task_rates = {
            # "cf" : t_cnf,
            "t-t00R" : t_t00R,
            "t-f01R" : t_f01R, 
            "t-f10R" : t_f10R, 
            "t-t11R" : t_t11R,
            "mean-GAP-overall" : 0.0,
            "mean-GAP-subset" : 0.0
        }

        scores["-".join(task_comb)]['subset-rates'] = task_rates

        for group_comb in group_combinations:
            # group scores
            task_comb_scores = {}

            group_df = df[task_comb_data(df, task_comb, group_comb)]
            # print(group_df)
            group_key = "_".join(task_comb+[str(i) for i in group_comb])
            dist_results[group_key] = (binary_label_dist(group_df), len(group_df))

            # group rates
            g_cnf = metrics.confusion_matrix(
                y_true=list(group_df[label]), y_pred=list(group_df[pred])
            )
            g_cnf = g_cnf/np.sum(g_cnf,axis=1)[:, np.newaxis]

            try:
                g_t00R, g_f01R, g_f10R, g_t11R = g_cnf.ravel()
            except:
                print(group_key)
                continue

            task_comb_scores["Number"] = len(group_df)

            task_comb_scores["t00R"] = g_t00R
            task_comb_scores["f01R"] = g_f01R
            task_comb_scores["f10R"] = g_f10R
            task_comb_scores["t11R"] = g_t11R
            
            # GAP compared to overall
            task_comb_scores["GAP-overall-t00R"] = g_t00R - t00R
            task_comb_scores["GAP-overall-f01R"] = g_f01R - f01R
            task_comb_scores["GAP-overall-f10R"] = g_f10R - f10R
            task_comb_scores["GAP-overall-t11R"] = g_t11R - t11R

            scores["-".join(task_comb)]['subset-rates']["mean-GAP-overall"] += (abs(g_f01R - f01R)+abs(g_f10R - f10R))

            # GAP compared to the full label subset 
            task_comb_scores["GAP-subset-t00R"] = g_t00R - t_t00R
            task_comb_scores["GAP-subset-f01R"] = g_f01R - t_f01R
            task_comb_scores["GAP-subset-f10R"] = g_f10R - t_f10R
            task_comb_scores["GAP-subset-t11R"] = g_t11R - t_t11R

            scores["-".join(task_comb)]['subset-rates']["mean-GAP-subset"] += (abs(g_f01R - t_f01R)+abs(g_f10R - t_f10R))
 
            # accuracy, f1, auc
            task_comb_scores['accuracy'] = metrics.accuracy_score(
                y_true=list(group_df[label]), y_pred=list(group_df[pred])
            )
            task_comb_scores['f1-macro'] = metrics.f1_score(
                y_true=list(group_df[label]), y_pred=list(group_df[pred]),
                average='macro'
            )
            task_comb_scores['f1-weight'] = metrics.f1_score(
                y_true=list(group_df[label]), y_pred=list(group_df[pred]),
                average='weighted'
            )
            scores["-".join(task_comb)]["-".join([str(i) for i in group_comb])] = task_comb_scores
        scores["-".join(task_comb)]['subset-rates']["mean-GAP-overall"] = scores["-".join(task_comb)]['subset-rates']["mean-GAP-overall"]/len(group_combinations)
        scores["-".join(task_comb)]['subset-rates']["mean-GAP-subset"] = scores["-".join(task_comb)]['subset-rates']["mean-GAP-subset"]/len(group_combinations)
    return dist_results, scores