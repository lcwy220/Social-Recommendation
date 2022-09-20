# -*-coding:utf-8-*-

import math
import pickle
import numpy as np
import random
from collections import defaultdict, Counter



def implicit_item_relations(dataset_path):
   with open(dataset_path + 'dataset_filter5.pkl', 'rb') as f:
        train_set = pickle.load(f)
        vaild_set = pickle.load(f)
        test_set = pickle.load(f)

   with open(dataset_path + 'list_filter5.pkl', 'rb') as f:
        u_item_list = pickle.load(f)
        u_users_list = pickle.load(f)
        u_users_items_list = pickle.load(f)
        i_users_list = pickle.load(f)
        (user_count, item_count, rate_count) = pickle.load(f)


   item_graph_dict = defaultdict(dict)
   for uid in range(1, user_count+1):
       item_list = u_item_list[uid]
       for idx, item in enumerate(item_list):
           for idy in range(idx+1, len(item_list)):
               idx_iid, idx_score = item_list[idx]
               idy_iid, idy_score = item_list[idy]
               score = 1/(abs(idx_score-idy_score)+1)

               if idx_iid in item_graph_dict:
                   if idy_iid in item_graph_dict[idx_iid]:
                       item_graph_dict[idx_iid][idy_iid] += score
                   else:
                       item_graph_dict[idx_iid][idy_iid] = score
               else:
                   item_graph_dict[idx_iid][idy_iid] = score


               if idy_iid in item_graph_dict:
                   if idx_iid in item_graph_dict[idy_iid]:
                       item_graph_dict[idy_iid][idx_iid] += score
                   else:
                       item_graph_dict[idy_iid][idx_iid] = score
               else:
                   item_graph_dict[idy_iid][idx_iid] = score
                   similar_item_list = []

   similar_if_users_list = []
   for i in range(0, item_count+1):
       tmp_list = [(i, 1)]
       tmp_user_list = [i_users_list[i]]
       if i in item_graph_dict:
           similar_is = sorted(item_graph_dict[i].items(), key=lambda x:x[1], reverse=True)
           for k,v in similar_is:
               if v >= 1:
                   tmp_list.append((k,v))
                   tmp_user_list.append(i_users_list[k])

       similar_if_users_list.append(tmp_user_list)
       similar_item_list.append(tmp_list)

   with open(dataset_path + 'bal_sample_item_list_filter5.pkl', 'wb') as f:
        pickle.dump(similar_item_list, f)
   with open(dataset_path + 'bal_sample_item_users_list_filter5.pkl', 'wb') as f:
        pickle.dump(similar_if_users_list, f)




def implicit_user_relations(dataset_path):
    with open(dataset_path + 'list_filter5.pkl', 'rb') as f:
        u_item_list = pickle.load(f)
        u_users_list = pickle.load(f)
        u_users_items_list = pickle.load(f)
        i_users_list = pickle.load(f)
        (user_count, item_count, rate_count) = pickle.load(f)


    share_fans_rel = dict()
    rel_set = set()
    for i in range(1, user_count+1):
        followee_list = u_users_list[i]
        if len(followee_list) < 2:
            continue

        for idx_1, u1 in enumerate(followee_list):
            for u2 in followee_list[idx_1+1:]:
                if u1 in share_fans_rel:
                    share_fans_rel[u1][u2] += 1
                else:
                    share_fans_rel[u1] = defaultdict(int)
                    share_fans_rel[u1][u2] += 1

                if u2 in share_fans_rel:
                    share_fans_rel[u2][u1] += 1
                else:
                    share_fans_rel[u2] = defaultdict(int)
                    share_fans_rel[u2][u1] += 1

                rel_set.add((u1, u2))
                rel_set.add((u2, u1))



    u_haskey = list(share_fans_rel.keys())
    shrink_rel = dict()
    for k,v in share_fans_rel.items():
        sorted_v = sorted(v.items(), key=lambda x:x[1], reverse=True)
        sorted_v = random.sample(sorted_v, min(len(sorted_v),29)) # reduce the probability of coverage with u-u follow relation
        tmp = [(k, 1)]
        tmp.extend(sorted_v)
        shrink_rel[k] = tmp

    non_haskey = set(range(0, user_count+1)) - set(u_haskey)
    for k in non_haskey:
        shrink_rel[k] = [(k,1)]

    shrink_rel_set = set()
    sorted_rel = sorted(shrink_rel.items(), key=lambda x:x[0], reverse=False)
    new_sorted_rel = []
    for kt, vt in sorted_rel:
        new_sorted_rel.append(vt)

        for id1, v1 in vt:
            shrink_rel_set.add((kt, id1))
            shrink_rel_set.add((id1, kt))


    sf_user_item_list = []
    for uid, sf_list in sorted_rel:
        if sf_list == [(0,0)]:
            sf_user_item_list.append( [[(0,0)]])
            continue

        uu_items = []
        for sf_idx, rel in sf_list:
            uu_items.append(u_item_list[sf_idx])
        sf_user_item_list.append(uu_items)


    print('not has sf: ',len(non_haskey))
    print('has sf: ', len(u_haskey))


    ## self: 增加自身

    with open(dataset_path + 'self_sf_user_list_filter5.pkl', 'wb') as f:
        pickle.dump(new_sorted_rel, f)

    with open(dataset_path + 'self_sf_user_items_list_filter5.pkl', 'wb') as f:
        pickle.dump(sf_user_item_list, f)






if __name__ == '__main__':
    implicit_item_relations('Epinions/')
    implicit_user_relations('Epinions/')



