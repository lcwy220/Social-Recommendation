#coding:UTF-8

# 将深度模型的输入转换成svd处理的形式

dataset_path = 'ciao_v4/'
version = 'v4'
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict

with open(dataset_path + 'dataset_filter5_%s.pkl' %version, 'rb') as f:
    train_set = pickle.load(f)
    valid_set = pickle.load(f)
    test_set = pickle.load(f)

print(len(train_set), len(valid_set), len(test_set))

count_dict = defaultdict(int)
for u,i,r in train_set:
    count_dict[r] += 1
print(count_dict)


train_user_set = set([u for u,i,r in train_set])
train_item_set = set([i for u,i,r in train_set])
new_valid_set = []
for u,i,r in valid_set:
    if u in train_user_set and i in train_item_set and r != 0:
        new_valid_set.append((u,i,r))
new_test_set = []
for u,i,r in test_set:
    if u in train_user_set and i in train_item_set and r != 0:
        new_test_set.append((u,i,r))

valid_set = new_valid_set
test_set = new_test_set


with open(dataset_path + 'new_train_set_filter5_%s.txt' %version, 'w') as f:
    for u,i,r in train_set:
        f.write(' '.join([str(u), str(i), str(r)]) + '\n')

with open(dataset_path + 'new_vaild_set_filter5_%s.txt' %version, 'w') as f:
    for u,i,r in valid_set:
        f.write(' '.join([str(u), str(i), str(r)]) + '\n')

with open(dataset_path + 'new_test_set_filter5_%s.txt' %version, 'w') as f:
    for u,i,r in test_set:
        f.write(' '.join([str(u), str(i), str(r)]) + '\n')

print(len(train_set), len(valid_set), len(test_set))



with open(dataset_path + 'list_filter5_%s.pkl' %version, 'rb') as f:
    u_items_list = pickle.load(f)
    u_users_list = pickle.load(f)
    u_users_items_list = pickle.load(f)
    i_users_list = pickle.load(f)
    (user_count, item_count, rate_count) = pickle.load(f)

social_dict = defaultdict(int)
for u in range(1,user_count+1):
    social_dict[len(u_users_list[u])] += 1

print(social_dict)

"""
new_u_items_list = []
for items in u_items_list:
    tmp = [(i, int(r*2)) for i,r in items]
    new_u_items_list.append(tmp)

new_u_users_items_list = []
for items in u_users_list:
    tmp = []
    for u in items:
        tmp.append(new_u_items_list[u])
    new_u_users_items_list.append(tmp)

new_i_users_list = []
for items in i_users_list:
    tmp = [(u, int(r*2)) for u,r in items]
    new_i_users_list.append(tmp)

with open(dataset_path + 'list_filter5_z.pkl', 'wb') as f:
    pickle.dump(new_u_items_list, f)
    pickle.dump(u_users_list, f)
    pickle.dump(new_u_users_items_list,f)
    pickle.dump(new_i_users_list, f)
    pickle.dump((user_count, item_count, 8), f)

print(new_i_users_list[10])
print(i_users_list[10])

with open(dataset_path + "self_sf_user_list_filter5.pkl", 'rb') as f:
    sf_users = pickle.load(f)

new_sf_users_items_list = []
for items in sf_users:
    tmp = []
    for u,_ in items:
        tmp.append(new_u_items_list[u])
    new_sf_users_items_list.append(tmp)

with open(dataset_path + "self_sf_user_items_list_filter5_z.pkl", "wb") as f:
    pickle.dump(new_sf_users_items_list, f)

with open(dataset_path + "bal_sample_item_list_filter5.pkl", 'rb') as f:
    if_items = pickle.load(f)

new_if_items_users_list = []
for items in if_items:
    tmp = []
    for i,_ in items:
        tmp.append(new_i_users_list[i])
    new_if_items_users_list.append(tmp)
with open(dataset_path + 'bal_sample_item_users_list_filter5_z.pkl', 'wb') as f:
    pickle.dump(new_if_items_users_list, f)
"""

f = open(dataset_path + 'trust_data_%s.txt' %version, 'w')
count = 0
for idx, items in enumerate(u_users_list):
    if items == [0]:
        print(idx, 'no relations')
        count += 1
        continue
    for each in items:
        f.write(' '.join([str(idx), str(each), str(1)]) + '\n')
f.close()

print(count)
