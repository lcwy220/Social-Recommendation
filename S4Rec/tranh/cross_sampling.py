import pickle
import random
import sys
import os
from collections import defaultdict

# you need to change the dataset_path for different data
dataset_name = 'Ciao'
dataset_path = '../dataset/'+dataset_name+'/'

if not os.path.exists(dataset_name):
    os.mkdir(dataset_name)

with open(dataset_path + 'dataset_filter5.pkl', 'rb') as f:
    train_set = pickle.load(f)
    valid_set = pickle.load(f)
    test_set = pickle.load(f)

with open(dataset_path + 'list_filter5.pkl', 'rb') as f:
    u_items_list = pickle.load(f)
    u_users_list = pickle.load(f)
    u_users_items_list = pickle.load(f)
    i_users_list = pickle.load(f)
    (user_count, item_count, rate_count) = pickle.load(f)


u_dict = defaultdict(list)
i_dict = defaultdict(list)


for u,i, r in train_set:
    if r == 0:
        continue
    u_dict[str(u)+'_'+str(r)].append(i)
    i_dict[str(i)+'_'+str(r)].append(u)




neg_train_list = []
pos_train_list = []
all_user_set = set(range(1, user_count+1))
all_item_set = set(range(1, item_count+1))

count = 0
new_train_set = []
pos_new_train_list = []

for u,i, r in train_set:
    if r == 0:
        continue
    new_train_set.append((u,i,r)) #

    tmp_list = []
    # specify to 1
    if r == 1:
        for idx in [2,3,4,5]:
            item_set = set(u_dict[str(u)+'_'+str(idx)])
            for k in item_set:
                pos_train_list.append((u,i,1))
                neg_train_list.append((u,k,1))
                count += 1
                if count % 1000 == 0:
                    print(count)

            user_set = set(i_dict[str(i)+'_'+str(idx)])
            for k in user_set:
                pos_train_list.append((u,i,1))
                neg_train_list.append((k,i,1))
                tmp_list.append((k,i,1))
                count += 1
                if count % 1000 == 0:
                    print(count)


    for j in range(1,r):
        item_set = set(u_dict[str(u)+'_'+str(j)])
        for k in item_set:
            pos_train_list.append((u,i,r))
            neg_train_list.append((u,k,r))
            tmp_list.append((u,k,r))
            count += 1
            if count % 1000 == 0:
                print(count)
            
        user_set = set(i_dict[str(i)+'_'+str(j)])
        for k in user_set:
            pos_train_list.append((u,i,r))
            neg_train_list.append((k,i,r))
            tmp_list.append((k,i,r))
            count += 1
            if count % 1000 == 0:
                print(count)
        
    neg_tmp_list = random.sample(tmp_list, min(30, len(tmp_list)))
    pos_new_train_list.append(neg_tmp_list)

print(len(new_train_set))
print(len(pos_train_list), len(neg_train_list))


with open('%s/new_train_set_transh_30.pkl' %dataset_name, 'wb') as f:
    pickle.dump(new_train_set, f)
    pickle.dump(pos_new_train_list, f)

with open('%s/relation_train_set.pkl' %dataset_name, 'wb') as f:
    pickle.dump(pos_train_list, f)
    pickle.dump(neg_train_list, f)


