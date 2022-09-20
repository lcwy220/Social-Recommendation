import torch
import random

truncate_len = 30 #30
truncate_len_i = 20 #20
max_neg_size = 30

"""
Ciao dataset info:
Avg number of items rated per user: 38.3
Avg number of users interacted per user: 2.7
Avg number of users connected per item: 16.4
"""

def collate_fn(batch_data):
    """This function will be used to pad the graph to max length in the batch
       It will be used in the Dataloader
    """
    uids, iids, labels = [], [], []
    u_items, u_users, u_users_items, i_users, sf_users, sf_users_items, i_friends, if_items_users = [], [], [], [], [], [], [], []
    u_items_len, u_users_len, i_users_len, sf_users_len, sf_items_len, i_friends_len, if_users_len = [], [], [], [], [], [], []
    neg_transh_list, pos_transh_list = [], []
    neg_len = []
    count = 0
    for data, u_items_u, u_users_u, u_users_items_u, i_users_i, u_sf_users, u_sf_users_items, i_items_friends, i_items_users, neg_tr_list in batch_data:

        (uid, iid, label) = data
        uids.append(uid)
        iids.append(iid)
        labels.append(label)

        neg_size = min(len(neg_tr_list), max_neg_size)
        neg_list = random.sample(neg_tr_list, neg_size)
        neg_transh_list.extend(neg_list)
        pos_transh_list.extend([data] * neg_size)
        neg_len.append(neg_size)

        # user-items    
        if len(u_items_u) <= truncate_len:
            u_items.append(u_items_u)
        else:
            u_items.append(random.sample(u_items_u, truncate_len))
        u_items_len.append(min(len(u_items_u), truncate_len))

        # user_sf
        if len(u_sf_users) < truncate_len:
            sf_users.append(u_sf_users)
            sf_u_u_items = []
            for uui in u_sf_users_items:
                if len(uui) <= truncate_len:
                    sf_u_u_items.append(uui)
                else:
                    sf_u_u_items.append(random.sample(uui, truncate_len))
            sf_users_items.append(sf_u_u_items)
        else:
            sample_index = random.sample(list(range(len(u_sf_users))), truncate_len)
            sf_users.append([u_sf_users[si] for si in sample_index])

            u_sf_users_items_tr = [u_sf_users_items[si] for si in sample_index]
            sf_u_u_items = []
            for uui in u_sf_users_items_tr:
                if len(uui) <= truncate_len:
                    sf_u_u_items.append(uui)
                else:
                    sf_u_u_items.append(random.sample(uui, truncate_len))
            sf_users_items.append(sf_u_u_items)
        sf_users_len.append(min(len(u_sf_users), truncate_len))

        # user-users and user-users-items
        if len(u_users_u) < truncate_len:
            tmp_users = [item for item in u_users_u]
            tmp_users.append(uid)
            u_users.append(tmp_users)
            u_u_items = [] 
            for uui in u_users_items_u:
                if len(uui) < truncate_len:
                    u_u_items.append(uui)
                else:
                    u_u_items.append(random.sample(uui, truncate_len))
            # self -loop
            u_u_items.append(u_items[-1])
            u_users_items.append(u_u_items)

        else:
            sample_index = random.sample(list(range(len(u_users_u))), truncate_len-1)
            tmp_users = [u_users_u[si] for si in sample_index]
            tmp_users.append(uid)
            u_users.append(tmp_users)

            u_users_items_u_tr = [u_users_items_u[si] for si in sample_index]
            u_u_items = [] 
            for uui in u_users_items_u_tr:
                if len(uui) < truncate_len:
                    u_u_items.append(uui)
                else:
                    u_u_items.append(random.sample(uui, truncate_len))
            u_u_items.append(u_items[-1])
            u_users_items.append(u_u_items)

        u_users_len.append(min(len(u_users_u)+1, truncate_len))

        # item-users
        if len(i_users_i) <= truncate_len:
            i_users.append(i_users_i)
        else:
            i_users.append(random.sample(i_users_i, truncate_len))
        i_users_len.append(min(len(i_users_i), truncate_len))


        #i_item_friends
        if len(i_items_friends) <= truncate_len_i:
            i_friends.append(i_items_friends)
            i_u_items = []
            for iui in i_items_users:
                if len(iui) <= truncate_len:
                    i_u_items.append(iui)
                else:
                    i_u_items.append(random.sample(iui, truncate_len))
            if_items_users.append(i_u_items)

        else:
            #sample_index = random.sample(list(range(len(i_items_friends))), truncate_len_i)
            sample_index = list(range(truncate_len_i))
            i_friends.append([i_items_friends[si] for si in sample_index])

            if_items_users_tr = [i_items_users[si] for si in sample_index]
            i_u_items = []
            for iui in if_items_users_tr:
                if len(iui) <= truncate_len:
                    i_u_items.append(iui)
                else:
                    i_u_items.append(random.sample(iui, truncate_len))
            if_items_users.append(i_u_items)
        i_friends_len.append(min(len(i_items_friends), truncate_len_i))

        count += 1
    batch_size = len(batch_data)

    # padding
    u_items_maxlen = max(u_items_len)
    u_users_maxlen = max(u_users_len)
    i_users_maxlen = max(i_users_len)
    sf_user_maxlen = max(sf_users_len)
    i_friends_maxlen = max(i_friends_len)
    neg_maxlen = len(neg_transh_list)


    neg_pad = torch.tensor(neg_transh_list, dtype=torch.long)
    pos_pad = torch.tensor(pos_transh_list, dtype=torch.long)
    

    u_item_pad = torch.zeros([batch_size, u_items_maxlen, 2], dtype=torch.long)
    for i, ui in enumerate(u_items):
        u_item_pad[i, :len(ui), :] = torch.LongTensor(ui)
    
    u_user_pad = torch.zeros([batch_size, u_users_maxlen], dtype=torch.long)
    for i, uu in enumerate(u_users):
        u_user_pad[i, :len(uu)] = torch.LongTensor(uu)

    # 包含了对每个sf user的紧密度关系，在维度1上
    sf_user_pad = torch.zeros([batch_size, sf_user_maxlen,2], dtype=torch.long)
    for i, usf in enumerate(sf_users):
        sf_user_pad[i, :len(usf),:] = torch.LongTensor(usf)

    sf_user_item_pad = torch.zeros([batch_size, sf_user_maxlen, truncate_len, 2], dtype=torch.long)
    for i, uu_items in enumerate(sf_users_items):
        for j, ui in enumerate(uu_items):
            sf_user_item_pad[i,j, :len(ui), :] = torch.LongTensor(ui)

    u_user_item_pad = torch.zeros([batch_size, u_users_maxlen, u_items_maxlen, 2], dtype=torch.long)
    for i, uu_items in enumerate(u_users_items):
        for j, ui in enumerate(uu_items):
            u_user_item_pad[i, j, :len(ui), :] = torch.LongTensor(ui)

    i_user_pad = torch.zeros([batch_size, i_users_maxlen, 2], dtype=torch.long)
    for i, iu in enumerate(i_users):
        i_user_pad[i, :len(iu), :] = torch.LongTensor(iu)

    i_friends_pad = torch.zeros([batch_size, i_friends_maxlen, 2], dtype=torch.long)
    for i, i_f in enumerate(i_friends):
        i_friends_pad[i,:len(i_f), :] = torch.LongTensor(i_f)

    if_items_users_pad = torch.zeros([batch_size, i_friends_maxlen, i_users_maxlen,2], dtype=torch.long)
    for i, iui in enumerate(if_items_users):
        for j, ui in enumerate(iui):
            if_items_users_pad[i, j, :len(ui), :] = torch.LongTensor(ui)

    return torch.LongTensor(uids), torch.LongTensor(iids), torch.FloatTensor(labels), \
            u_item_pad, u_user_pad, u_user_item_pad, i_user_pad, sf_user_pad, sf_user_item_pad, i_friends_pad, if_items_users_pad, pos_pad, neg_pad
