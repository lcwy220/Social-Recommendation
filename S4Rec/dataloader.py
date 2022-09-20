import numpy as np
import random
import torch
from torch.utils.data import Dataset

class GRDataset(Dataset):
	def __init__(self, data, u_items_list, u_users_list, u_users_items_list, i_users_list, sf_list, sf_user_item_list, i_items_friend_list, if_item_users_list, neg_transh_list):
		self.data = data
		self.u_items_list = u_items_list
		self.u_users_list = u_users_list
		self.u_users_items_list = u_users_items_list
		self.i_users_list = i_users_list
		self.sf_list = sf_list
		self.sf_user_item_list = sf_user_item_list
		self.i_items_friend_list = i_items_friend_list
		self.if_item_users_list = if_item_users_list
		self.neg_transh_list = neg_transh_list


	def __getitem__(self, index):
		uid = self.data[index][0]
		iid = self.data[index][1]
		label = self.data[index][2]
		neg_train_list = self.neg_transh_list[index] if self.neg_transh_list != [] else [(0,0,0)]
		u_items = self.u_items_list[uid]
		u_users = self.u_users_list[uid]
		u_users_items = self.u_users_items_list[uid]
		i_users = self.i_users_list[iid]
		sf_users = self.sf_list[uid]
		sf_user_items = self.sf_user_item_list[uid]
		if_item_friends = self.i_items_friend_list[iid]
		if_items_users = self.if_item_users_list[iid]

		return (uid, iid, label), u_items, u_users, u_users_items, i_users, sf_users, sf_user_items, if_item_friends, if_items_users, neg_train_list

	def __len__(self):
		return len(self.data)
