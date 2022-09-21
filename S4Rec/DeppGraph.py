from torch import nn
import torch.nn.functional as F
import torch
from tranh import TransH

class _MultiLayerPercep(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_MultiLayerPercep, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim // 2, output_dim, bias=True),
        )

    def forward(self, x):
        return self.mlp(x)


class _Aggregation(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_Aggregation, self).__init__()
        self.aggre = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.aggre(x)


class _UserModel(nn.Module):
    ''' User modeling to learn user latent factors.
    User modeling leverages two types aggregation: item aggregation and social aggregation
    '''
    def __init__(self, emb_dim, user_emb, item_emb, rate_emb):
        super(_UserModel, self).__init__()
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.rate_emb = rate_emb
        self.emb_dim = emb_dim

        self.w1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w3 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w4 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w5 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w6 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w7 = nn.Linear(self.emb_dim, self.emb_dim)


        self.g_v = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        self.g_sf = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)

        self.user_items_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_items = _Aggregation(self.emb_dim, self.emb_dim)

        self.user_items_att_s1 = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_items_s1 = _Aggregation(self.emb_dim, self.emb_dim)
        self.user_users_att_s2 = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_neigbors_s2 = _Aggregation(self.emb_dim, self.emb_dim)

        self.u_user_users_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.u_aggre_neigbors = _Aggregation(self.emb_dim, self.emb_dim)

        self.sf_user_users_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.sf_aggre_neigbors = _Aggregation(self.emb_dim, self.emb_dim)

        self.user_items_att_sf1 = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_items_sf1 = _Aggregation(self.emb_dim, self.emb_dim)
        self.sf_users_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_sf_neigbors = _Aggregation(self.emb_dim, self.emb_dim)

        self.combine_mlp = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(5 * self.emb_dim, 3*self.emb_dim, bias = True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(3*self.emb_dim, self.emb_dim, bias = True),
            nn.ReLU()
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # used for preventing zero div error when calculating softmax score
        self.eps = 1e-10

    def forward(self, uids, iids, u_item_pad, u_user_pad, u_user_item_pad, sf_user_pad, sf_user_item_pad):
        # item aggregation
        q_a = self.item_emb(u_item_pad[:,:,0])   # B x maxi_len x emb_dim
        mask_u = torch.where(u_item_pad[:,:,0] > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))   # B x maxi_len
        u_item_er = self.rate_emb(u_item_pad[:, :, 1])  # B x maxi_len x emb_dim
        x_ia = self.g_v(torch.cat([q_a, u_item_er], dim=2).view(-1, 2 * self.emb_dim)).view(q_a.size())  # B x maxi_len x emb_dim
        p_i = mask_u.unsqueeze(2).expand_as(q_a) * self.user_emb(uids).unsqueeze(1).expand_as(q_a)  # B x maxi_len x emb_dim

        # alpha = self.user_items_att(torch.cat([x_ia, p_i], dim = 2)) 就够了，B, maxi_len,1
        # 计算attention的另一种方法，之前是pygat中增加大矩阵
        alpha = self.user_items_att(torch.cat([self.w1(x_ia), self.w1(p_i)], dim = 2).view(-1, 2 * self.emb_dim)).view(mask_u.size()) # B x maxi_len
        alpha = torch.exp(alpha) * mask_u
        alpha = alpha / (torch.sum(alpha, 1).unsqueeze(1).expand_as(alpha) + self.eps)

        h_iI = self.aggre_items(torch.sum(alpha.unsqueeze(2).expand_as(x_ia) * self.w1(x_ia), 1))     # B x emb_dim
        h_iI = F.dropout(h_iI, 0.5, training=self.training)

        # sf_user
        q_a_f = self.item_emb(sf_user_item_pad[:, :, :, 0])  # B x maxu_len x maxi_len x emb_dim
        mask_sf = torch.where(sf_user_item_pad[:, :, :, 0] > 0, torch.tensor([1.], device=self.device),torch.tensor([0.], device=self.device))  # B x maxu_len x maxi_len
        sf_user_item_er = self.rate_emb(sf_user_item_pad[:, :, :, 1])  # B x maxu_len x maxi_len x emb_dim
        x_ia_sf = self.g_v(torch.cat([q_a_f, sf_user_item_er], dim=3).view(-1, 2 * self.emb_dim)).view( q_a_f.size())  # B x maxu_len x maxi_len x emb_dim

        p_i_sf = mask_sf.unsqueeze(3).expand_as(q_a_f) * self.user_emb(sf_user_pad[:,:,0]).unsqueeze(2).expand_as(q_a_f)  # B x maxu_len x maxi_len x emb_dim

        alpha_i_sf = self.user_items_att_sf1(torch.cat([self.w2(x_ia_sf), self.w2(p_i_sf)], dim=3).view(-1, 2 * self.emb_dim)).view(mask_sf.size())  # B x maxu_len x maxi_len
        alpha_i_sf = torch.exp(alpha_i_sf) * mask_sf
        alpha_i_sf = alpha_i_sf / (torch.sum(alpha_i_sf, 2).unsqueeze(2).expand_as(alpha_i_sf) + self.eps)

        h_sfI_temp = torch.sum(alpha_i_sf.unsqueeze(3).expand_as(x_ia_sf) * self.w2(x_ia_sf), 2)  # B x maxu_len x emb_dim
        h_sfI = self.aggre_items_sf1(h_sfI_temp.view(-1, self.emb_dim)).view(h_sfI_temp.size())  # B x maxu_len x emb_dim
        h_sfI = F.dropout(h_sfI, 0.5, training=self.training)


        mask_u_f = torch.where(sf_user_pad[:,:,0] > 0, torch.tensor([1.0], device=self.device), torch.tensor([0.], device=self.device))

        ##calculate attention score
        p_u_sf = mask_u_f.unsqueeze(2).expand_as(h_sfI) * self.user_emb(uids).unsqueeze(1).expand_as(h_sfI)
        #p_u_sf = self.user_emb(sf_user_pad[:,:,0])
        beta_sf = self.sf_users_att(torch.cat([self.w3(h_sfI), self.w3(p_u_sf)], dim=2).view(-1, 2*self.emb_dim)).view(mask_u_f.size())
        beta_sf = torch.exp(beta_sf) * mask_u_f
        beta_sf = beta_sf / (torch.sum(beta_sf, 1).unsqueeze(1).expand_as(beta_sf) + self.eps)

        h_i_sf = self.aggre_sf_neigbors(torch.sum(beta_sf.unsqueeze(2).expand_as(h_sfI) * self.w3(h_sfI), 1))
        h_i_sf = F.dropout(h_i_sf, p=0.5, training=self.training)


        # social aggregation
        q_a_s = self.item_emb(u_user_item_pad[:,:,:,0])   # B x maxu_len x maxi_len x emb_dim
        mask_s = torch.where(u_user_item_pad[:,:,:,0] > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))  # B x maxu_len x maxi_len
        p_i_s = mask_s.unsqueeze(3).expand_as(q_a_s) * self.user_emb(u_user_pad).unsqueeze(2).expand_as(q_a_s)  # B x maxu_len x maxi_len x emb_dim
        u_user_item_er = self.rate_emb(u_user_item_pad[:, :, :, 1])  # B x maxu_len x maxi_len x emb_dim
        x_ia_s = self.g_v(torch.cat([q_a_s, u_user_item_er], dim=3).view(-1, 2 * self.emb_dim)).view(q_a_s.size())  # B x maxu_len x maxi_len x emb_dim

        alpha_s = self.user_items_att_s1(torch.cat([self.w4(x_ia_s), self.w4(p_i_s)], dim = 3).view(-1, 2 * self.emb_dim)).view(mask_s.size())    # B x maxu_len x maxi_len
        alpha_s = torch.exp(alpha_s) * mask_s
        alpha_s = alpha_s / (torch.sum(alpha_s, 2).unsqueeze(2).expand_as(alpha_s) + self.eps)


        h_oI_temp = torch.sum(alpha_s.unsqueeze(3).expand_as(x_ia_s) * self.w4(x_ia_s), 2)    # B x maxu_len x emb_dim
        h_oI = self.aggre_items_s1(h_oI_temp.view(-1, self.emb_dim)).view(h_oI_temp.size())  # B x maxu_len x emb_dim
        h_oI = F.dropout(h_oI, p=0.5, training=self.training)

        ## calculate attention scores in social aggregation
        mask_su = torch.where(u_user_pad > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))

        beta = self.user_users_att_s2(torch.cat([self.w5(h_oI), self.w5(self.user_emb(u_user_pad))], dim = 2).view(-1, 2 * self.emb_dim)).view(u_user_pad.size())
        beta = torch.exp(beta) * mask_su
        beta = beta / (torch.sum(beta, 1).unsqueeze(1).expand_as(beta) + self.eps)
        h_iS = self.aggre_neigbors_s2(torch.sum(beta.unsqueeze(2).expand_as(h_oI) * self.w5(h_oI), 1))     # B x emb_dim
        h_iS = F.dropout(h_iS, p=0.5, training=self.training)



        su = self.user_emb(u_user_pad)
        p_uf = mask_su.unsqueeze(2).expand_as(su) * self.user_emb(uids).unsqueeze(1).expand_as(su)
        alpha_su = self.u_user_users_att(torch.cat([self.w6(su), self.w6(p_uf)], dim=2)).view(mask_su.size())
        #alpha_su = torch.matmul(su, F.tanh(self.user_users_att(ti_emb)).unsqueeze(2)).squeeze()
        alpha_su = torch.exp(alpha_su) * mask_su
        alpha_su = alpha_su / (torch.sum(alpha_su, 1).unsqueeze(1).expand_as(alpha_su) + self.eps)

        h_su = self.u_aggre_neigbors(torch.sum(alpha_su.unsqueeze(2).expand_as(su) * self.w6(su), 1))
        h_su = F.dropout(h_su, p=0.5, training=self.training)


        sf = self.user_emb(sf_user_pad[:,:,0])
        p_sf = mask_u_f.unsqueeze(2).expand_as(sf) * self.user_emb(uids).unsqueeze(1).expand_as(sf)
        alpha_sf = self.sf_user_users_att(torch.cat([self.w7(sf), self.w7(p_sf)], dim=2)).view(mask_u_f.size())
        alpha_sf = torch.exp(alpha_sf) * mask_u_f
        alpha_sf = alpha_sf / (torch.sum(alpha_sf, 1).unsqueeze(1).expand_as(alpha_sf) + self.eps)

        h_sf = self.sf_aggre_neigbors(torch.sum(alpha_sf.unsqueeze(2).expand_as(sf) * self.w7(sf), 1))
        h_sf = F.dropout(h_sf, p=0.5, training=self.training)


        ## learning user latent factor
        h =  self.combine_mlp(torch.cat([h_iI, h_iS, h_i_sf, h_su, h_sf], dim = 1))

        return h


class _ItemModel(nn.Module):
    '''Item modeling to learn item latent factors.
    '''
    def __init__(self, emb_dim, user_emb, item_emb, rate_emb):
        super(_ItemModel, self).__init__()
        self.emb_dim = emb_dim
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.rate_emb = rate_emb

        self.w1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w3 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w4 = nn.Linear(self.emb_dim, self.emb_dim)

        self.g_u = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        self.g_v = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)

        self.item_users_att_i = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_users_i = _Aggregation(self.emb_dim, self.emb_dim)

        self.item_users_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_users = _Aggregation(self.emb_dim, self.emb_dim)

        self.i_friends_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_i_friends = _Aggregation(self.emb_dim, self.emb_dim)

        self.if_friends_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_if_friends = _Aggregation(self.emb_dim, self.emb_dim)

        self.combine_mlp = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(3* self.emb_dim, 2*self.emb_dim, bias = True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2*self.emb_dim, self.emb_dim, bias = True),
            nn.ReLU()
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # used for preventing zero div error when calculating softmax score
        self.eps = 1e-10

    def forward(self, uids, iids, i_user_pad, i_friends_pad, i_friends_user_pad):
        # user aggregation
        p_t = self.user_emb(i_user_pad[:,:,0])
        mask_i = torch.where(i_user_pad[:,:,0] > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))
        i_user_er = self.rate_emb(i_user_pad[:,:,1])
        f_jt = self.g_u(torch.cat([p_t, i_user_er], dim = 2).view(-1, 2 * self.emb_dim)).view(p_t.size())
        
        # calculate attention scores in user aggregation
        q_j = mask_i.unsqueeze(2).expand_as(f_jt) * self.item_emb(iids).unsqueeze(1).expand_as(f_jt)
        
        miu = self.item_users_att_i(torch.cat([self.w1(f_jt), self.w1(q_j)], dim = 2).view(-1, 2 * self.emb_dim)).view(mask_i.size())
        miu = torch.exp(miu) * mask_i
        miu = miu / (torch.sum(miu, 1).unsqueeze(1).expand_as(miu) + self.eps)
        z_j = self.aggre_users_i(torch.sum(miu.unsqueeze(2).expand_as(f_jt) * self.w1(f_jt), 1))
        z_j = F.dropout(z_j, p=0.5, training=self.training)

        # item aggregation
        q_a = self.item_emb(i_friends_pad[:,:,0])   # B x maxi_len x emb_dim
        mask_u = torch.where(i_friends_pad[:,:,0] > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))   # B x maxi_len
        ## calculate attention scores in item aggregation
        # x_ia & p_i cat时候，保证pad位置不能被cat上
        p_i = mask_u.unsqueeze(2).expand_as(q_a) * self.item_emb(iids).unsqueeze(1).expand_as(q_a)  # B x maxi_len x emb_dim
        # alpha = self.user_items_att(torch.cat([x_ia, p_i], dim = 2)) 就够了，B, maxi_len,1
        # 计算attention的另一种方法，之前是pygat中增加大矩阵
        alpha = self.i_friends_att(torch.cat([self.w2(q_a), self.w2(p_i)], dim = 2).view(-1, 2 * self.emb_dim)).view(mask_u.size()) # B x maxi_len
        alpha = torch.exp(alpha) * mask_u
        alpha = alpha / (torch.sum(alpha, 1).unsqueeze(1).expand_as(alpha) + self.eps)

        z_if = self.aggre_i_friends(torch.sum(alpha.unsqueeze(2).expand_as(q_a) * self.w2(q_a), 1))     # B x emb_dim
        z_if = F.dropout(z_if, p=0.5, training=self.training)

        # item users friends aggregation
        q_a_s = self.user_emb(i_friends_user_pad[:, :, :, 0])  # B x maxi_len x maxu_len x emb_dim
        mask_s = torch.where(i_friends_user_pad[:, :, :, 0] > 0, torch.tensor([1.], device=self.device),torch.tensor([0.], device=self.device))  # B x maxu_len x maxi_len
        u_user_item_er = self.rate_emb(i_friends_user_pad[:, :, :, 1])  # B x maxu_len x maxi_len x emb_dim
        x_ia_s = self.g_u(torch.cat([q_a_s, u_user_item_er], dim=3).view(-1, 2 * self.emb_dim)).view(q_a_s.size())  # B x maxu_len x maxi_len x emb_dim

        p_i_s = mask_s.unsqueeze(3).expand_as(x_ia_s) * q_a.unsqueeze(2).expand_as(x_ia_s)  # B x maxu_len x maxi_len x emb_dim
        alpha_s = self.item_users_att(torch.cat([self.w3(x_ia_s), self.w3(p_i_s)], dim=3).view(-1, 2 * self.emb_dim)).view(mask_s.size())  # B x maxu_len x maxi_len
        alpha_s = torch.exp(alpha_s) * mask_s
        alpha_s = alpha_s / (torch.sum(alpha_s, 2).unsqueeze(2).expand_as(alpha_s) + self.eps)

        h_oI_temp = torch.sum(alpha_s.unsqueeze(3).expand_as(x_ia_s) * self.w3(x_ia_s), 2)  # B x maxu_len x emb_dim
        h_oI = self.aggre_users(h_oI_temp.view(-1, self.emb_dim)).view(h_oI_temp.size())  # B x maxu_len x emb_dim
        h_oI = F.dropout(h_oI, p=0.5, training=self.training)

        ## calculate attention scores in social aggregation
        alpha_i = self.if_friends_att(torch.cat([self.w4(h_oI), self.w4(p_i)], dim = 2).view(-1, 2 * self.emb_dim)).view(mask_u.size()) # B x maxi_len
        alpha_i = torch.exp(alpha_i) * mask_u
        alpha_i = alpha_i / (torch.sum(alpha_i, 1).unsqueeze(1).expand_as(alpha_i) + self.eps)

        z_uf = self.aggre_if_friends(torch.sum(alpha_i.unsqueeze(2).expand_as(h_oI) * h_oI, 1))     # B x emb_dim
        z_uf = F.dropout(z_uf, p=0.5, training=self.training)


        z = self.combine_mlp(torch.cat([z_if, z_j, z_uf], dim=1))

        return z


class DeppGraph(nn.Module):
    '''GraphRec model proposed in the paper Graph neural network for social recommendation 

    Args:
        number_users: the number of users in the dataset.
        number_items: the number of items in the dataset.
        num_rate_levels: the number of rate levels in the dataset.
        emb_dim: the dimension of user and item embedding (default = 64).

    '''
    def __init__(self,  user_emb, item_emb, rate_emb, num_users, num_items, num_rate_levels, emb_dim = 64):
        super(DeppGraph, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_rate_levels = num_rate_levels
        self.emb_dim = emb_dim
        self.user_emb = nn.Embedding(self.num_users, self.emb_dim, padding_idx = 0)
        self.user_emb.weight.data.copy_(torch.from_numpy(user_emb))
        self.item_emb = nn.Embedding(self.num_items, self.emb_dim, padding_idx = 0)
        self.item_emb.weight.data.copy_(torch.from_numpy(item_emb))
        self.rate_emb = nn.Embedding(self.num_rate_levels, self.emb_dim, padding_idx = 0)
        self.rate_emb.weight.data.copy_(torch.from_numpy(rate_emb))

        self.user_model = _UserModel(self.emb_dim, self.user_emb, self.item_emb, self.rate_emb)

        self.item_model = _ItemModel(self.emb_dim, self.user_emb, self.item_emb, self.rate_emb)

        self.tranh = TransH(self.user_emb, self.item_emb, self.item_emb, num_users, num_items, 5, self.emb_dim)
        
        self.rate_pred = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(3* self.emb_dim, 2*self.emb_dim, bias = True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2*self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.emb_dim, 1)
        )





    def forward(self, uids, iids, u_item_pad, u_user_pad, u_user_item_pad, i_user_pad, sf_user_pad, sf_user_item_pad, i_friends_pad, i_friends_user_pad, pos_list, neg_list, train_state=True):
        '''
        Args:
            uids: the user id sequences.
            iids: the item id sequences.
            u_item_pad: the padded user-item graph.
            u_user_pad: the padded user-user graph.
            u_user_item_pad: the padded user-user-item graph.
            i_user_pad: the padded item-user graph.
            sf_user_pad: the padded user-user share common fans graph

        Shapes:
            uids: (B).
            iids: (B).
            u_item_pad: (B, ItemSeqMaxLen, 2).
            u_user_pad: (B, UserSeqMaxLen).
            u_user_item_pad: (B, UserSeqMaxLen, ItemSeqMaxLen, 2).
            i_user_pad: (B, UserSeqMaxLen, 2).
            sf_user_pad: (B, SfuserSeqMaxLen, 2)

        Returns:
            the predicted rate scores of the user to the item.
        '''

        h = self.user_model(uids, iids, u_item_pad, u_user_pad, u_user_item_pad, sf_user_pad, sf_user_item_pad)
        z = self.item_model(uids, iids, i_user_pad, i_friends_pad, i_friends_user_pad)

        tranh_loss = 0
        if train_state:
            self.tranh.normalizeEmbedding()
            tranh_loss = self.tranh(pos_list, neg_list)

        r_ij = self.rate_pred(torch.cat([h,z,h*z ], dim = 1))

        return r_ij, tranh_loss
