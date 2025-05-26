import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple

from problems.petra.petra import PetraEnv
from utils.tensor_functions import compute_in_batches

from nets.graph_encoder import GraphAttentionEncoder, MultiHeadAttention, MultiHeadAttentionLayer
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many
import copy
import random


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        # if torch.is_tensor(key) or isinstance(key, slice):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )
        # return super(AttentionModelFixed, self).__getitem__(key)


# 2D-Ptr
class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 obj,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 opts=None):
        super(AttentionModel, self).__init__()

        assert opts is not None, "opts should not be None"
        self.opts = opts
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim # deprecated
        self.obj = obj
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.is_hcvrp = problem.NAME == 'hcvrp'
        self.is_petra = problem.NAME == 'petra'
        self.feed_forward_hidden = 4*embedding_dim

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.depot_token = nn.Parameter(torch.randn(embedding_dim)) # depot token
        node_embedding_dim = 5 if self.is_petra else 3
        self.init_embed = nn.Linear(node_embedding_dim, embedding_dim)  # embed linear in customer encoder
        self.node_encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization,
            feed_forward_hidden=self.feed_forward_hidden
        )
        veh_encoder_dim = 5 if self.is_petra else 4
        self.veh_encoder_mlp = nn.Sequential(
            nn.Linear(veh_encoder_dim, embedding_dim * veh_encoder_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim * veh_encoder_dim, embedding_dim)
        )
        self.veh_encoder_self_attention = MultiHeadAttention(n_heads=n_heads, input_dim=embedding_dim, embed_dim=embedding_dim)
        self.veh_encoder_ca_node_linear_kv = nn.Linear(embedding_dim,2*embedding_dim) # veh-node-cross-attn w_k,w_v
        self.veh_encoder_ca_veh_linear_q = nn.Linear(embedding_dim, embedding_dim) # veh-node-cross-attn w_q
        self.veh_encoder_ca_linear_o = nn.Linear(embedding_dim, embedding_dim) # veh-node-cross-attn w_o
        self.veh_encoder_w = nn.Linear(2*embedding_dim,embedding_dim)
        assert embedding_dim % n_heads == 0
        self.fulfilment_mlp = nn.Sequential(
            nn.Linear(2*embedding_dim, embedding_dim, bias=True),
            nn.ReLU(),
        )
        self.fulfilment_mlp_alpha = nn.Sequential(
            nn.Linear(embedding_dim, 1, bias=True),
            # nn.ReLU(),
            # nn.Softplus(beta=0.5, threshold=20),
        )  # used to calculate the fulfillment of the selected node
        # nn.init.constant_(self.fulfilment_mlp_alpha[0].bias, 1)
        self.fulfilment_mlp_beta = nn.Sequential(
            nn.Linear(embedding_dim, 1, bias=True),
            # nn.ReLU(),
            # nn.Softplus(beta=0.5, threshold=20),
        )
        # nn.init.constant_(self.fulfilment_mlp_beta[0].bias, 1)

    def pre_calculate_node(self,input):
        nhead = self.n_heads
        env = HcvrpEnv(input, scale=(1, 40, 1)) if self.is_hcvrp else PetraEnv(input, opts=self.opts)
        # embed node (depot and customer)
        node_embedding = self.init_embed(env.get_all_node_state())
        # add depot token
        node_embedding[:, 0] = node_embedding[:, 0] + self.depot_token
        node_embedding = self.node_encoder(node_embedding)[0]
        bs,N,d = node_embedding.size()
        # pre-calculate the K,V of the cross-attention in vehcle encoder, avoid double calculation
        kv = self.veh_encoder_ca_node_linear_kv(node_embedding).reshape(bs,N,nhead,-1).transpose(1,2) # bs,nhead,N,d_k*2
        k,v = torch.chunk(kv,2,-1) # bs,nhead,n,d_k,bs,nhead,n,d_k,
        return input, node_embedding, (k, v)
    def veh_encoder_cross_attention(self,veh_em,node_kv,mask=None):
        '''
        :param veh_em:
        :param node_kv:
        :param action_mask: bs,M,N
        :return:
        '''
        bs,M,d = veh_em.size()
        nhead = self.n_heads
        k,v = node_kv
        q = self.veh_encoder_ca_veh_linear_q(veh_em).reshape(bs,M,nhead,-1).transpose(1,2) # bs,nhead,M,d_k
        attn = q @ k.transpose(-1,-2)/np.sqrt(q.size(-1)) # bs,nhead,M,N
        if mask is not None:
            attn[mask.unsqueeze(1).expand(attn.size())]=-math.inf
        attn = attn.softmax(-1) #bs,nhead,M,N
        out = attn @ v # bs,nhead,M,d_k
        out = self.veh_encoder_ca_linear_o(out.transpose(1,2).reshape(bs,M,-1)) # bs,M,d
        return out

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, actions=None, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        # norm of self.init_embed.weight
        input, node_embedding,node_kv = self.pre_calculate_node(input)
        # input, node_embedding, veh_embedding = self.initial_em(input)
        ll, pi, veh_list, fulfilments, cost, infos = self._inner(input, node_embedding, node_kv, actions=actions)
        if return_pi:
            return cost, ll, pi, veh_list, fulfilments, infos
        return cost, ll

    def _inner(self,input ,node_embeddings,node_kv,actions=None):
        env = HcvrpEnv(input, scale=(1, 40, 1)) if self.is_hcvrp else PetraEnv(input, opts=self.opts)
        ll,pi,veh_list,fulfilments,infos=[],[],[],[],[]
        step = 0
        while not env.all_finished():
            # DEBUG
            if (~env.get_action_mask()).any(-1).any(-1).all() == False:
                env.finished()
                env.get_action_mask()

            # update vehicle embeddings
            veh_embeddings = self.veh_encoder(node_embeddings,node_kv,env)
            # select action
            veh, node, fulfilment, log_p, info = self.decoder(veh_embeddings, node_embeddings, mask=env.get_action_mask(), action=actions[step] if actions is not None else None)
            # update env
            drop = env.update(veh,node,fulfilment)  # --> lets talk about this with fulfilment
            info.append(drop.detach())

            veh_list.append(veh)
            pi.append(node)
            ll.append(log_p)
            fulfilments.append(fulfilment)
            infos.append(info)
            step += 1

        # get the final cost
        cost, cost_info = env.get_cost(self.obj)
        ll = torch.stack(ll, 1)  # bs,step
        pi = torch.stack(pi, 1)  # bs,step
        veh_list = torch.stack(veh_list, 1)  # bs,step
        fulfilments = torch.stack(fulfilments, 1)
        infos_ = torch.stack([torch.stack(inner, dim=0) for inner in infos], dim=0).permute(1, 2, 0)
        return ll.sum(1),pi,veh_list,fulfilments,cost,{'inner_info': infos_,'cost_info': cost_info}

    def decoder(self,q_em,k_em,mask=None,action=None):
        '''
        :param q_em: Q: bs,m,d
        :param k_em: K: bs,n,d
        :param mask: bs,m,n
        :return: selected index,log_pro
        '''
        bs,m,d = q_em.size()
        _,n,_ = k_em.size()
        bs_index = torch.arange(bs,device=q_em.device)
        logits = (q_em @ k_em.transpose(1, 2) / np.sqrt(d))  # bs,m,n
        if self.tanh_clipping > 0:  # 10
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:  # True
            if mask is not None:
                logits[mask] = -math.inf
        logits = logits.reshape(bs,-1) # bs,M*N
        p = logits.softmax(1) # bs,M*N
        if action is None:
            if self.decode_type=='greedy':
                selected = p.max(1)[1] # bs
            else:
                selected = p.multinomial(1).squeeze(1)
        else:
            selected = action['veh'] * n + action['node']
        log_p = p[bs_index,selected].log()
        veh,node = selected//n,selected%n

        # fulfillment with beta distribution
        veh_em = q_em[bs_index,veh]
        node_em = k_em[bs_index,node]
        veh_node_em = torch.cat([veh_em,node_em],dim=-1)
        # Add small epsilon to ensure parameters are strictly gt 1
        eps = 1e-6
        hidden = self.fulfilment_mlp(veh_node_em)
        alpha = (5*torch.sigmoid(self.fulfilment_mlp_alpha(hidden)) + 1).squeeze(1)
        beta = (5*torch.sigmoid(self.fulfilment_mlp_beta(hidden)) + 1).squeeze(1)
        p = torch.distributions.Beta(alpha, beta)
        # sample from the beta distribution
        if action is None:
            if self.decode_type=='greedy':
                fulfilment = p.mean # bs
            else:
                fulfilment = p.sample() # bs
            fulfilment = torch.clamp(fulfilment, min=eps, max=1.0-eps)        
        else:
            fulfilment = action['fulfilment']
        # Clamp fulfilment to prevent numerical issues at the boundaries
        lop_p_fulfilment = p.log_prob(fulfilment) # bs
        lop_p_fulfilment = torch.clamp(lop_p_fulfilment, min=-1e8, max=1e8)
        fulfilment = fulfilment

        log_p_ = log_p + lop_p_fulfilment
        info = [lop_p_fulfilment.detach(), alpha.detach(), beta.detach()]
        return veh,node,fulfilment,log_p_,info
    
    def veh_encoder(self,node_embeddings,node_kv,env):
        veh_embeddings = self.veh_encoder_mlp(env.get_all_veh_state())
        bs, N, d = node_embeddings.size()
        bs, M, d = veh_embeddings.size()
        bs_index = torch.arange(bs, device=node_embeddings.device)
        veh_node_em = node_embeddings[bs_index.unsqueeze(-1),env.veh_cur_node.clone()] # PE:bs,M,d
        veh_embeddings = self.veh_encoder_w(torch.cat([veh_node_em,veh_embeddings],dim=-1))
        mask = env.visited.clone()
        # depot will not be masked
        mask[:,0] = False
        mask = mask.unsqueeze(1).expand(bs,M,N)
        veh_embeddings = veh_embeddings + self.veh_encoder_self_attention(veh_embeddings)
        veh_embeddings = veh_embeddings + self.veh_encoder_cross_attention(veh_embeddings,node_kv,mask)
        return veh_embeddings

    def sample_many(self, input, batch_rep=1, iter_rep=1):

        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            None,
            self.pre_calculate_node(input),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep
        )

