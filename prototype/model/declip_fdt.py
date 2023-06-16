
#import ipdb
import torch
import torch.nn.functional as F
from torch import nn
import math
from .image_encoder.visual_transformer import visual_transformer_B32, visual_transformer_B16
from .text_encoder.text_transformer import text_transformers
from .utils.nnclr_modules import NNMemoryBankModule

import prototype.linklink as link
from random import choice
from .sparsemax import Sparsemax
from .clip import CLIP


BN = None


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=1024, num_layers=3):
        super(projection_MLP, self).__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out-
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d.
        This MLP has 3 layers.
        '''
        self.num_layers = num_layers

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.bn1 = BN(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        # self.bn2 = BN(hidden_dim)

        if self.num_layers == 3:
            self.relu2 = nn.ReLU(inplace=True)
            self.linear3 = nn.Linear(hidden_dim, out_dim)
            self.bn3 = nn.BatchNorm1d(hidden_dim)
            # self.bn3 = BN(hidden_dim)

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        # b, _ = x.shape
        # layer 1
        x = self.linear1(x)
        # x.reshape(b, self.hidden_dim, 1)
        x = self.bn1(x)
        x = self.relu1(x)
        # x.reshape(b, self.hidden_dim)

        # layer 2
        x = self.linear2(x)
        # x.reshape(b, self.hidden_dim, 1)
        x = self.bn2(x)


        if self.num_layers == 3:
            x = self.relu2(x)
            # x.reshape(b, self.hidden_dim)
            # layer 3
            x = self.linear3(x)
            # x.reshape(b, self.out_dim, 1)
            x = self.bn3(x)
            # x.reshape(b, self.out_dim)

        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=1024): # bottleneck structure
        super(prediction_MLP, self).__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers.
        The dimension of h’s input and output (z and p) is d = 2048,
        and h’s hidden layer’s dimension is 512, making h a
        bottleneck structure (ablation in supplement).
        '''
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.bn1 = BN(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing.
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        b, _ = x.shape

        # layer 1
        x = self.linear1(x)
        # x.reshape(b, self.hidden_dim, 1)
        x = self.bn1(x)
        x = self.relu1(x)
        # x.reshape(b, self.hidden_dim)

        x = self.layer2(x)
        return x

class Query_model(nn.Module):
    def __init__(self, ft_dim, sd_dim, temperature=1, att_func_type='softmax', pool_type='sum'):
        #d_model, the dimension of features
        super().__init__()

        #activation 
        assert att_func_type in ['softmax', 'sigmoid', 'sparsemax']
        self.att_func_type = att_func_type

        assert pool_type in ['mean', 'max', 'sum']
        self.pool_type = pool_type

        if self.att_func_type  == 'softmax':
            self.att_activation = nn.Softmax(dim=-1)
        elif self.att_func_type  == 'sparsemax':
            #print('sparsemax')
            #1/0
            self.att_activation = Sparsemax(dim=-1)
        else:
            self.att_activation = nn.Sigmoid()

        self.att_dim = sd_dim
        self.temperature = temperature


        self.q_map = nn.Sequential(
            nn.LayerNorm(ft_dim),
            nn.Linear(ft_dim, sd_dim),
            nn.GELU(),
            nn.LayerNorm(sd_dim),
            nn.Linear(sd_dim, sd_dim)
        )
        

    def forward(self, ft, sd, mask=None, return_token_att=False):

        #ft = ft.unsqueeze(0) # [bacth, dim] ---> [bacth, 1, dim]
        #sd = sd.expand(ft.shape[0], sd.shape[1], sd.shape[2]) #[1, dictory_size, dim] --> [bacth, dictory_size, dim]

        #map feature to query space
        q = self.q_map(ft) #bacth, token_num, dim

        k = sd #code_num, sd_dim
        k = k.unsqueeze(0) #[1, code_num, sd_dim]
        k = k.transpose(2, 1) #[1,sd_dim, sd_num]


        #-----calculate inner dot
        inner_dot = torch.matmul(q, k) #[bacth, token_num, code_num]
        inner_dot = inner_dot / math.sqrt(self.att_dim) #scale dot norm




        if mask is not None: # mask paded tokens
            #print('mask.shape:', mask.shape)
            #mask [bacth, token_num]
            assert mask.shape == q.shape[:2]
            mask = (mask == 0) * 1 #0 --> 1, inf --> 0
            inner_dot = inner_dot * mask.unsqueeze(-1) #sigmod(-inf) = 0, softmax(-inf) = 0
        # temptural norm
        inner_dot = inner_dot / self.temperature #[bacth, token_num, code_num]


        if return_token_att:
            token_att = self.att_activation(inner_dot)

        
        
        if self.pool_type == 'sum':
            inner_dot = inner_dot.sum(1) #mean poolings
        elif self.pool_type == 'mean':
            inner_dot = inner_dot.mean(1)
        else:
            inner_dot = inner_dot.max(1)[0]

        #----get attention weights
        att_weight = self.att_activation(inner_dot) #normaliztion

        #----calculate weighted sum of v
        #v = self.ln_v(ft) #map to v_space
        
        att_ft = att_weight @ sd  #[bacth, dictory_size] * [dictory_size, dim]  ---> [bacth, sd_num, dim]

        if self.att_func_type == 'sigmoid':
            att_ft = att_ft / att_weight.sum(dim=-1, keepdim=True)

        if return_token_att:
            return token_att, att_ft, sd
        
        return att_weight, att_ft, sd


class DECLIP_FDT(CLIP):
    def __init__(self,image_encode, text_encode, use_allgather, sd_num, sd_dim, sd_temperature, pool_type, att_func_type, raw_img_ft_dim=768, raw_txt_ft_dim=512, nn_size=2**16, nn_topk=1, \
                 return_dense=False, return_simsiam_text=False, return_simsiam_nn_text=False, return_caption=False, return_nn_bank=False, text_mask_type=None,
                 EDA=True, feature_dim=1024, forward_type='split'):
        super(DECLIP_FDT, self).__init__(image_encode, text_encode, use_allgather)
        # TODO change for r50 checkpoint
        self.projector = projection_MLP(feature_dim)
        # self.projector = projection_MLP(1024)
        self.predictor = prediction_MLP(1024)
        self.return_dense = return_dense
        self.return_simsiam_nn_text = return_simsiam_nn_text
        self.return_nn_bank = return_nn_bank
        self.return_caption = return_caption
        self.return_simsiam_text = return_simsiam_text
        self.return_simsiam_nn_text = return_simsiam_nn_text
        self.text_mask_type = text_mask_type
        self.EDA = EDA
        self.forward_type = forward_type
        #import gensim
        #from textaugment import Word2vec
        #model = gensim.models.KeyedVectors.load_word2vec_format('/mnt/cache/liyangguang/GoogleNews-vectors-negative300.bin.gz', binary=True)
        #self.word2vec = Word2vec(model=model)
        from textaugment import EDA
        self.emd = EDA()

        self.space_dict = nn.Parameter(torch.randn(sd_num, sd_dim))
        self.img_query_model = Query_model(ft_dim=raw_img_ft_dim, sd_dim=sd_dim, temperature=sd_temperature, att_func_type=att_func_type, pool_type=pool_type)
        self.txt_query_model = Query_model(ft_dim=raw_txt_ft_dim, sd_dim=sd_dim, temperature=sd_temperature, att_func_type=att_func_type, pool_type=pool_type)


        if self.return_dense:
            raise NotImplementedError('These are bugs in the model, Please Check The Codes!')
            self.projector_d = projection_MLP(2048)  # dense
            self.predictor_d = prediction_MLP(1024)
        if self.return_simsiam_text:
            self.projector_text = projection_MLP(feature_dim)
            self.predictor_text = prediction_MLP(1024)
        if self.return_simsiam_nn_text:
            self.projector_nn_text = projection_MLP(feature_dim)
            self.predictor_nn_text = prediction_MLP(1024)
        if self.return_caption:
            raise NotImplementedError('Not Available')
        if text_mask_type is not None:
            enc_dim = self.encode_text.text_projection.weight.shape[-1]
            self.text_label_predictor = nn.Linear(enc_dim, self.encode_text.vocab_size)
        if self.return_nn_bank:
            #nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
            self.nn_replacer_img = NNMemoryBankModule(size=nn_size, topk=nn_topk)
            self.nn_replacer_text = NNMemoryBankModule(size=nn_size, topk=nn_topk)

    def text_modules(self):
        ret = super(self).text_modules()
        if self.text_mask_type is not None:
            ret.append(self.text_label_predictor)
        return ret

    def visual_modules(self):
        ret = super(self).visual_modules()
        ret.extend([self.predictor, self.projector])
        return ret

    def encode_image(self, image, return_dense=False):
        if return_dense:
            raise ValueError('return dense')
            output = self.visual(image.type(self.dtype), return_dense=return_dense)
            return output

        return self.visual(image.type(self.dtype), return_dense=True, return_raw_feature=True)

    def extract_img_sd_ft(self, images, return_token_att=False):

        #etract image represenation
        img_ft, patch_ft, raw_img_ft_1 = self.encode_image(images)

        img_att, sd_img_ft, img_k = self.img_query_model(patch_ft, self.space_dict, return_token_att=return_token_att)
        
        return sd_img_ft, img_att

    def extract_patch_ft(self, image):
        _, patch_ft = self.visual(image.type(self.dtype), return_dense=True)

        #project
        patch_ft = self.visual.ln_post(patch_ft)
        patch_ft = patch_ft @ self.visual.proj

        return patch_ft

    def extract_txt_sd_ft(self, texts,return_token_att=False):


        txt_ft, word_ft, raw_txt_ft, pad_mask = self.encode_text(texts, return_dense=True,  return_padmask=True, return_raw_feature=True)

        txt_att , sd_txt_ft, txt_k = self.txt_query_model(word_ft, self.space_dict, mask=pad_mask, return_token_att=return_token_att)

        return sd_txt_ft, txt_att

    def extract_word_ft(self, texts):
        
        _, word_ft, pad_masks = self.encode_text(texts, return_dense=True, return_padmask=True)

        #project
        #already LN
        word_ft = self.encode_text.text_projection(word_ft)

        return word_ft, pad_masks




    def forward(self, input, return_dict=False):
        #--------input
        #---images
        if self.return_dense:
            raise ValueError('return dense')


        images = input['images']
        images_1, images_2 = torch.split(images, [3,3], dim=1)

        #--captions
        texts = input['captions']
        texts_aug = []
        #data augmentations
        for caption in texts:
            if self.EDA:
                emd_aug = choice([self.emd.synonym_replacement, self.emd.random_swap, self.emd.random_deletion])
                cap_new = emd_aug(caption)
                if isinstance(cap_new, list):
                    cap_new = ' '.join(cap_new)
                texts_aug.append(cap_new)   # single word: there is a bug
            else:
                raise NotImplementedError('No EDA')
        

        #------extract text embd
        if self.text_mask_type is not None:
            #masked view
            mask_txt_ft, mask_word_ft, text_labels, mask_pad_masks = self.encode_text(texts, mask_type = self.text_mask_type)
            #codebooks
            mask_txt_att , mask_sd_txt_ft, txt_k = self.txt_query_model(mask_word_ft, self.space_dict, mask=mask_pad_masks)
            #aug view
            aug_txt_ft, aug_word_ft, aug_raw_txt_ft, aug_pad_mask = self.encode_text(texts_aug, return_dense=True,  return_padmask=True, return_raw_feature=True)

            aug_txt_att , aug_sd_txt_ft, txt_k = self.txt_query_model(aug_word_ft, self.space_dict, mask=aug_pad_mask)
            
            # mask_sd_txt_ft_aug, word_features_aug, text_labels_aug = self.encode_text(texts_aug, mask_type = self.text_mask_type)
        else:
            raise ValueError('must have text_mask_type')
            

        #extract image embeddings
        img_ft_1, patch_ft_1, raw_img_ft_1 = self.encode_image(images_1)
        img_att_1, sd_img_ft_1, img_k = self.img_query_model(patch_ft_1, self.space_dict)

        img_ft_2, patch_ft_2, raw_img_ft_2 = self.encode_image(images_2)
        img_att_2, sd_img_ft_2, img_k = self.img_query_model(patch_ft_2, self.space_dict)

            

        #simsiam

        z1 = self.projector(sd_img_ft_1)
        z2 = self.projector(sd_img_ft_2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        

        # normalized features
        sd_img_ft_1 = sd_img_ft_1 / (sd_img_ft_1.norm(dim=-1, keepdim=True))
        sd_img_ft_2 = sd_img_ft_2 / (sd_img_ft_2.norm(dim=-1, keepdim=True))
        mask_sd_txt_ft = mask_sd_txt_ft / (mask_sd_txt_ft.norm(dim=-1, keepdim=True)+1e-10)
        aug_sd_txt_ft = aug_sd_txt_ft / (aug_sd_txt_ft.norm(dim=-1, keepdim=True)+1e-10)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)

        if self.training and self.use_allgather:
            link.barrier()
            gathered_sd_img_ft_1 = self.all_gather(sd_img_ft_1)
            gathered_sd_img_ft_2 = self.all_gather(sd_img_ft_2)
            gathered_mask_sd_txt_ft = self.all_gather(mask_sd_txt_ft)
            gathered_aug_sd_txt_ft = self.all_gather(aug_sd_txt_ft)
            link.barrier()

            #image to text
            logits_per_image_1 = logit_scale * sd_img_ft_1 @ gathered_mask_sd_txt_ft.t()
            logits_per_image_2 = logit_scale * sd_img_ft_2 @ gathered_mask_sd_txt_ft.t()
            logits_per_image_1_aug = logit_scale * sd_img_ft_1 @ gathered_aug_sd_txt_ft.t()
            logits_per_image_2_aug = logit_scale * sd_img_ft_2 @ gathered_aug_sd_txt_ft.t()

            #text to images
            logits_per_text_1 = logit_scale * mask_sd_txt_ft @ gathered_sd_img_ft_1.t()
            logits_per_text_2 = logit_scale * mask_sd_txt_ft @ gathered_sd_img_ft_2.t()
            logits_per_text_1_aug = logit_scale * aug_sd_txt_ft @ gathered_sd_img_ft_1.t()
            logits_per_text_2_aug = logit_scale * aug_sd_txt_ft @ gathered_sd_img_ft_2.t()

            if self.return_nn_bank:
                text_features_nn = self.nn_replacer_text(mask_sd_txt_ft.detach().float(), update=False)
                text_features_nn = [t_feat.type(self.dtype) for t_feat in text_features_nn]
                text_features_nn = [t_feat / (t_feat.norm(dim=-1, keepdim=True) + 1e-10) for t_feat in text_features_nn]
                text_features_nn_aug = self.nn_replacer_text(aug_sd_txt_ft.detach().float(), update=True)
                text_features_nn_aug = [t_feat.type(self.dtype) for t_feat in text_features_nn_aug]
                text_features_nn_aug = [t_feat / (t_feat.norm(dim=-1, keepdim=True) + 1e-10) for t_feat in text_features_nn_aug]
                self.nn_replacer_text(mask_sd_txt_ft.detach().float(), update=True)  # update

                gathered_text_features_nn = [self.all_gather(t_feat) for t_feat in text_features_nn]
                gathered_text_features_nn_aug = [self.all_gather(t_feat) for t_feat in text_features_nn_aug]

                logits_per_image_1_nn = torch.cat([logit_scale * sd_img_ft_1 @ t_feat.t()
                                                  for t_feat in gathered_text_features_nn])
                logits_per_image_2_nn = torch.cat([logit_scale * sd_img_ft_2 @ t_feat.t()
                                                  for t_feat in gathered_text_features_nn])
                logits_per_image_1_nn_aug = torch.cat([logit_scale * sd_img_ft_1 @ t_feat.t()
                                                      for t_feat in gathered_text_features_nn_aug])
                logits_per_image_2_nn_aug = torch.cat([logit_scale * sd_img_ft_2 @ t_feat.t()
                                                      for t_feat in gathered_text_features_nn_aug])
        else:
            raise NotImplementedError('2-View: Not Implemented')

        if return_dict:
            link.barrier()
            ret_dict = {}
            ret_dict['logits'] = logits_per_image_1, logits_per_image_2, logits_per_text_1, logits_per_text_2
            ret_dict['logits_aug'] = logits_per_image_1_aug, logits_per_image_2_aug, logits_per_text_1_aug, logits_per_text_2_aug
            ret_dict['simsiam_features'] = p1, p2, z1, z2
            ret_dict['features'] = mask_sd_txt_ft, sd_img_ft_1, sd_img_ft_2

            if self.return_simsiam_nn_text:
                #simsiam_text
                z_text = self.projector_nn_text(mask_sd_txt_ft)
                z_text_nn = [self.projector_nn_text(t_feat) for t_feat in text_features_nn]
                p_text = self.predictor_nn_text(z_text)
                ret_dict['nn_text_simsiam'] = p_text, z_text_nn
            if self.return_simsiam_text:
                z1t = self.projector(mask_sd_txt_ft)
                z2t = self.projector(aug_sd_txt_ft)
                p1t = self.predictor(z1t)
                p2t = self.predictor(z2t)
                ret_dict['text_simsiam'] = p1t, p2t, z1t, z2t
            if self.return_nn_bank:
                ret_dict['nn_text_logits'] = logits_per_image_1_nn, logits_per_image_2_nn, logits_per_image_1_nn_aug, logits_per_image_2_nn_aug
            if self.text_mask_type is not None:
                if self.encode_text.text_encode_type == 'Bert':
                    text_pred_mask = mask_word_ft
                else:  # 30000
                    text_pred_mask = self.text_label_predictor(mask_word_ft)

                pred_mask = (text_labels != -100)
                mlm = F.cross_entropy(text_pred_mask[pred_mask], text_labels[pred_mask].to(text_pred_mask.device), reduce=None)
                ret_dict['text_self_supervised'] = mlm.mean()
            return ret_dict
        raise NotImplementedError('Must Return A Dict')



def declip_fdt_vitb32(**kwargs):
    """
    Constructs a clip_vitb32 model.
    """
    image_encode = visual_transformer_B32(**kwargs['image_encode'])
    text_encode = text_transformers(**kwargs['text_encode'])
    model = DECLIP_FDT(image_encode,text_encode,**kwargs['clip'])
    return model
