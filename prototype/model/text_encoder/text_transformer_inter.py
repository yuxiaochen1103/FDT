import sys
sys.path.append('../../')

import os
import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple, Union, List
from .base_transformer import Transformer_module_list, LayerNorm
from prototype.model.utils.text_utils.simple_tokenizer import SimpleTokenizer as _Tokenizer
from prototype.model.utils.text_utils.mask_tokens import MaskTokens



class TextTransformer(nn.Module):
    def __init__(self,
                embed_dim: int,
                context_length: int,
                transformer_width: int,
                transformer_heads: int,
                transformer_layers: int,
                positional_embedding_flag: bool,
                checkpoint: bool,
                bpe_path=None,
                text_encode_type=None,
                text_model_utils=None):
        super().__init__()
        self.context_length = context_length
        self.positional_embedding_flag = positional_embedding_flag
        self.text_encode_type = text_encode_type
        self.text_model_utils = text_model_utils

        if self.text_encode_type == 'Transformer':

            self.tokenizer = _Tokenizer(bpe_path=bpe_path)
            self.transformer = Transformer_module_list(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask(),
                checkpoint=checkpoint,
            )
            self.vocab_size = len(self.tokenizer.encoder)
            self.token_embedding = nn.Embedding(self.vocab_size, transformer_width)
            self.positional_embedding = nn.Parameter(torch.normal(mean=0, std=0.02, size=(self.context_length, transformer_width)))  # Fix!!!
            #self.ln_final = LayerNorm(transformer_width)
            self.text_projection = None
            # text transformer init
            self.initialize_parameters()

        elif self.text_encode_type in ['Bert', 'Bert_half']: # BertForMaskedLM.cls.bias is equal to BertForMaskedLM.decoder.bias (self.decoder.bias=self.bias in BertLMPredictionHead line 654)
            from transformers import AutoTokenizer, BertForMaskedLM # tokenizer: multi-process(should remove it)
            # os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            # # You Should Download it first(python and build a model is just okay)
            self.text_max_length = 100
            self.text_tokenizer = AutoTokenizer.from_pretrained("/mnt/lustre/share_data/zhaolichen/cached_pretrained_transformers/bert-base-uncased")
            self.vocab_size = len(self.text_tokenizer)
            self.text_transformer = BertForMaskedLM.from_pretrained("/mnt/lustre/share_data/zhaolichen/cached_pretrained_transformers/bert-base-uncased")
            if text_model_utils.get('random', False):
                print('text encode random')
                from transformers import BertModel, BertConfig
                self.text_transformer = BertModel(BertConfig())
            self.text_projection = nn.Linear(768, embed_dim)

        elif self.text_encode_type in ['Bert_gvx', 'Bert_gvx_half']:
            from transformers import BertTokenizer, BertModel
            self.text_max_length = 100
            self.text_tokenizer = BertTokenizer.from_pretrained("/mnt/lustre/share_data/huangbin1/pretrained/BERT")
            self.vocab_size = len(self.text_tokenizer)
            self.text_transformer = BertModel.from_pretrained("/mnt/lustre/share_data/huangbin1/pretrained/BERT")
            if text_model_utils.get('random', False):
                print('text encode random')
                from transformers import BertModel, BertConfig
                self.text_transformer = BertModel(BertConfig())
            self.text_projection = nn.Linear(768, embed_dim)

        elif self.text_encode_type == 'GPT2':
            from transformers import AutoTokenizer, AutoModel
            self.text_max_length = 100
            self.text_tokenizer = AutoTokenizer.from_pretrained("/mnt/lustre/share_data/zhaolichen/cached_pretrained_transformers/gpt2")
            self.vocab_size = len(self.text_tokenizer)
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token  # PAD TOKEN
            self.text_transformer = AutoModel.from_pretrained("/mnt/lustre/share_data/zhaolichen/cached_pretrained_transformers/gpt2")
            if text_model_utils.get('random', False):
                from transformers import GPT2Model, GPT2Config
                self.text_transformer = GPT2Model(GPT2Config())
            self.text_projection = nn.Linear(768, embed_dim)

        elif self.text_encode_type in ['Roberta', 'Roberta_large', 'Bert_large']:
            if self.text_encode_type == 'Roberta':
                pretrained_path = '/mnt/lustre/share_data/zhaolichen/cached_pretrained_transformers/' + 'roberta-base'
                projection_shape = 768
            elif self.text_encode_type == 'Roberta_large':
                pretrained_path = '/mnt/lustre/share_data/zhaolichen/cached_pretrained_transformers/' + 'roberta-large'
                projection_shape = 1024
            elif self.text_encode_type == 'Bert_large':
                pretrained_path = '/mnt/lustre/share_data/zhaolichen/cached_pretrained_transformers/' + 'bert-large-uncased'
                projection_shape = 1024
            else:
                raise NotImplementedError(self.text_encode_type)
            from transformers import AutoTokenizer, AutoModel # tokenizer: multi-process(should remove it)
            # os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            # # You Should Download it first(python and build a model is just okay)
            self.text_max_length = 100
            self.text_tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
            self.vocab_size = len(self.text_tokenizer)
            self.text_transformer = AutoModel.from_pretrained(pretrained_path)
            if text_model_utils.get('random', False):
                print('text encode random')
                from transformers import BertModel, BertConfig
                self.text_transformer = BertModel(BertConfig())
            self.text_projection = nn.Linear(projection_shape, embed_dim)

        else:
            raise NotImplementedError(str(self.text_encode_type))

        if self.text_model_utils.get('freeze', False):
            self.text_transformer.eval()
            for param in self.text_transformer.parameters():
                param.requires_grad = False

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection.weight, std=self.transformer.width ** -0.5)

    @property
    def dtype(self):
        return self.positional_embedding.dtype

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def tokenize(self, texts: Union[str, List[str]], context_length: int = 77, return_length: bool = False, mask_type=None):
        if isinstance(texts, str):
            texts = [texts]

        #add sot, eot
        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]

        #tokenize
        all_tokens =  [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                all_tokens[i] = [tokens[0]]+tokens[1:context_length-1]+[tokens[-1]]
            all_tokens[i] = torch.Tensor(all_tokens[i]).long()

        #mask token
        if mask_type is not None:
            mask_token = self.tokenizer.encoder["<|mask|>"]
            special_tokens = [sot_token, eot_token, mask_token]
            masked_tokens = [MaskTokens(tokens, mask_type=mask_type, mask_token=mask_token, special_tokens=special_tokens, tokenizer_length=len(self.tokenizer.encoder)) for tokens in all_tokens]
            all_tokens = [item[0] for item in masked_tokens]
            all_labels = [item[1] for item in masked_tokens]

        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        labels = torch.ones(len(all_tokens), context_length, dtype=torch.long) * -100
        token_lengths = torch.ones(len(all_tokens), dtype=torch.long)

        pad_mask = torch.ones(len(all_tokens), context_length)



        for i, tokens in enumerate(all_tokens):
            result[i, :len(tokens)] = tokens
            token_len = min(len(tokens), context_length)
            pad_mask[i, :token_len] = 0 #1 are none_masked elements
            token_lengths[i] = token_len
            if mask_type is not None:
                labels[i, :len(tokens)] = all_labels[i]
        
        pad_mask = pad_mask.masked_fill(pad_mask == 1, float("-inf"))

        if mask_type:
            # print(result[0], labels[0], '<< masking', flush=True)
            return result, labels, pad_mask
        if return_length:
            return result, token_lengths, pad_mask
        else:
            return result, pad_mask

    def forward_low(self, text, low_level_idx, mask_type=None):

        if self.text_encode_type == 'Transformer':
            #---tokenize
            if mask_type is not None:
                texts, labels, pad_mask = self.tokenize(text, context_length=self.context_length, mask_type=mask_type)
            else:
                texts, pad_mask = self.tokenize(text, context_length=self.context_length, mask_type=mask_type)
            #------feed to transformer
            pad_mask = pad_mask.cuda()
            x = self.token_embedding(texts.cuda()).type(self.dtype)  # [batch_size, n_ctx, d_model]
            if self.positional_embedding_flag:
                x = x + self.positional_embedding.type(self.dtype)  # Fix!!!
            x = x.permute(1, 0, 2)  # NLD -> LND

            #forward
            for i in range(low_level_idx):
                blk = self.transformer.resblocks[i]
                x = blk(x)

            #get index for last word
            cls_idxs = texts.argmax(dim=-1)
            return x, cls_idxs, pad_mask
        else:
            raise ValueError('no text encoder')
       
    
    def forward_high(self, x, low_level_idx):
        layer_num = len(self.transformer.resblocks)
        for i in range(low_level_idx, layer_num):
            blk = self.transformer.resblocks[i]
            x = blk(x)
        return x

    # TODO Bert-Masking Not Implemented for most tokenizers
    def forward(self, text, mask_type=None, return_dense=False, return_raw_feature=False, return_padmask=False):

        #return_dense: return word features
        #return_feature: return the features before linear projection
        #return_padask
        if self.text_encode_type == 'Transformer':
            #---tokenize
            
            if mask_type is not None:
                texts, labels, pad_mask = self.tokenize(text, context_length=self.context_length, mask_type=mask_type)
            else:
                texts, pad_mask = self.tokenize(text, context_length=self.context_length, mask_type=mask_type)
            #------feed to transformer
            pad_mask = pad_mask.cuda()
            x = self.token_embedding(texts.cuda()).type(self.dtype)  # [batch_size, n_ctx, d_model]
            if self.positional_embedding_flag:
                x = x + self.positional_embedding.type(self.dtype)  # Fix!!!
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)

            cls_idxs = texts.argmax(dim=-1)

        elif self.text_encode_type in ['Bert_gvx', 'Bert_large', 'Bert_gvx_half', 'Bert', 'GPT2', 'Roberta', 'Roberta_large']:
            batch_tokenids = list()
            for txt in text:
                tokens = self.text_tokenizer.tokenize(txt.strip())
                if len(tokens) > self.context_length - 2:
                    tokens = tokens[:(self.context_length - 2)]
                if self.text_encode_type in ['Bert_gvx', 'Bert', 'Bert_large', 'Bert_gvx_half']:
                    tokens = ["[CLS]"] + tokens + ["[SEP]"]
                elif self.text_encode_type in ['Roberta', 'Roberta_large']:
                    tokens = ['<s>'] + tokens + ['</s>']
                else:
                    raise NotImplementedError()
                    # import ipdb
                    # ipdb.set_trace()
                token_ids = self.text_tokenizer.convert_tokens_to_ids(tokens)
                padding = [0] * (self.context_length - len(token_ids))
                token_ids += padding
                batch_tokenids.append(token_ids)
            tokens_tensor = torch.tensor(batch_tokenids).long()

            if mask_type is not None:
                masked_tokens = [MaskTokens(tokens, mask_type=mask_type,
                                            mask_token=self.text_tokenizer.mask_token_id,
                                            special_tokens_mask=self.text_tokenizer.get_special_tokens_mask(tokens, already_has_special_tokens=True),
                                            tokenizer_length=self.vocab_size) for tokens in tokens_tensor]
                all_tokens = [item[0] for item in masked_tokens]
                all_labels = [item[1] for item in masked_tokens]
                labels = torch.stack(all_labels)
                tokens_tensor = torch.stack(all_tokens)

            tokens_tensor = tokens_tensor.cuda()

            segments_tensors = torch.ones_like(tokens_tensor)
            outputs = self.text_transformer(tokens_tensor, token_type_ids=segments_tensors, output_hidden_states=True)
            words_feat = outputs[0]
            cls_emb=outputs[0][:,0,:]
            # import ipdb
            # ipdb.set_trace()
            if self.text_encode_type == 'Bert_gvx_half':
                cls_emb = outputs[2][6][:, 0, :]  # 0-12
            elif self.text_encode_type == 'Bert':
                cls_emb = outputs[1][-1][:, 0, :]
            if self.text_model_utils.get('text_projection', True):
                x = self.text_projection(cls_emb)
            else:
                x = cls_emb

        if mask_type is not None:
            return x, cls_idxs, pad_mask, labels

        return x, cls_idxs, pad_mask


def text_transformers(**kwargs):
    default_kwargs = {
        # 'embed_dim':1024, embed_dim from config
        'context_length':77,
        'transformer_width':512,
        'transformer_heads':8,
        'transformer_layers':12,
        'positional_embedding_flag':True,
        'checkpoint': False
    }
    default_kwargs.update(**kwargs)
    model = TextTransformer(**default_kwargs)
    return model
