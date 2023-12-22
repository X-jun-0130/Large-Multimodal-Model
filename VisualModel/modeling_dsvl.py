import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers import LlamaForCausalLM
from VisualModel.clip_encoder import CLIPVisionTower
from VisualModel.qwen_clip import VisionTransformer
from torch import nn
from torch.nn import  CrossEntropyLoss
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
from VisualModel.vis_proj import VisProjection_vit, VisProjection_perceiver

DEFAULT_LABEL_PADDING_NUM = -100

def create_dsvl_model_and_transforms(
        text_tokenizer=None,
        args=None):
    assert args['vision_model_name_or_path'] is not None
    assert args['lm_model_name_or_path'] is not None

    lang_config = AutoConfig.from_pretrained(args['lm_model_name_or_path'])

    if 'qwen' in args['vision_model_name_or_path'].lower():
        # use a fake config for consistent
        vis_config = AutoConfig.from_pretrained(args['vision_model_name_or_path'])
        vis_config = vis_config.vision_config
        vis_encoder = VisionTransformer(
            image_size=448,
            patch_size=vis_config.patch_size,
            width=vis_config.hidden_size,
            layers=vis_config.num_hidden_layers,
            heads=vis_config.num_attention_heads,
            mlp_size=vis_config.intermediate_size,
            output_dim=4096,
        ) 
        vis_encoder.load_state_dict(torch.load(os.path.join(args['vision_model_name_or_path'], 'pytorch_model.bin'), map_location='cpu'), strict=True)
        vis_config.hidden_size = 4096 # we need to change the hidden size to 4096
    elif 'clip' in args['vision_model_name_or_path'].lower():
        vis_encoder = CLIPVisionTower(args['vision_model_name_or_path'], -2, 'patch', delay_load=True)
        vis_encoder.load_model()
        vis_config = vis_encoder.config 
    else:
        raise ValueError("We currently only support qwen's modifed clip and other clip models")
     
    tokenizer = text_tokenizer

    lang_decoder = LlamaForCausalLM.from_pretrained(args['lm_model_name_or_path'],  use_cache =args['use_cache'], use_flash_attention_2=args['use_flash_attention_2'])

    if lang_config.vocab_size < len(tokenizer):
        lang_config.vocab_size = len(tokenizer)
        lang_decoder.resize_token_embeddings(len(tokenizer))

    model = DeepSpeedViLModel(vis_encoder, lang_decoder, \
                                tokenizer, \
                                vis_config=vis_config, \
                                lang_config=lang_config, \
                                args=args)
    
    return model,  tokenizer


class DeepSpeedViLModel(nn.Module):
    def __init__(self, vis_encoder,
                    lang_decoder,
                    tokenizer,
                    vis_config=None, 
                    lang_config=None,
                    args=None):
        super().__init__()
        self.vis_encoder = vis_encoder #图像层
        self.lang_decoder = lang_decoder #语言层

        self.emd_tokens = self.lang_decoder.get_input_embeddings() #词向量模块
        
        self.tokenizer = tokenizer 
        self.lang_config = lang_config
        self.vocab_size = getattr(self.lang_config, 'vocab_size')
        self.args = args

        self._enable_special_token()
        self.projection = self.build_projection(vis_config, self.lang_config.hidden_size) #线性映射层   
        self._init_weight() #冻结
        
        # get padding token embedding
        self.padding_embedding = None 
        self.vis_encoder_update = None

    def _enable_special_token(self):
        self.DEFAULT_IMAGE_TOKEN_ID = self.tokenizer.convert_tokens_to_ids('<|image|>')
  
    def _init_weight(self):
        self.vis_encoder.requires_grad_(False) 
        self.projection.requires_grad_(True)  
        self.lang_decoder.requires_grad_(True)
        self.emd_tokens.weight.requires_grad_(True)
    
    def build_projection(self, vis_config, lang_dim):
        if self.args['vis_proj'] == 'vit':
            output =  VisProjection_vit(vis_config, lang_dim=lang_dim)
            return output 
        elif 'baseline' in  self.args['vis_proj']:
            if int(self.args['vis_proj'][-1]) == 1:
                return nn.Sequential( 
                                nn.Linear(vis_config.hidden_size, lang_dim), # an example implementation
                                nn.LayerNorm(lang_dim, eps=1e-12)
                                )
            else:
                mlp_depth = int(self.args['vis_proj'][-1])
                modules = [nn.Linear(vis_config.hidden_size, lang_dim)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(lang_dim, lang_dim))
                modules.append(nn.LayerNorm(lang_dim, eps=1e-12))
                return nn.Sequential(*modules)

        elif self.args['vis_proj'] == 'perceiver':
            return VisProjection_perceiver(vis_config, lang_dim=lang_dim)

    def concat(self, img_proj, lang, attention_mask, input_labels, image_num, do_generation=False):
        output_lang = []
        output_attention_mask = []
        output_input_labels = []
        
        def split_tensor_by_a_list(tensor, split_list):
            output = []
            initial_pos = 0
            accumulated_sum = [sum(split_list[:i]) for i in range(1, len(split_list)+1)]
            for pos in accumulated_sum:
                output.append(tensor[initial_pos:pos])
                initial_pos = pos
            del tensor
            return output
        
        img_proj = split_tensor_by_a_list(img_proj, image_num)

        for index in range(len(img_proj)): # each seq has multi iamges, so we need to use it as index
            initial_pos = 0
            cur_img = img_proj[index]
            cur_lang = lang[index]
            cur_attention_mask = attention_mask[index]
            cur_input_labels = input_labels[index]
            img_pos_list = cur_lang.eq(self.DEFAULT_IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
            assert len(img_pos_list) == image_num[index], "the number of images in the lang and image_num does not match"
            if len(img_pos_list) == 0:
                continue # there is no image probably it is a pure text insturction
            
            cur_lang = self.emd_tokens(cur_lang) # get the real embedding
 
            for img_i, img_pos in zip(torch.flip(cur_img, dims=(0,)), torch.flip(img_pos_list, dims=(0,))): # do it reversely so that we can easily insert the image
                lang_pre_img_embed = cur_lang[initial_pos:img_pos]
                attention_mask_pre_img = cur_attention_mask[initial_pos:img_pos]
                input_labels_pre_img = cur_input_labels[initial_pos:img_pos]

                lang_post_img_embed = cur_lang[img_pos+1:]
                attention_mask_post_img = cur_attention_mask[img_pos+1:]
                input_labels_post_img = cur_input_labels[img_pos+1:]
                # now we need to concat the image embedding
                lang_full = torch.cat((lang_pre_img_embed, img_i, lang_post_img_embed), dim=0)
    
                attention_mask_full = torch.cat((attention_mask_pre_img, 1 * torch.ones_like(img_i[:, 0]), attention_mask_post_img), dim=0)

                input_labels_full = torch.cat((input_labels_pre_img.long(), DEFAULT_LABEL_PADDING_NUM * torch.ones_like(img_i[:, 0], dtype=torch.long), input_labels_post_img),   dim=0)

                cur_lang = lang_full
                cur_attention_mask = attention_mask_full
                cur_input_labels = input_labels_full
            # append to the output 
            output_lang.append(lang_full.unsqueeze(0))
            output_attention_mask.append(attention_mask_full.unsqueeze(0))
            output_input_labels.append(input_labels_full.unsqueeze(0))

        if self.padding_embedding is None:
            # with torch.no_grad():
            self.padding_embedding = self.emd_tokens(torch.tensor(self.tokenizer.pad_token_id).to(lang.device).unsqueeze(0)).unsqueeze(0).detach()

        def pad_tensor_list(tensor_list, pad_token_id, pad_vec=False):
            max_len = max([tensor.size(1) for tensor in tensor_list])
            if not do_generation:
                max_len = int(np.ceil(max_len / 8) * 8) # make it divisible by 8
            padded_tensor_list = []
            for tensor in tensor_list:
                if max_len > tensor.size(1):
                    if pad_vec: # output_lang padding
                        # pad with self.padding_embedding 
                        padded_tensor = torch.cat([tensor] + [self.padding_embedding] * (max_len - tensor.size(1)), dim=1)
                        
                    else:
                        padded_tensor = F.pad(tensor, (0, max_len - tensor.size(1)), value=pad_token_id)
                else:
                    padded_tensor = tensor
                padded_tensor_list.append(padded_tensor)
            return padded_tensor_list
        
        output_lang = pad_tensor_list(output_lang, self.tokenizer.pad_token_id, pad_vec=True)
        output_attention_mask = pad_tensor_list(output_attention_mask, 0)
        output_input_labels = pad_tensor_list(output_input_labels, DEFAULT_LABEL_PADDING_NUM)

        return torch.cat(output_lang, dim=0), torch.cat(output_attention_mask, dim=0), torch.cat(output_input_labels, dim=0)

    def forward(self, images, input_ids,
            attention_mask=None,
            labels=None,
            image_num=1,
            past_key_values=None,
            use_cache=False,
            output_attentions=False, 
            output_hidden_states=False,
            return_dict=True):
        
        assert attention_mask is not None, "attention mask is required"
        assert labels is not None, "labels is required"

        with torch.no_grad():
            img_feature = self.vis_encoder(images)
        img_proj = self.projection(img_feature) # [batch_size, 576, hidden_size] 
        
        hidden_states, attention_mask, labels = self.concat(img_proj, input_ids, attention_mask, labels, image_num)
 
        logits = self.lang_decoder(input_ids=None, 
                                inputs_embeds=hidden_states,
                                attention_mask=attention_mask,
                                labels=None,
                                past_key_values=past_key_values,
                                use_cache=use_cache,
                                output_attentions=output_attentions, 
                                output_hidden_states=output_hidden_states,
                                return_dict=return_dict).logits

        logits_shift = logits[..., :-1, :].contiguous().view(-1, self.vocab_size) # remove the last token
        labels_shift = labels[..., 1:].contiguous().to(logits_shift.device).view(-1) # remove the first token
        # select index that is not -100
        labels_index = labels_shift != -100
        if torch.sum(labels_index) ==0:
            logits_shift = logits_shift[-2:,:].contiguous()
            labels_shift = labels_shift[-2:].contiguous()            
        else:
            logits_shift = logits_shift[labels_index,:].contiguous()
            labels_shift = labels_shift[labels_index].contiguous()

        loss_fct = CrossEntropyLoss() 
        loss = loss_fct(logits_shift, labels_shift) 
        
        return [loss, logits]

    
    @torch.no_grad()
    def generate(self, 
            images=None,
            input_ids=None,
            attention_mask=None,
            generation_kwargs={}, # add some meaningful default values
            ):
        assert input_ids.size()[0] == 1, "only support batch size == 1 for now"
        assert input_ids is not None, "input_ids is required"

        if images is None:
            output = self.lang_decoder.generate(input_ids=input_ids,
                                                **generation_kwargs)
            
            return self.tokenizer.batch_decode(output)[0].split('\n Assistant:')[-1].strip(self.tokenizer.eos_token)
        
        else:
            attention_mask = torch.ones_like(input_ids)
            input_labels = torch.ones_like(input_ids) 
            
            # this part for now does not require gradient
            img_feature = self.vis_encoder(images) 
            img_proj = self.projection(img_feature)
            
            hidden_states, _, _ = self.concat(img_proj, input_ids, attention_mask, input_labels, image_num=[images.size(0)], do_generation=True)
   
            output = self.lang_decoder.generate(input_ids=None,
                                                attention_mask=None,
                                                inputs_embeds=hidden_states,
                                                **generation_kwargs)
            
            return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]


    def gradient_checkpointing_enable(self):
        self.vis_encoder.gradient_checkpointing_enable()
        self.lang_decoder.gradient_checkpointing_enable()
