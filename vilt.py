import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.models.vilt.modeling_vilt import TextEmbeddings
from transformers.models.vilt.modeling_vilt import ViltEncoder
from transformers.models.vilt.modeling_vilt import ViltMLMHead
from transformers.models.vilt.modeling_vilt import ViltPooler
from transformers.models.vilt.modeling_vilt import ViltPreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,  
)


class MyVisualFeatureEncoder(nn.Module):
    def __init__(self,config,feat_dim = 2048):
        super().__init__()
        self.visn_fc = nn.Linear(feat_dim,config.hidden_size)
        self.visn_layer_norm = nn.LayerNorm(config.hidden_size , eps = 1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(
        self,
        visual_feats
    ):
        return self.dropout(self.visn_layer_norm(self.visn_fc(visual_feats)))

class MyViltEmbedding(nn.Module):
    def __init__(self,config,feat_dim = 2048) :
        super().__init__()
        self.config = config
        # text embedding
        self.text_embedding = TextEmbeddings(config)
        # img embedding
        self.img_embedding = MyVisualFeatureEncoder(config, feat_dim = feat_dim)
        # modality type (text/patch) embedding
        self.token_type_embedding = nn.Embedding(config.modality_type_vocab_size,config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(
        self,
        input_ids , 
        attention_mask ,
        token_type_ids , 
        feats , 
    ):  
        # 1. text embedding
        text_embeds = self.text_embedding( input_ids = input_ids, token_type_ids = token_type_ids)
        # 2. img embedding
        img_embeds = self.img_embedding(feats)      # shape [batch_size, 1 ,hidden_size] 
        img_masks = torch.ones(img_embeds.size()[:-1],dtype = torch.long,device=input_ids.device)
        # 3. add modality type embeddings
        image_token_type_idx = 1
        text_embeds = text_embeds + self.token_type_embedding(torch.zeros_like(attention_mask, dtype=torch.long, device=text_embeds.device))
        img_embeds = img_embeds + self.token_type_embedding(torch.full_like(img_masks, image_token_type_idx, dtype=torch.long, device=text_embeds.device))

        output_embedding = torch.cat([text_embeds,img_embeds],dim = 1)
        output_masks = torch.cat([attention_mask , img_masks],dim = 1)
        return output_embedding , output_masks

class MyViltModel(ViltPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        self.embeddings = MyViltEmbedding(config,feat_dim=2048)
        self.encoder = ViltEncoder(config)
        self.pooler = ViltPooler(config)
        self.layernorm = nn.LayerNorm(config.hidden_size , eps = config.layer_norm_eps)
        self.post_init()
    def get_input_embeddings(self):
        return self.embeddings.text_embedding.word_embeddings
    def forward(
        self,
        input_ids ,             # shape:[batch_size, seq_len]
        attention_mask  ,       # shape:[batch_size, seq_len]   
        token_type_ids ,        # shape:[batch_size, seq_len]
        feats ,                 # shape:[batch_size,num_channels,height,width]
    ):  
        input_shape , device = input_ids.size() , input_ids.device
        output_embeddings , output_masks = self.embeddings(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            feats = feats,
        )
        # transformers
        extended_attention_mask = self.get_extended_attention_mask(output_masks, input_shape, device)
        encoder_outputs = self.encoder(
            hidden_states = output_embeddings,
            attention_mask = extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) 

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )

class MyViltForPretrain(ViltPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.vilt = MyViltModel(config)
        self.mlm_score = ViltMLMHead(config,weight=self.vilt.get_input_embeddings().weight)   # 用于MLM
        self.seq_relationship = nn.Linear(config.hidden_size , 2) # 用于图文匹配
        self.post_init()
        self.loss_fct = CrossEntropyLoss()
    def forward(
        self,
        input_ids ,
        attention_mask,
        token_type_ids,
        feats,
        labels,         
        matchs,         
    ):
        device = input_ids.device
        output = self.vilt(
            input_ids = input_ids,           
            attention_mask = attention_mask,     
            token_type_ids = token_type_ids,       
            feats =feats ,               
        )
        sequence_output , pooled_output = output[:2]
        text_seq_len = input_ids.shape[1]
        text_features , _ = (sequence_output[:, :text_seq_len], sequence_output[:, text_seq_len:])
        match_logits = self.seq_relationship(pooled_output)
        match_loss = self.loss_fct(match_logits.view(-1,2),matchs.view(-1))

        mlm_logits = self.mlm_score(text_features)
        pred_match_txt = mlm_logits[matchs.view(-1) == 1]
        true_match_txt = labels[matchs.view(-1)==1]
        if true_match_txt.shape[0] > 0:
            masked_loss = self.loss_fct(pred_match_txt.view(-1,self.config.vocab_size),true_match_txt.view(-1))
        else :
            masked_loss = torch.tensor(0.0).to(device=device)
        
        _ ,idx = match_logits.max(1)
        right_match = (matchs.squeeze(1) == idx).sum().item()
        
        return {
            'mlm_loss':masked_loss,
            'match_loss':match_loss,
            'right_match' : right_match
        }

class MyViltFinetune(ViltPreTrainedModel):
    def __init__(self,config , output_dim = 13) :
        super().__init__(config)
        self.config = config
        self.output_dim = output_dim 
        self.vilt = MyViltModel(config)
        self.cls = nn.Linear(config.hidden_size, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        input_ids ,
        attention_mask ,
        token_type_ids , 
        feats , 
    ):
        outputs = self.vilt(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids , 
            feats = feats,
        )
        _ , pooled_output = outputs[:2]
        outputs = self.sigmoid(self.cls(pooled_output))
        return outputs