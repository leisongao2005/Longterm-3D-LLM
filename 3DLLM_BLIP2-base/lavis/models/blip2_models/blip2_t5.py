import logging

import torch
import torch.nn as nn
# from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from positional_encodings.torch_encodings import PositionalEncoding1D

import time


@registry.register_model("blip2_t5")
class Blip2T5(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )

        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder = self.visual_encoder.eval()
        self.visual_encoder.train = disabled_train
        
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, 1408)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)

        location_tokens = []
        for i in range(32764):  
            location_tokens.append("<loc%d>" % i)  
        action_tokens = [
            "<PICK UP ",
            "<PUT DOWN ",
            "<GO TO NEW ROOM>",
            "<GO TO ROOM",
        ]
        self.t5_tokenizer.add_special_tokens({"additional_special_tokens": location_tokens + action_tokens})

        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model, config=t5_config)

        self.t5_model.resize_token_embeddings(len(self.t5_tokenizer))

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data

        self.t5_model.get_output_embeddings().requires_grad_(True)
        self.t5_model.get_input_embeddings().requires_grad_(True)

        self.t5_proj = nn.Linear(self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)

        pos_model = PositionalEncoding1D(1408 // 3)
        x = torch.zeros(1, 256, 1408 // 3)
        self.pos_embedding = pos_model(x).squeeze().cuda()

        self.max_txt_len = max_txt_len
        self.prompt = ""
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

    def forward(self, samples):
        batch_size = samples["pc_feat"].size(0)
        num_scenes = samples["pc_feat"].size(1) 
        
        # Reshape pc_feat from [B, S, 5000, 1408] to [B*S, 5000, 1408]
        pc_embeds = samples["pc_feat"].reshape(batch_size * num_scenes, -1, samples["pc_feat"].size(-1))
        
        # Reshape pc from [B, S, 5000, 3] to [B*S, 5000, 3]
        pc = samples["pc"].reshape(batch_size * num_scenes, -1, 3).long()
        
        with torch.amp.autocast("cuda", dtype=torch.float32):
            # Create tensor for positional embeddings, matching pc_embeds shape
            all_pcs = torch.zeros_like(pc_embeds)  # [B*S, 5000, 1408]
            
            # Vectorized positional embedding lookup (no loops)
            # Extract x, y, z indices - each will be [B*S, 5000]
            x_indices = pc[..., 0]
            y_indices = pc[..., 1]
            z_indices = pc[..., 2]
            
            # Lookup embeddings for each coordinate - each will be [B*S, 5000, 469]
            x_embeds = self.pos_embedding[x_indices]
            y_embeds = self.pos_embedding[y_indices]
            z_embeds = self.pos_embedding[z_indices]
            
            # Concatenate along the last dimension to get [B*S, 5000, 1407]
            pos_embeds = torch.cat([x_embeds, y_embeds, z_embeds], dim=-1)

             # Concatenate along the last dimension to get [B*S, 5000, 1407]
            pos_embeds = torch.cat([x_embeds, y_embeds, z_embeds], dim=-1)
            
            # Create tensor for positional embeddings with zeros in remaining dims
            all_pcs = torch.zeros_like(pc_embeds)
            all_pcs[..., :pos_embeds.size(-1)] = pos_embeds

            # # Process each batch item (now including scenes)
            # for j in range(pc.shape[0]):  # Iterate over B*S combined batch items
            #     pcs = []
            #     for i in range(3):  # Iterate over x, y, z
            #         # Extract coordinate values for the j-th batch+scene item
            #         pc_i = pc[j][:, i]  # [5000]
            #         # Look up positional embeddings (exactly as in original code)
            #         pcs.append(self.pos_embedding[pc_i])  # Each has shape [5000, 469]
                
            #     # Concatenate xyz positional embeddings
            #     pcs = torch.cat(pcs, -1)  # Shape becomes [5000, 1407]
                
            #     # Assign to the first 1407 dimensions of all_pcs for this batch+scene
            #     all_pcs[j][:, :1407] = pcs
            
            # all_pcs = all_pcs.cuda()

            pc_embeds = pc_embeds + 0.01 * all_pcs
            image_atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(pc_embeds.device)  # [B*S, 5000]

            query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)  # [B*S, 32, 768]
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=pc_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            
            scene_outputs = self.t5_proj(query_output.last_hidden_state)  # [B*S, 32, t5_dim]
        
            # reshape to get the scenes back for each batch [B*S, 32, t5_dim] to [B, S*32, t5_dim]
            t5_dim = scene_outputs.size(-1)
            inputs_t5 = scene_outputs.reshape(batch_size, num_scenes * 32, t5_dim)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(inputs_t5.device)  # [B, S*32]
        
        # The rest of the code remains unchanged
        if self.prompt:
            text_input = [self.prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]
        
        with torch.amp.autocast("cuda", dtype=torch.float32):
            input_tokens = self.t5_tokenizer(
                text_input,
                padding="longest",
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to(inputs_t5.device)
            
            output_tokens = self.t5_tokenizer(
                samples["answer"],
                padding="longest",
                truncation=True,
                max_length=300,
                return_tensors="pt",
            ).to(inputs_t5.device)
            
            # Repeat inputs for each answer
            batch_input_tokens_input_ids = []
            batch_input_tokens_atts = []
            batch_atts_t5 = []
            batch_inputs_t5 = []
            
            for b, n in enumerate(samples["n_answers"]):
                batch_input_tokens_input_ids += [input_tokens.input_ids[b]] * n
                batch_input_tokens_atts += [input_tokens.attention_mask[b]] * n
                batch_atts_t5 += [atts_t5[b]] * n
                batch_inputs_t5 += [inputs_t5[b]] * n
            
            batch_input_tokens_input_ids = torch.stack(batch_input_tokens_input_ids, dim=0)
            batch_input_tokens_atts = torch.stack(batch_input_tokens_atts, dim=0)
            batch_atts_t5 = torch.stack(batch_atts_t5, dim=0)
            batch_inputs_t5 = torch.stack(batch_inputs_t5, dim=0)
            
            encoder_atts = torch.cat([batch_atts_t5, batch_input_tokens_atts], dim=1)
            
            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )
            
            inputs_embeds = self.t5_model.encoder.embed_tokens(batch_input_tokens_input_ids)
            inputs_embeds = torch.cat([batch_inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            
            loss = outputs.loss
            
            return {"loss": loss}

    # # sequential/slow but correct
    # def forward(self, samples):
    #     batch_size = samples["pc_feat"].size(0)
    #     num_scenes = samples["pc_feat"].size(1)
        
    #     scene_outputs = []
    #     for scene_idx in range(num_scenes):

    #         pc_feat_scene = samples["pc_feat"][:, scene_idx]
    #         pc_scene = samples["pc"][:, scene_idx]

    #         with torch.amp.autocast("cuda", dtype=torch.float32):
    #             pc_embeds = pc_feat_scene

    #         with torch.amp.autocast("cuda", dtype=torch.float32):
    #             pc = pc_scene.long()
    #             all_pcs = torch.zeros((pc_embeds.shape))
    #             for j in range(pc.shape[0]):
    #                 pcs = []
    #                 for i in range(3):
    #                     pc_i = pc[j][:, i]
    #                     pcs.append(self.pos_embedding[pc_i])
    #                 pcs = torch.cat(pcs, -1)
    #                 all_pcs[j][:, :1407] = pcs
    #             all_pcs = all_pcs.cuda()

    #         pc_embeds = pc_embeds + 0.01 * all_pcs
    #         image_atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(pc_embeds.device)

    #         query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)  # 768
    #         query_output = self.Qformer.bert(
    #             query_embeds=query_tokens,
    #             encoder_hidden_states=pc_embeds,
    #             encoder_attention_mask=image_atts,
    #             return_dict=True,
    #         )

    #         scene_output = self.t5_proj(query_output.last_hidden_state)
    #         scene_outputs.append(scene_output)

    #     inputs_t5 = torch.cat(scene_outputs, dim=1)
    #     atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(pc_embeds.device)

    #     if self.prompt:
    #         text_input = [self.prompt.format(question) for question in samples["text_input"]]
    #     else:
    #         text_input = samples["text_input"]

    #     with torch.amp.autocast("cuda", dtype=torch.float32):
    #         input_tokens = self.t5_tokenizer(
    #             text_input,
    #             padding="longest",
    #             truncation=True,
    #             max_length=400,
    #             return_tensors="pt",
    #         ).to(pc_embeds.device)
    #         output_tokens = self.t5_tokenizer(
    #             samples["answer"],
    #             padding="longest",
    #             truncation=True,
    #             max_length=300,
    #             return_tensors="pt",
    #         ).to(pc_embeds.device)
    #         batch_input_tokens_input_ids = []
    #         batch_input_tokens_atts = []
    #         batch_atts_t5 = []
    #         batch_inputs_t5 = []

    #         for b, n in enumerate(samples["n_answers"]):
    #             batch_input_tokens_input_ids += [input_tokens.input_ids[b]] * n
    #             batch_input_tokens_atts += [input_tokens.attention_mask[b]] * n
    #             batch_atts_t5 += [atts_t5[b]] * n
    #             batch_inputs_t5 += [inputs_t5[b]] * n

    #         batch_input_tokens_input_ids = torch.stack(batch_input_tokens_input_ids, dim=0)
    #         batch_input_tokens_atts = torch.stack(batch_input_tokens_atts, dim=0)
    #         batch_atts_t5 = torch.stack(batch_atts_t5, dim=0)
    #         batch_inputs_t5 = torch.stack(batch_inputs_t5, dim=0)

    #         encoder_atts = torch.cat([batch_atts_t5, batch_input_tokens_atts], dim=1)

    #         targets = output_tokens.input_ids.masked_fill(
    #             output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
    #         )

    #         inputs_embeds = self.t5_model.encoder.embed_tokens(batch_input_tokens_input_ids)
    #         inputs_embeds = torch.cat([batch_inputs_t5, inputs_embeds], dim=1)
    #         outputs = self.t5_model(
    #             inputs_embeds=inputs_embeds,
    #             attention_mask=encoder_atts,
    #             decoder_attention_mask=output_tokens.attention_mask,
    #             return_dict=True,
    #             labels=targets,
    #         )
    #         loss = outputs.loss
    #         return {"loss": loss}
    
    #### old forward pass
    # def forward(self, samples):
    #     with torch.amp.autocast("cuda", dtype=torch.float32):
    #         pc_embeds = samples["pc_feat"] # B x 5000 x 1408

    #     with torch.amp.autocast("cuda", dtype=torch.float32):
    #         pc = samples["pc"].long() # B x 5000 x 3
    #         all_pcs = torch.zeros((pc_embeds.shape)) # B x 5000 x 1408
    #         for j in range(pc.shape[0]): # iterate over batch
    #             pcs = [] # list for point cloud embeddings
    #             for i in range(3): # iterate over x, y, z
    #                 pc_i = pc[j][:, i] # in jth batch, get 5000 x 1 of all x coords
    #                 pcs.append(self.pos_embedding[pc_i]) # look in pos_embedding dict and append pcs
    #             pcs = torch.cat(pcs, -1) # concatentate across last dim --> 5000 x 3
    #             all_pcs[j][:, :1407] = pcs # how does casting work here?
    #         all_pcs = all_pcs.cuda()

    #     pc_embeds = pc_embeds + 0.01 * all_pcs # add point cloud embeddings
    #     image_atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(pc_embeds.device)

    #     query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)  # 768
    #     query_output = self.Qformer.bert(
    #         query_embeds=query_tokens,
    #         encoder_hidden_states=pc_embeds,
    #         encoder_attention_mask=image_atts,
    #         return_dict=True,
    #     )
    #     inputs_t5 = self.t5_proj(query_output.last_hidden_state)
    #     atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(pc_embeds.device)

    #     if self.prompt:
    #         text_input = [self.prompt.format(question) for question in samples["text_input"]]
    #     else:
    #         text_input = samples["text_input"]

    #     with torch.amp.autocast("cuda", dtype=torch.float32):
    #         input_tokens = self.t5_tokenizer(
    #             text_input,
    #             padding="longest",
    #             truncation=True,
    #             max_length=400,
    #             return_tensors="pt",
    #         ).to(pc_embeds.device)
    #         output_tokens = self.t5_tokenizer(
    #             samples["answer"],
    #             padding="longest",
    #             truncation=True,
    #             max_length=300,
    #             return_tensors="pt",
    #         ).to(pc_embeds.device)
    #         batch_input_tokens_input_ids = []
    #         batch_input_tokens_atts = []
    #         batch_atts_t5 = []
    #         batch_inputs_t5 = []

    #         for b, n in enumerate(samples["n_answers"]):
    #             batch_input_tokens_input_ids += [input_tokens.input_ids[b]] * n
    #             batch_input_tokens_atts += [input_tokens.attention_mask[b]] * n
    #             batch_atts_t5 += [atts_t5[b]] * n
    #             batch_inputs_t5 += [inputs_t5[b]] * n

    #         batch_input_tokens_input_ids = torch.stack(batch_input_tokens_input_ids, dim=0)
    #         batch_input_tokens_atts = torch.stack(batch_input_tokens_atts, dim=0)
    #         batch_atts_t5 = torch.stack(batch_atts_t5, dim=0)
    #         batch_inputs_t5 = torch.stack(batch_inputs_t5, dim=0)

    #         encoder_atts = torch.cat([batch_atts_t5, batch_input_tokens_atts], dim=1)

    #         targets = output_tokens.input_ids.masked_fill(
    #             output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
    #         )

    #         inputs_embeds = self.t5_model.encoder.embed_tokens(batch_input_tokens_input_ids)
    #         inputs_embeds = torch.cat([batch_inputs_t5, inputs_embeds], dim=1)
    #         outputs = self.t5_model(
    #             inputs_embeds=inputs_embeds,
    #             attention_mask=encoder_atts,
    #             decoder_attention_mask=output_tokens.attention_mask,
    #             return_dict=True,
    #             labels=targets,
    #         )
    #         loss = outputs.loss
    #         return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        # why no positional embeddings? --> check generation code

        with torch.amp.autocast("cuda", enabled=(self.device != torch.device("cpu")), dtype=torch.float32):
            pc_embeds = samples["pc_feat"]
        image_atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(pc_embeds.device)

        query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=pc_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(pc_embeds.device)

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        if isinstance(prompt, str):
            prompt = [prompt] * pc_embeds.size(0)
        else:
            assert len(prompt) == pc_embeds.size(0), "The number of prompts must be equal to the batch size."

        input_tokens = self.t5_tokenizer(prompt, padding="longest", return_tensors="pt").to(pc_embeds.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.amp.autocast("cuda", enabled=(self.device != torch.device("cpu")), dtype=torch.float32):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=200,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        repetition_penalty=1.0,
        **kwargs,
    ):
        
        batch_size = samples["pc_feat"].size(0)  # B
        num_scenes = samples["pc_feat"].size(1)  # S
        
        # Reshape pc_feat from [B, S, 5000, 1408] to [B*S, 5000, 1408]
        pc_embeds = samples["pc_feat"].reshape(batch_size * num_scenes, -1, samples["pc_feat"].size(-1))
        
        # Reshape pc from [B, S, 5000, 3] to [B*S, 5000, 3]
        pc = samples["pc"].reshape(batch_size * num_scenes, -1, 3).long()
        
        with torch.amp.autocast("cuda", dtype=torch.float32):
            # Create tensor for positional embeddings, matching pc_embeds shape
            all_pcs = torch.zeros_like(pc_embeds)  # [B*S, 5000, 1408]
            
            # Process each batch item (now including scenes)
            for j in range(pc.shape[0]):  # Iterate over B*S combined batch items
                pcs = []
                for i in range(3):  # Iterate over x, y, z
                    # Extract coordinate values for the j-th batch+scene item
                    pc_i = pc[j][:, i]  # [5000]
                    # Look up positional embeddings (exactly as in original code)
                    pcs.append(self.pos_embedding[pc_i])  # Each has shape [5000, 469]
                
                # Concatenate xyz positional embeddings
                pcs = torch.cat(pcs, -1)  # Shape becomes [5000, 1407]
                
                # Assign to the first 1407 dimensions of all_pcs for this batch+scene
                all_pcs[j][:, :1407] = pcs
            
            all_pcs = all_pcs.cuda()
            pc_embeds = pc_embeds + 0.01 * all_pcs
            image_atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(pc_embeds.device)  # [B*S, 5000]
            
            query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)  # [B*S, 32, 768]
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=pc_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            
            scene_outputs = self.t5_proj(query_output.last_hidden_state)  # [B*S, 32, t5_dim]
        
            # reshape to get the scenes back for each batch [B*S, 32, t5_dim] to [B, S*32, t5_dim]
            t5_dim = scene_outputs.size(-1)
            inputs_t5 = scene_outputs.reshape(batch_size, num_scenes * 32, t5_dim)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(inputs_t5.device)  # [B, S*32]
        
        # with torch.amp.autocast("cuda", enabled=(self.device != torch.device("cpu")), dtype=torch.float32):
        #     pc_embeds = samples["pc_feat"]

        # with torch.amp.autocast("cuda", dtype=torch.float32):
        #     pc = samples["pc"].long()
        #     all_pcs = torch.zeros((pc_embeds.shape))
        #     for j in range(pc.shape[0]):
        #         pcs = []
        #         for i in range(3):
        #             pc_i = pc[j][:, i]
        #             pcs.append(self.pos_embedding[pc_i])
        #         pcs = torch.cat(pcs, -1)
        #         all_pcs[j][:, :1407] = pcs
        #     all_pcs = all_pcs.cuda()

        # pc_embeds = pc_embeds + 0.01 * all_pcs
        # image_atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(pc_embeds.device)

        # query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)
        # query_output = self.Qformer.bert(
        #     query_embeds=query_tokens,
        #     encoder_hidden_states=pc_embeds,
        #     encoder_attention_mask=image_atts,
        #     return_dict=True,
        # )

        # inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        # atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(pc_embeds.device)

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        prompt = self.prompt

        if prompt:
            text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        input_tokens = self.t5_tokenizer(text_input, padding="longest", return_tensors="pt").to(pc_embeds.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
        num_beams = 1
        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.amp.autocast("cuda", enabled=(self.device != torch.device("cpu")), dtype=torch.float32):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                # for description, also use repetition penalty = 1.5
            )
            output_text = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=False)

        if self._apply_lemmatizer:
            output_text_new = self._lemmatize(output_text)
            output_text = output_text_new
            # if output_text_new!=output_text:
            #    print("old: %s, new: %s\n"%(output_text, output_text_new))
        # import pdb; pdb.set_trace()
        return output_text

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)

        return model
    
