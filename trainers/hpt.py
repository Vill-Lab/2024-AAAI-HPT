import os.path as osp

import json
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    model = clip.build_model(state_dict or model.state_dict())

    return model


class VisionEncoderZS(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()

        visual = clip_model.visual
        self.ln_pre = visual.ln_pre
        self.transformer = visual.transformer.resblocks
        self.ln_post = visual.ln_post
        self.proj = visual.proj
        self.dtype = clip_model.dtype
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x).type(self.dtype)
        x = x.permute(1, 0, 2)

        x = self.transformer(x)
        
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        x = x @ self.proj
        
        return x


class VisionEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        visual = clip_model.visual
        self.ln_pre = visual.ln_pre
        self.transformer = visual.transformer.resblocks
        self.ln_post = visual.ln_post
        self.proj = visual.proj
        self.dtype = clip_model.dtype
        self.n_vpro = cfg.TRAINER.HPT.N_VPRO # prompt length

    def forward(self, x, p_visual):
        x = self.ln_pre(x).type(self.dtype)
        x = x.permute(1, 0, 2)

        for layer_idx, layer in enumerate(self.transformer):
            if layer_idx > 0:
                # insert layer-wise global visual prompt
                x[-self.n_vpro:] = p_visual[layer_idx-1].unsqueeze(1).expand(-1, x.shape[1], -1)
            x = layer(x)
            
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        x = x @ self.proj

        return x


class VisionPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.n_vpro = cfg.TRAINER.HPT.N_VPRO
        self.pro_dim = clip_model.visual.ln_pre.weight.shape[0]
        self.dtype = clip_model.dtype
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.layers = len(clip_model.visual.transformer.resblocks)
        # global prompt for image encoder (except for the first layer)
        self.p_visual = nn.ParameterList([nn.Parameter(torch.empty(self.n_vpro, self.pro_dim).type(self.dtype))
                                          for _ in range(self.layers-1)])
        for p in self.p_visual:
            nn.init.normal_(p, std=0.02)
            
        # global prompt for the first layer of image encoder
        self.p_input = nn.Parameter(torch.empty(self.n_vpro, self.pro_dim))
        nn.init.normal_(self.p_input, std=0.02)

    def forward(self, x):
        x = x.type(self.dtype)
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], 
                                                                      dtype=x.dtype, device=x.device), x], dim=1) 
        x = x + self.positional_embedding.to(x.dtype)
        
        # insert global visual prompt of the first layer
        p_input = self.p_input.unsqueeze(0).expand(len(x), -1, -1)
        x = torch.cat([x, p_input], dim=1)

        return x, self.p_visual


class TextEncoderZS(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer.resblocks
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        
        feats = []
        for _, layer in enumerate(self.transformer):
            x = layer(x)
            # save class embeddings from different layers
            feats.append(x[text.argmax(dim=-1), torch.arange(x.shape[1])])

        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        txt_feats = torch.stack(feats)

        return x, txt_feats


class TextEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer.resblocks
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.n_tpro = cfg.TRAINER.HPT.N_TPRO # prompt length
        self.n_set = cfg.TRAINER.HPT.N_SET # number of descriptions for each category

    def forward(self, x, p_ins, p_uni, tokenized_prompts, attn, flag):
        # p_ins: instance-specific prompt, a.k.a high-level prompt from descriptions
        # p_uni: task-unified prompt, a.k.a global-level prompt
        # flag: True when training and False when testing
        # Since we use all (self.n_set) descriptions for learning high-level prompt, we should reshape p_ins first.
        (l, c, d) = p_ins.shape
        p_ins = p_ins.reshape(l, c//self.n_set, self.n_set, d) # (L, C, n_set, D)

        # During evaluation, we leverage all (n_set) structures according to descriptions for modeling one category (N*C*n_set steps in total), 
        # instead of randomly picking one structure for each category (N*C steps in one epoch). 
        if not flag:
            p_ins = p_ins.unsqueeze(2).expand(-1, -1, self.n_set, -1, -1)
            p_ins = torch.flatten(p_ins, 1, 2) # (L, C*n_set, n_set, D)
            
        p_ins = p_ins.permute(0, 2, 1, 3).type(self.dtype)
        x = (x + self.positional_embedding).type(self.dtype)
        x = x.permute(1, 0, 2)

        for layer_idx, layer in enumerate(self.transformer):
            if layer_idx > 0:                
                prefix = x[:1]
                suffix = x[1+self.n_tpro+self.n_set:]
                
                # global-level prompt
                ctx_g = p_uni[layer_idx - 1].unsqueeze(1).expand(self.n_tpro, prefix.shape[1], -1)
                
                # high-level prompt
                ctx_h = p_ins[layer_idx - 1]
                x = torch.cat([prefix, ctx_g, ctx_h, suffix], dim=0)
                
                # 'attn' is attention matrix from topological prompt learner, 
                # considering as low-level prompt which models relationships in an explicit way.
                x = layer(x, attn[:, layer_idx])
            elif layer_idx == 0:
                x = layer(x, attn[:, layer_idx])
            else:
                x = layer(x)

        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        if not flag:
            x = x.reshape(x.shape[0]//5, 5, -1)

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, info_topo, clip_model):
        super().__init__()
        self.n_tpro = cfg.TRAINER.HPT.N_TPRO # prompt length
        self.n_set = cfg.TRAINER.HPT.N_SET # number of descriptions for each category
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.layers = len(clip_model.transformer.resblocks)

        # global prompt for text encoder (except for the first layer)
        self.p_uni = nn.ParameterList([nn.Parameter(torch.empty(self.n_tpro, self.ctx_dim).type(self.dtype))
                                                      for _ in range(self.layers - 1)])
        for p in self.p_uni:
            nn.init.normal_(p, std=0.02)
            
        # projector for learning high-level prompt (a.k.a p_ins)
        self.p_ins_projector = nn.Linear(self.ctx_dim, self.ctx_dim)
        
        # global prompt for the first layer of the text encoder
        self.p_input = nn.Parameter(torch.empty(self.n_tpro+self.n_set, self.ctx_dim))
        nn.init.normal_(self.p_input, std=0.02)
        
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.info_topo = info_topo # topological structure in a dictionary form
        self.n_cls = len(classnames)
        self.clip_model = clip_model
    def forward(self, feats, attns, flag):
        p_uni = self.p_uni
        prompts, attn = [], []
        prompt_prefix = " ".join(["X"] * (self.n_tpro+self.n_set))

        if flag:
            for name in self.classnames:
                # For efficiency, we randomly pick one structure as a part of input during training, 
                # while leveraging all descriptions of the category for learning high-level prompt.
                id = random.randint(0, self.n_set-1)
                topo = self.info_topo[name][id]
                p = prompt_prefix + " " + name + ". " + ", ".join(topo['Entities']) + ". " + ", ".join(topo['Attributes']) + "."
                attn.append(attns[name][:, id])
                prompts.append(p)
        else:
            for name in self.classnames:
                # We leverage all structures from descriptions as a part of input respectively during evaluation.
                for id in range(self.n_set):
                    topo = self.info_topo[name][id]
                    p = prompt_prefix + " " + name + ". " + ", ".join(topo['Entities']) + ". "  + ", ".join(topo['Attributes']) + "." 
                    attn.append(attns[name][:, id])
                    prompts.append(p)
        
        attn = torch.stack(attn, dim=0)
            
        self.tokenized_prompts = torch.cat([clip.tokenize(p, truncate=True) for p in prompts]).cuda()  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(self.tokenized_prompts).type(self.dtype)
        
        p_input = self.p_input.unsqueeze(0).expand(len(prompts), -1, -1)
        prefix = embedding[:, :1]
        suffix = embedding[:, 1+self.n_tpro+self.n_set:]
        
        # the input of the prompted text encoder
        p_ori = torch.cat([prefix, p_input, suffix], dim=1)

        # generate corresponding high-level prompt (p_ins)
        p_ins = []
        (l, c, n, d) = feats.shape
        feats = feats.reshape(l, c*n, d)
        for idx in range(self.layers - 1):
            feat = feats[idx].float()
            feat = feat + self.p_ins_projector(feat) 
            p_ins.append(feat)
        p_ins = torch.stack(p_ins, dim=0)

        return p_ori, p_ins, p_uni, attn

    
class TopoPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, prompt_topo, clip_model):
        super().__init__()

        self.classnames = classnames
        self.dtype = clip_model.dtype
        self.n_set = cfg.TRAINER.HPT.N_SET # number of descriptions for each category
        self.n_tpro = cfg.TRAINER.HPT.N_TPRO # prompt length
        self.layers = len(clip_model.transformer.resblocks)
        
        # layer-wise scalar to weight indicating the strength of the relationship of entity-entity pairs and entity-attribute pairs
        self.e2e_scal = nn.Parameter(torch.zeros(self.layers, 1, 1, 1)) 
        self.e2a_scal = nn.Parameter(torch.zeros(self.layers, 1, 1, 1))
        
        self.attns_e2e = {classname: [] for classname in classnames}
        self.attns_e2a = {classname: [] for classname in classnames}

        prompt_prefix = " ".join(["X"] * (self.n_tpro + self.n_set))

        for classname in classnames:
            topos = prompt_topo[classname]
            for id in range(self.n_set):
                # generate text with classname, entities and attributes
                txt = self.generate_text(classname, prompt_prefix, topos[id])
                tokens = clip.tokenize(txt, truncate=True)[0]
                
                # generate pair-wise relationships
                e2e, e2a = self.extract_relationships(tokens, topos[id])

                # create attention matrix based on pair-wise relationships
                attn_e2e = self.create_attention_matrix(tokens, e2e)
                attn_e2a = self.create_attention_matrix(tokens, e2a)

                # save attention matrices
                self.attns_e2e[classname].append(attn_e2e)
                self.attns_e2a[classname].append(attn_e2a)

    # generate text with classname, entities and attributes
    def generate_text(self, classname, prompt_prefix, topo):
        entities = [w.lower() for w in topo['Entities']]
        attributes = [w.lower() for w in topo['Attributes']]
        txt = prompt_prefix + " " + classname + ". " + ", ".join(entities) + ". " + ", ".join(attributes) + "."
        return txt

    # generate pair-wise relationships from topological structure
    def extract_relationships(self, tokens, topo):
        entities = [w.lower() for w in topo['Entities']]
        attributes = [w.lower() for w in topo['Attributes']]
        e2e, e2a = [], []

        for w in topo['Entity-to-Entity Relationships']:
            if w['entity1'].lower() in entities and w['entity2'].lower() in entities:
                e1 = list(self.align(tokens, self.truncate(clip.tokenize(w['entity1']))[0]))
                e2 = list(self.align(tokens, self.truncate(clip.tokenize(w['entity2']))[0]))
                e2e.append([e1, e2])

        for w in topo['Entity-to-Attribute Relationships']:
            if w['entity'].lower() in entities and w['attribute'].lower() in attributes:
                e1 = list(self.align(tokens, self.truncate(clip.tokenize(w['entity']))[0]))
                e2 = list(self.align(tokens, self.truncate(clip.tokenize(w['attribute']))[0]))
                e2a.append([e1, e2])

        return e2e, e2a

    # create attention matrix based on pair-wise relationships
    def create_attention_matrix(self, tokens, relationships):
        n_tokens = len(tokens)
        attn = torch.zeros(n_tokens, n_tokens).cuda()

        for e in relationships:
            d11 = torch.tensor([[i] for i in e[0]]).type(torch.long)
            d21 = torch.tensor([e[1] for _ in range(len(e[0]))]).type(torch.long)
            d12 = torch.tensor([[i] for i in e[1]]).type(torch.long)
            d22 = torch.tensor([e[0] for _ in range(len(e[1]))]).type(torch.long)
            attn[d11, d21] += 1
            attn[d12, d22] += 1

        return attn

    # truncate token sequence according to EOS token
    def truncate(self, array):
        return array[:, 1:torch.argmax(array)]

    # find a sequence that matches the target token(s)
    def align(self, seq1, seq2):
        for idx in range(len(seq1) - len(seq2) + 1):
            if seq1[idx:idx + len(seq2)].equal(seq2):
                return range(idx, idx + len(seq2))
        return []

    def forward(self):
        attns = {}
        for classname in self.classnames:
            classname = classname.replace("_", " ")
            # weight generated matrices with two learnable scalars
            attns[classname] = self.e2e_scal * torch.stack(self.attns_e2e[classname]).cuda() + \
                               self.e2a_scal * torch.stack(self.attns_e2a[classname]).cuda()
        return attns



class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        for p in clip_model.parameters():
            p.requires_grad = False

        if 'ImageNet' not in cfg.DATASET.NAME:
            dname = cfg.DATASET.NAME
        else:
            dname = 'ImageNet'
            
        f_json = osp.join(cfg.DATASET.GPT_DIR+'/description', dname+'.json')
        f_topo = osp.join(cfg.DATASET.GPT_DIR+'/structure', dname+'.json')
        
        with open(f_json, 'r') as f:
            text_prompts = json.load(f)
        with open(f_topo, 'r') as f:
            text_topos = json.load(f)

        classnames = [name.replace("_", " ") for name in classnames]
        self.topo_prompt_learner = TopoPromptLearner(cfg, classnames, text_topos, clip_model)
        self.prompt_learner = PromptLearner(cfg, classnames, text_topos, clip_model)
        self.vision_prompt_learner = VisionPromptLearner(cfg, clip_model)
        self.image_encoder = VisionEncoder(cfg, clip_model)
        self.text_encoder = TextEncoder(cfg, clip_model)
        self.text_encoder_zs = TextEncoderZS(cfg, clip_model)
        self.image_encoder_zs = VisionEncoderZS(cfg, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.model = clip_model

        with torch.no_grad():
            # zs_feats: layer-wise class embeddings from frozen text encoder
            # zs_repres: final representations from frozen text encoder
            zs_feats, zs_repres = [], []
            for classname in classnames:
                texts = text_prompts[classname]
                texts = clip.tokenize(texts).cuda()
                class_embeddings, features = self.text_encoder_zs(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                features /= features.norm(dim=-1, keepdim=True)
                zs_feats.append(features)
                zs_repres.append(class_embedding)
            self.text_features_zs = torch.stack(zs_repres, dim=1).cuda()
            self.text_features_ft = torch.stack(zs_feats, dim=1).cuda()


    def forward(self, image):
        logit_scale = self.logit_scale.exp()
        
        text_features_zs = self.text_features_zs
        image_features_zs = self.image_encoder_zs(image.type(self.dtype))
        image_features_zs = image_features_zs / image_features_zs.norm(dim=-1, keepdim=True)
        
        attns = self.topo_prompt_learner()
        p_ori, p_ins, p_uni, attns = self.prompt_learner(self.text_features_ft, attns, self.training)

        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(p_ori, p_ins, p_uni, tokenized_prompts, attns, self.training)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Since we use multiple structures for producing representations of one category, 
        # we should take their mean value as the final representation.
        if not self.training:
            text_features = text_features.mean(dim=1)
        
        x, p_visual = self.vision_prompt_learner(image)
        image_features = self.image_encoder(x, p_visual)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # asymmetric loss
        logits_i = logit_scale * (image_features @ text_features_zs)
        logits_t = logit_scale * (image_features_zs @ text_features.t())
        logits = (logits_i + logits_t)/2

        if self.training:
            return logits, logits_i, logits_t
        else:
            return logits


@TRAINER_REGISTRY.register()
class HPT(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.HPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.n_class = len(self.dm.dataset.classnames)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg).cuda()

        if cfg.TRAINER.HPT.PREC == "fp32" or cfg.TRAINER.HPT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")

        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("Model", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.HPT.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        logits, logits_i, logits_t = self.model(image)
        loss = F.cross_entropy(logits, label)
        loss_i = F.cross_entropy(logits_i, label)
        loss_t = F.cross_entropy(logits_t, label)
        loss = loss + loss_i + loss_t

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
