import os.path as osp
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.distributions import Normal

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import matplotlib.pyplot as plt

from utils_tip import * 
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
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

class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.activate = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h_ = self.activate(self.FC_input(x))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)                     
                                                       
        return mean, log_var
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_output = nn.Linear(latent_dim,output_dim)
        
        self.activate = nn.LeakyReLU(0.2)
    def forward(self, x):
        x_hat = self.activate(self.FC_output(x))
        return x_hat

class VAE(nn.Module):
    def __init__(self, x_dim,hidden_dim,latent_dim,device):
        super(VAE, self).__init__()
        self.Encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.Decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)
        self.latent_dim = latent_dim
        self.device = device
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)     
        z = mean + var*epsilon
        return z
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.Decoder(z)
        loss = self.loss_function(x,x_hat,mean,log_var)
        
        return x_hat,loss

    def loss_function(self,x, x_hat, mean, log_var):
        reproduction_loss = (x_hat - x).pow(2).sum(1).mean()
        KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD


class UpdatedModel(nn.Module):
    def __init__(self,device):
        super().__init__()

        ctx_dim=1024

        self.vae = VAE(ctx_dim,ctx_dim,128,device)
        
        self.linear = nn.Linear(2*ctx_dim,ctx_dim)
        self.activate = nn.GELU()
    
    def forward_vae(self,x):
        x,loss = self.vae(x)
        return x,loss
    
    def forward_text(self,x):
        x = self.activate(self.linear(x))
        return x

class CustomCLIP(nn.Module):
    def __init__(self,
                 classnames,
                 templates,
                 clip_model,
                 device="cuda",
                 seed=1):
        super().__init__()

        self.model = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = device
        self.seed = seed
        self.classnames = classnames
        self.templates = templates

        self.updated_model = UpdatedModel(self.device)

        for param in self.model.parameters():
            param.requires_grad = False

        class_list = []
        for name in self.classnames:
            class_list.append(clip.tokenize(name))
        self.class_features = torch.stack(class_list,dim=0).squeeze(1).to(device) # n_cls,len

    def update_prompts(self, epoch):

        original_seed = random.getstate()
        epoch_seed = self.seed + epoch
        random.seed(epoch_seed)
        selected_templates = random.sample(self.templates,50)

        templates = []
        for template in selected_templates:
            templates.append(clip.tokenize(template))
            
        random.setstate(original_seed)
        templates_feat = torch.stack(templates,dim=0).to(self.device).squeeze(1) # n_p,len

        return templates_feat
    
    def forward(self, image,labels=None,epoch=None):

        image_features = self.model.encode_image(image.type(self.dtype))
        image_features /= image_features.norm(dim=-1, keepdim=True) # bs,dim 32,1024

        if epoch is not None:
            templates = self.update_prompts(epoch) # n_p,len
        else:
            templates = torch.tensor(clip.tokenize(self.templates),device=self.device)

        template_features_old = self.model.encode_text(templates) 
        # vae
        template_features,vae_loss = self.updated_model.forward_vae(template_features_old)
        
        n_p = template_features.shape[0]

        class_features = self.model.encode_text(self.class_features)
        n_cls = class_features.shape[0]

        template_features = template_features.unsqueeze(0).repeat(n_cls, 1 , 1) # n_cls,n_p,1024
        class_features = class_features.unsqueeze(1).repeat(1, n_p, 1)
        text_features = torch.cat([template_features,class_features],dim=-1)# n_cls,n_p,2048
        text_features = self.updated_model.forward_text(text_features).view(-1,1024)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t() # bs,n_cls*n_p

        if labels is not None:
            
            loss = 0
            logits = logits.view(-1,n_cls,n_p)
            for i in range(n_p):
                prompt_logits = logits[:,:,i]
                loss += F.cross_entropy(prompt_logits,labels)

            return loss,vae_loss

        return logits

@TRAINER_REGISTRY.register()
class PromptCLIP(TrainerX):

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(classnames,                               
                                cfg.TRAINER.PROMPTCLIP.TEMPLATES,
                                clip_model,
                                device=self.device,
                                )
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f'Layer: {name}, Shape: {param.shape}, Requires Grad: True')

        print("Turning off gradients in both the image and the text encoder")

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.updated_model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model.updated_model, self.optim, self.sched)

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        total = 0
        correct = 0
    
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

            predicted = torch.argmax(output, dim=1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

        results = self.evaluator.evaluate()

        high_acc = round(correct / total, 8)
        print(f"Accuracy: {high_acc}")

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def forward_backward(self, batch):

        image, label = self.parse_batch_train(batch)
        loss,vae_loss = self.model(image,label,self.epoch)

        all_loss = loss+vae_loss
        self.model_backward_and_update(all_loss)

        loss_summary = {
            "loss": loss.item(),
            "vae_loss":vae_loss.item()
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

            # Ignore fixed token vectors
            if 'token_prefix' in state_dict:
                del state_dict['token_prefix']
            
            if 'token_suffix' in state_dict:
                del state_dict['token_suffix']

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
