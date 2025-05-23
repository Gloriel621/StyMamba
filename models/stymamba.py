import torch
import torch.nn.functional as F
from torch import nn
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from function import normal
from function import calc_mean_std
from models.ViT_helper import to_2tuple
import clip
from torchvision.transforms import Normalize
from .mamba_clip_models import CLIP, CLIP_VMamba_S

class PatchEmbed(nn.Module):
    """ Image or Text to Patch Embedding with optional patch shuffling """
    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # Image projection layers
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward_image(self, x, clip_model, shuffle_patches=False, project=True):
        """
        Forward pass for image inputs
        """
        B, C, H, W = x.shape

        if project:  # For content input
            x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)

        if shuffle_patches:  # Shuffle patches logic
            x = self.shuffle_patches(x)
            proj_embed = self.proj(x)
            x = F.interpolate(x, size=224, mode='bilinear', align_corners=False)  # Resize for CLIP
            x = clip_model.encode_image(x)  # Get image features from CLIP
            x = x.view(B, 512, 1, 1)  # Reshape for upsampling
            x = self.upsample(x)  # Upsample to match spatial size
            x = self.relu(self.conv(x.float()))  # Apply a convolution and ReLU
            x = x.view(-1, 512, 32, 32) 

            return x + proj_embed

        return x

    def forward_text(self, text_tokens, clip_model: CLIP):
        """
        Forward pass for text inputs
        """
        with torch.no_grad():
            # text_features = clip_model.encode_text(text_tokens)  # Get text features from CLIP
            text_features = clip_model.encode_text_mamba(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize

        # Reshape and process text features
        text_features = text_features.unsqueeze(-1).unsqueeze(-1)  # Reshape to (B, embed_dim, 1, 1)
        x = self.upsample(text_features)  # Upsample to match the spatial size (B, embed_dim, 32, 32)
        x = self.relu(self.conv(x.float()))  # Apply a convolution and ReLU
        return x

    def shuffle_patches(self, x):
        """
        Shuffle the patches in the feature map
        """
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)
        idx = torch.randperm(H * W)  # Generate a random permutation of patch indices
        x = x[:, :, idx]  # Shuffle patches
        x = x.view(B, C, H, W)  # Reshape back to the original dimensions
        return x


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
class StyTrans(nn.Module):
    """ This is the style transform transformer module """
    
    def __init__(self, encoder, decoder, PatchEmbed: PatchEmbed, mambanet, args, name_info, device, text_inference, **kwargs):
        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])
        self.enc_2 = nn.Sequential(*enc_layers[4:11])
        self.enc_3 = nn.Sequential(*enc_layers[11:18])
        self.enc_4 = nn.Sequential(*enc_layers[18:31])
        self.enc_5 = nn.Sequential(*enc_layers[31:44])
        
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss()
        self.mambanet = mambanet      
        self.decode = decoder
        self.embedding = PatchEmbed
        self.name_info = name_info
        self.device=device

        self.clip_model = CLIP_VMamba_S(**kwargs)
        self.clip_model.cuda()

        for module in self.clip_model.modules():
            if isinstance(module, torch.utils.checkpoint.CheckpointFunction):
                module.use_reentrant = False

        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.clip_normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    
    # inference path
    def inf_image(self, samples_c: NestedTensor, samples_s: NestedTensor):

        samples_c = nested_tensor_from_tensor_list(samples_c)
        samples_s = nested_tensor_from_tensor_list(samples_s)
        style = self.embedding.forward_image(samples_s.tensors, self.clip_model, shuffle_patches=True, project=False)
        content = self.embedding.forward_image(samples_c.tensors, self.clip_model)

        hs = self.mambanet(style, None, content, None, None)
        Ics = self.decode(hs)

        return Ics

    # inference path
    def inf_text(self, samples_c: NestedTensor, _text_style: str):
        samples_c = nested_tensor_from_tensor_list(samples_c)
        content = self.embedding.forward_image(samples_c.tensors, self.clip_model)

        text_tokens = clip.tokenize(_text_style).to(samples_c.tensors.device)
        text_style = self.embedding.forward_text(text_tokens, self.clip_model)

        hs = self.mambanet(text_style, None, content, None, None)
        Ics = self.decode(hs)

        return Ics

    # loss computation
    def forward(self, samples_c: NestedTensor, samples_s: NestedTensor, style_labels):
        content_input = samples_c
        style_input = samples_s
        if isinstance(samples_c, (list, torch.Tensor)):
            samples_c = nested_tensor_from_tensor_list(samples_c)
        if isinstance(samples_s, (list, torch.Tensor)):
            samples_s = nested_tensor_from_tensor_list(samples_s)

        content_feats = self.encode_with_intermediate(samples_c.tensors)
        style_feats = self.encode_with_intermediate(samples_s.tensors)

        # Image style
        style = self.embedding.forward_image(samples_s.tensors, self.clip_model, shuffle_patches=True, project=False)

        # Text style
        style_labels_list = style_labels.tolist()
        texts = [self.name_info[label] for label in style_labels_list]
        text_tokens = clip.tokenize(texts).to(samples_c.tensors.device)
        text_style = self.embedding.forward_text(text_tokens, self.clip_model)

        # Content is projection only
        content = self.embedding.forward_image(samples_c.tensors, self.clip_model)

        pos_s = None
        pos_c = None
        mask = None

        hs = self.mambanet(style, mask, content, pos_c, pos_s)
        hs_text = self.mambanet(text_style, mask, content, pos_c, pos_s)

        Ics = self.decode(hs)
        Ics_text = self.decode(hs_text)

        # ----- Loss Computations for Ics -----
        Ics_feats = self.encode_with_intermediate(Ics)
        loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1])) + \
                self.calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))
        loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])

        Icc = self.decode(self.mambanet(content, mask, content, pos_c, pos_c))
        Iss = self.decode(self.mambanet(style, mask, style, pos_s, pos_s))

        loss_lambda1 = self.calc_content_loss(Icc, content_input) + self.calc_content_loss(Iss, style_input)

        Icc_feats = self.encode_with_intermediate(Icc)
        Iss_feats = self.encode_with_intermediate(Iss)
        loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0]) + self.calc_content_loss(Iss_feats[0], style_feats[0])

        for i in range(1, 5):
            loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i]) + self.calc_content_loss(Iss_feats[i], style_feats[i])

        # ----- Loss Computations for Ics_text -----
        Ics_text_feats = self.encode_with_intermediate(Ics_text)
        loss_c_text = self.calc_content_loss(normal(Ics_text_feats[-1]), normal(content_feats[-1])) + \
                    self.calc_content_loss(normal(Ics_text_feats[-2]), normal(content_feats[-2]))
        loss_s_text = self.calc_style_loss(Ics_text_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s_text += self.calc_style_loss(Ics_text_feats[i], style_feats[i])

        Icc_text = self.decode(self.mambanet(content, mask, content, pos_c, pos_c))
        Iss_text = self.decode(self.mambanet(text_style, mask, text_style, pos_s, pos_s))

        loss_lambda1_text = self.calc_content_loss(Icc_text, content_input) + self.calc_content_loss(Iss_text, style_input)

        Icc_text_feats = self.encode_with_intermediate(Icc_text)
        Iss_text_feats = self.encode_with_intermediate(Iss_text)
        loss_lambda2_text = self.calc_content_loss(Icc_text_feats[0], content_feats[0]) + \
                            self.calc_content_loss(Iss_text_feats[0], style_feats[0])

        for i in range(1, 5):
            loss_lambda2_text += self.calc_content_loss(Icc_text_feats[i], content_feats[i]) + \
                                self.calc_content_loss(Iss_text_feats[i], style_feats[i])

        # ----- Combine Losses -----
        combined_loss_c = loss_c + loss_c_text
        combined_loss_s = loss_s + loss_s_text
        combined_loss_lambda1 = loss_lambda1 + loss_lambda1_text
        combined_loss_lambda2 = loss_lambda2 + loss_lambda2_text

        self.clip_model.train()
        for param in self.clip_model.parameters():
            param.requires_grad = True

        # CLIP LOSS
        Ics_clip_in  = self.clip_normalize(Ics.detach())
        Ics_text_in  = self.clip_normalize(Ics_text)

        feat_ics      = self.clip_model.encode_image(Ics_clip_in)
        feat_ics_text = self.clip_model.encode_image(Ics_text_in)

        feat_ics      = feat_ics      / feat_ics.norm(dim=-1, keepdim=True)
        feat_ics_text = feat_ics_text / feat_ics_text.norm(dim=-1, keepdim=True)

        cos_sim      = torch.sum(feat_ics * feat_ics_text, dim=-1)
        alignment_loss = (1.0 - cos_sim).mean()

        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        return Ics, Ics_text, combined_loss_c, combined_loss_s, combined_loss_lambda1, combined_loss_lambda2, alignment_loss
