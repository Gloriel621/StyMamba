import argparse, os, json, torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import time
import re

import models.mambanet as mambanet
import models.stymamba as StyMamba

"""
Usage example
-------------
python inference_text.py \
        --content_dir  path/to/contents \
        --ckpt         experiments/model_checkpoint.pth \
        --out_dir      results \
        --image_size   512 \
        --hidden_dim   256 \
        --style_texts  "A painting in the style of Van Gogh" "A watercolor painting of a cityscape"
"""

def test_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

class ImgFolder(torch.utils.data.Dataset):
    def __init__(self, root, tfm):
        self.paths = sorted([p for p in Path(root).glob('*') if p.is_file()])
        self.tfm   = tfm
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.tfm(img), self.paths[idx].stem  # (tensor, basename)

def print_param_counts(m: torch.nn.Module, name: str):
    total   = sum(p.numel() for p in m.parameters())
    train   = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"{name:25s} • total: {total/1e6:6.2f}M   trainable: {train/1e6:6.2f}M")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--content_dir', default='./examples/images_content', help="Directory with content images.")
    ap.add_argument('--style_texts', nargs='+', required=True, help="One or more text prompts for styling.")
    ap.add_argument('--ckpt',        required=True, help="Path to the model checkpoint.")
    ap.add_argument('--vgg',         default='./vgg/vgg_normalised.pth')
    ap.add_argument('--image_size', type=int, default=512, choices=[256, 512], help='Image size the model was trained on.')
    ap.add_argument('--hidden_dim', type=int, default=256, choices=[256, 512], help='Hidden dimension of the model.')
    ap.add_argument('--out_dir',     default='./results_text')
    ap.add_argument('--cfg', default='./VMamba2/classification/configs/vssm/vmambav2v_small_224.yaml')
    ap.add_argument('--opts', nargs='*', default=[], help='override yaml options')
    args = ap.parse_args()
    kwargs = vars(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    # load VGG encoder
    vgg = StyMamba.vgg
    vgg.load_state_dict(torch.load(args.vgg, map_location='cpu'))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    # Initialize model components based on arguments
    if args.hidden_dim == 256:
        decoder = StyMamba.decoder256
    elif args.hidden_dim == 512:
        decoder = StyMamba.decoder512
    else:
        raise ValueError(f"No decoder available for hidden_dim: {args.hidden_dim}")

    embedding = StyMamba.PatchEmbed(img_size=args.image_size, embed_dim=args.hidden_dim)
    mamba_dec = mambanet.MambaNet(d_model=args.hidden_dim)
    
    # name_info is not used in text inference, but might be required by the model's constructor
    with open(os.path.join('../wikiart','artist_mapping.json'), 'r', encoding='utf-8') as f:
        name = json.load(f)
    name_info = {int(k): v for k, v in name.items()}
    
    # --- IMPORTANT: Set text_inference=True ---
    net = StyMamba.StyTrans(
        vgg, decoder, embedding, mamba_dec,
        args, name_info, device=device,
        text_inference=True, **kwargs
    )

    # load checkpoint
    ckpt = torch.load(args.ckpt, map_location='cpu')
    net.mambanet.load_state_dict(ckpt['mambanet'])
    net.decode.load_state_dict(ckpt['decoder'])
    net.embedding.load_state_dict(ckpt['embedding'])
    net.clip_model.load_state_dict(ckpt['clip'])

    net.eval().to(device)

    print_param_counts(net, "Full")
    print_param_counts(net.mambanet, "MambaNet")
    print_param_counts(net.decode, "Decoder")
    print_param_counts(net.embedding, "PatchEmbed")
    print_param_counts(net.clip_model, "Mamba-CLIP encoder")

    torch.set_grad_enabled(False)

    # Prepare content images
    ttfm = test_transform(args.image_size)
    content_ds = ImgFolder(args.content_dir, ttfm)

    total_time = 0.0
    total_runs = 0

    for c_img, c_name in tqdm(content_ds, desc='Content images'):
        c_img = c_img.unsqueeze(0).to(device)

        # Loop through text prompts instead of style images
        for s_text in tqdm(args.style_texts, desc=f'Styles for {c_name}', leave=False):
            start = time.time()
            # --- Use the text inference function ---
            out = net.inf_text(c_img, s_text)
            elapsed = time.time() - start
            total_time += elapsed
            total_runs += 1

            # Create a safe filename from the text prompt
            s_name = re.sub(r'[^a-zA-Z0-9_]+', '', s_text.replace(" ", "_"))[:50]
            
            # save output
            save_path = Path(args.out_dir) / f'{c_name}_{s_name}.jpg'
            save_image(out.clamp(0,1), str(save_path))

    # final report
    avg_time = total_time / total_runs if total_runs > 0 else 0.0
    print(f"\nInference finished for {total_runs} image-text pairs.")
    print(f"Total inference time: {total_time:.4f} s")
    print(f"Average per stylization: {avg_time:.4f} s")

if __name__ == '__main__':
    main()