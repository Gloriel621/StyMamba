import argparse, os, json, torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import time

import models.mambanet as mambanet
import models.stymamba as StyMamba

"""
image_only_inference.py  ──  Mamba style-transfer inference (image → image)

Usage example
-------------
python image_only_inference.py \
       --content_dir  path/to/contents \
       --style_dir    path/to/styles   \
       --ckpt         experiments/model_iter_160000.pth \
       --out_dir      results
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
        return self.tfm(img), self.paths[idx].stem   # (tensor, basename)

def print_param_counts(m: torch.nn.Module, name: str):
    total   = sum(p.numel() for p in m.parameters())
    train   = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"{name:25s} • total: {total/1e6:6.2f}M   trainable: {train/1e6:6.2f}M")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--content_dir', default='/home/gloriel621/StyMamba/examples/images_content')
    ap.add_argument('--style_dir',   default='/home/gloriel621/StyMamba/examples/images_style')
    ap.add_argument('--ckpt',        required=True)
    ap.add_argument('--vgg',         default='./experiments/vgg_normalised.pth')
    ap.add_argument('--image_size', type=int, default=512, choices=[256, 512], help='Image size the model was trained on.')
    ap.add_argument('--out_dir',     default='./results_256')
    ap.add_argument('--cfg',  default='./VMamba2/classification/configs/vssm/vmambav2v_small_224.yaml')
    ap.add_argument('--opts', nargs='*', default=[], help='override yaml options')
    args = ap.parse_args()
    kwargs = vars(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    # load VGG encoder
    vgg = StyMamba.vgg
    vgg.load_state_dict(torch.load(args.vgg, map_location='cpu'))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    hidden_dim = 512
    if args.image_size == 512:
        hidden_dim = 256

    if hidden_dim == 256:
        decoder = StyMamba.decoder256
    elif hidden_dim == 512:
        decoder = StyMamba.decoder512
    else:
        raise ValueError(f"No decoder available for hidden_dim: {hidden_dim}")


    embedding = StyMamba.PatchEmbed(img_size=args.image_size, embed_dim=hidden_dim)
    mamba_dec = mambanet.MambaNet(d_model=hidden_dim)
    name_info = {0: 'unused'}

    # build network
    net = StyMamba.StyTrans(
        vgg, decoder, embedding, mamba_dec,
        args, name_info, device=device,
        text_inference=False, **kwargs
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

    # prepare data
    ttfm = test_transform(args.image_size) # Update this line
    content_ds = ImgFolder(args.content_dir, ttfm)
    style_ds = ImgFolder(args.style_dir, ttfm)

    total_time = 0.0
    total_runs = 0

    for c_img, c_name in tqdm(content_ds, desc='Content images'):
        c_img = c_img.unsqueeze(0).to(device)

        for s_img, s_name in tqdm(style_ds, desc=f'Styles for {c_name}', leave=False):
            s_img = s_img.unsqueeze(0).to(device)
            start = time.time()
            out = net.inf_image(c_img, s_img)
            elapsed = time.time() - start
            total_time += elapsed
            total_runs += 1
            # save output
            save_path = Path(args.out_dir) / f'{c_name}_{s_name}.jpg'
            save_image(out.clamp(0,1), str(save_path))

    # final report
    avg_time = total_time / total_runs if total_runs > 0 else 0.0
    print(f"\nInference finished on {total_runs} image pairs.")
    print(f"Total inference time: {total_time:.4f} s")
    print(f"Average per stylization: {avg_time:.4f} s")

if __name__ == '__main__':
    main()
