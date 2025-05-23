import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import models.mambanet as mambanet
import models.stymamba as StyMamba

def get_transform(size=256):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Text-guided style transfer on a single content image")
    parser.add_argument('--content_image', type=str, required=True, help='Path to the content image')
    parser.add_argument('--style_text', type=str, default="mondrian", help='Text description of the desired style')
    parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')
    parser.add_argument('--ckpt', type=str, default="./experiments/model_iter_160000.pth")
    parser.add_argument('--out_dir', type=str, default="/results_text")
    parser.add_argument('--out_name', type=str, default=None, help='Basename for the output file (without extension).')
    parser.add_argument('--size', type=int, default=256, help='Resize the content image to size×size before inference')
    parser.add_argument('--cfg', type=str, default='./VMamba2/classification/configs/vssm/vmambav2v_small_224.yaml')
    parser.add_argument('--opts', nargs='*', default=[])
    args = parser.parse_args()
    kwargs = vars(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prepare output folder
    os.makedirs(args.out_dir, exist_ok=True)

    # Load VGG encoder (first 44 layers)
    vgg = StyMamba.vgg
    vgg.load_state_dict(torch.load(args.vgg, map_location='cpu'))
    vgg = nn.Sequential(*list(vgg.children())[:44]).to(device)

    # Build style-transfer network
    decoder   = StyMamba.decoder.to(device)
    embedding = StyMamba.PatchEmbed().to(device)
    mamba_dec = mambanet.MambaNet().to(device)
    name_info = {0: args.style_text}

    net = StyMamba.StyTrans(
        vgg, decoder, embedding, mamba_dec,
        args, name_info, device,
        text_inference=True, **kwargs
    )

    # Load checkpoint weights
    ckpt = torch.load(args.ckpt, map_location='cpu')
    net.mambanet.load_state_dict(ckpt['mambanet'])
    net.decode.load_state_dict(ckpt['decoder'])
    net.embedding.load_state_dict(ckpt['embedding'])
    net.clip_model.load_state_dict(ckpt['clip'])

    net.eval().to(device)
    torch.set_grad_enabled(False)

    # Load and preprocess the content image
    tfm = get_transform(args.size)
    img = Image.open(args.content_image).convert('RGB')
    content = tfm(img).unsqueeze(0).to(device)

    # Run style transfer
    out = net.inf_text(content, args.style_text)

    # Determine output filename
    if args.out_name:
        base = args.out_name
    else:
        base = os.path.splitext(os.path.basename(args.content_image))[0]
    out_filename = base + '.jpg'
    out_path = os.path.join(args.out_dir, out_filename)

    # Save result
    save_image(out.clamp(0,1), out_path)
    print(f"Saved styled image to {out_path}")
