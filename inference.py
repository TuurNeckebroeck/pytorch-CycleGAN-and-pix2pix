import torch
from models import networks
from models.conditional_cycle_gan_model import ConditionalCycleGANModel
import os
from PIL import Image
import torchvision.transforms as transforms
from util import util
from argparse import ArgumentParser

def load_generator_model(path, device):
    state_dict_gen = torch.load(path, map_location=device)
    gen = networks.define_G(3+12, 3, 64, 'resnet_9blocks', norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]).cuda()
    if isinstance(gen, torch.nn.DataParallel):
        gen = gen.module
    if hasattr(state_dict_gen, '_metadata'):
        del state_dict_gen._metadata
    gen.load_state_dict(state_dict_gen)

    return gen

def transform(img):
    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Resize([256,256], Image.BICUBIC)]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    
    return transforms.Compose(transform_list)(img)

def inference(model, img, color):
    img = transform(img).cuda()
    
    color_embedding = ConditionalCycleGANModel.onehot_encode_colors(torch.Tensor([color]).detach()).cuda()

    with torch.no_grad():
        fake_img = model.forward(torch.cat([img.reshape((1, *img.shape)), color_embedding.detach()], dim=1))  # G(A)
        return util.tensor2im(fake_img.detach())
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained generator model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to folder containing images")
    parser.add_argument("--out", type=str, required=True, help="Path to output folder")
    args = parser.parse_args()
    
    assert os.path.isdir(args.dataset), "Couldn't find dataset at {}".format(args.dataset)
    assert os.path.isdir(args.out), "Couldn't find output folder at {}".format(args.out)    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   
    
    gen = load_generator_model(args.model_path, device)
    gen.eval()

    img_paths = [os.path.join(args.dataset, fn) for fn in os.listdir(args.dataset)]
    for filename in img_paths:
        img = Image.open(filename).convert('RGB')

        for color in range(3):
            img_out = inference(gen, img, color)
            path_img_colored = os.path.join(args.out, os.path.basename(filename.replace(".png",f"_{color}.png")))
            # print(f"Writing file {path_img_colored}")

            img.save(os.path.join(args.out, os.path.basename(filename.replace(".png",f"_orig.png"))))
            util.save_image(img_out, path_img_colored, aspect_ratio=1.0)
