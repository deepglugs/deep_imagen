import math
import numbers
import os
import random
import time
from collections import deque

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as VTF
from torchvision.utils import make_grid, save_image
from PIL import Image

from imagen_pytorch import ImagenTrainer, ElucidatedImagenConfig
from imagen_pytorch import load_imagen_from_checkpoint
from gan_utils import get_images, get_vocab
from data_generator import ImageLabelDataset


def get_padding(image):    
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class PadImage(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return VTF.pad(img, get_padding(img), self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=None, help="image source")
    parser.add_argument('--tags_source', type=str, default=None, help="tag files. will use --source if not specified.")
    parser.add_argument('--poses', type=str, default=None)
    parser.add_argument('--tags', type=str, default=None)
    parser.add_argument('--vocab', default=None)
    parser.add_argument('--size', default=256, type=int)
    parser.add_argument('--sample_steps', default=32, type=int)
    parser.add_argument('--num_unets', default=1, type=int, help="additional unet networks")
    parser.add_argument('--vocab_limit', default=None, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--imagen', default="imagen.pth")
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--replace', action='store_true', help="replace the output file")
    parser.add_argument('--unet_dims', default=128, type=int)
    parser.add_argument('--unet2_dims', default=64, type=int)
    parser.add_argument("--start_size", default=64, type=int)

    # training
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--micro_batch_size', default=8, type=int)
    parser.add_argument('--samples_out', default="samples")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--shuffle_tags', action='store_true')
    parser.add_argument('--train_unet', type=int, default=1)
    parser.add_argument('--random_drop_tags', type=float, default=0.)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--no_text_transform', action='store_true')
    parser.add_argument('--aug', action='store_true', help="additional image augmentations")

    args = parser.parse_args()

    if args.sample_steps is None:
        args.sample_steps = args.size

    if args.tags_source is None:
        args.tags_source = args.source

    if args.vocab is None:
        args.vocab = args.source
    else:
        assert os.path.isfile(args.vocab)

    if args.train:
        train(args)
    else:
        sample(args)

def sample(args):

    if os.path.isfile(args.output) and not args.replace:
        return

    try:
        imagen = load(args.imagen)
    except:
        print(f"Error loading model: {args.imagen}")
        return

    cond_image = None

    if args.poses is not None and os.path.isfile(args.poses):
        tforms = transforms.Compose([PadImage(),
                                     transforms.Resize((args.size, args.size)),
                                     transforms.ToTensor()])
        cond_image = Image.open(args.poses)
        cond_image = tforms(cond_image).to(imagen.device)
        cond_image = cond_image.view(1, *cond_image.shape)

    sample_images = imagen.sample(texts=[args.tags],
                                  cond_images=cond_image,
                                  cond_scale=7.,
                                  return_pil_images=True)

    sample_images[0].save(args.output)

def restore_parts(state_dict_target, state_dict_from):
    for name, param in state_dict_from.items():
        if name not in state_dict_target:
            continue
        # if isinstance(param, Parameter):
        #    param = param.data
        if param.size() == state_dict_target[name].size():
            state_dict_target[name].copy_(param)
        else:
            print(f"layer {name}({param.size()} different than target: {state_dict_target[name].size()}")

    return state_dict_target

def save(imagen, path):
    out = {}
    unets = []
    for unet in imagen.unets:
        unets.append(unet.cpu().state_dict())
    out["unets"] = unets

    out["imagen"] = imagen.cpu().state_dict()

    torch.save(out, path)


def load(path):

    imagen = load_imagen_from_checkpoint(path)

    return imagen


def get_imagen(args, unet_dims=None, unet2_dims=None):

    if unet_dims is None:
        unet_dims = args.unet_dims

    if unet2_dims is None:
        unet2_dims = args.unet2_dims

    if args.poses is not None:
        cond_images_channels = 3
    else:
        cond_images_channels = 0

    # unet for imagen
    unet1 = dict(
        dim=unet_dims,
        cond_dim=512,
        dim_mults=(1, 2, 3, 4),
        cond_images_channels=cond_images_channels,
        num_resnet_blocks=3,
        layer_attns=(False, True, True, True),
        memory_efficient=False
    )

    unets = [unet1]

    for i in range(args.num_unets):

        unet2 = dict(
            dim=unet2_dims // (i + 1),
            cond_dim=512,
            dim_mults=(1, 2, 3, 6),
            cond_images_channels=cond_images_channels,
            num_resnet_blocks=(2, 4, 8, 8),
            layer_attns=(False, False, False, True),
            layer_cross_attns=(False, False, True, True),
            final_conv_kernel_size=1,
            memory_efficient=True
        )

        unets.append(unet2)

    image_sizes = [args.start_size]

    for i in range(0, len(unets)-1):
        image_sizes.append(image_sizes[-1] * 4)

    print(f"image_sizes={image_sizes}")

    sample_steps = [args.sample_steps] * (args.num_unets + 1)

    imagen = ElucidatedImagenConfig(
        unets=unets,
        text_encoder_name='t5-large',
        num_sample_steps=sample_steps,
        # noise_schedules=["cosine", "cosine"],
        # pred_objectives=["noise", "x_start"],
        image_sizes=image_sizes,
        per_sample_random_aug_noise_level=True,
        lowres_sample_noise_level=0.3
    ).create().cuda()

    return imagen


def make_training_samples(poses, trainer, args, epoch, step):
    sample_texts = ['1girl, red_bikini, bikini, outdoors, pool, brown_hair',
                    '1girl, blue_dress, eyes_closed, blonde_hair',
                    '1boy, black_hair',
                    '1girl, wristwatch, blue_hair']

    disp_size = min(args.batch_size, 4)
    sample_poses = None

    if poses is not None:
        sample_poses = poses[:disp_size]


    if poses is not None and sample_poses is None:
        sample_poses = poses[:disp_size]

    sample_images = trainer.sample(texts=sample_texts,
                                   cond_images=sample_poses,
                                   cond_scale=7.,
                                   return_all_unet_outputs=True,
                                   stop_at_unet_number=args.train_unet)

    final_samples = None

    if len(sample_images) > 1:
        for si in sample_images:
            sample_images1 = transforms.Resize(args.size)(si)
            if final_samples is None:
                final_samples = sample_images1
                continue

            sample_images1 = transforms.Resize(args.size)(si)
            final_samples = torch.cat([final_samples, sample_images1])
        
        sample_images = final_samples
    else:
        sample_images = sample_images[0]
        sample_images = transforms.Resize(args.size)(sample_images)

    if poses is not None:
        sample_poses0 = transforms.Resize(args.size)(sample_poses)
        sample_images = torch.cat([sample_images.cpu(), sample_poses0.cpu()])

    grid = make_grid(sample_images, nrow=disp_size, normalize=False, range=(-1, 1))
    VTF.to_pil_image(grid).save(os.path.join(args.samples_out, f"imagen_{epoch}_{int(step / epoch)}.png"))


def train(args):

    imagen = get_imagen(args)

    trainer = ImagenTrainer(imagen, fp16=args.fp16)
    
    if args.imagen is not None and os.path.isfile(args.imagen):
        print(f"Loading model: {args.imagen}")
        trainer.load(args.imagen)

    print("Fetching image indexes...")

    imgs = get_images(args.source, verify=False)
    txts = get_images(args.tags_source, exts=".txt")
    # vocab = get_vocab(args.vocab, top=args.vocab_limit)

    poses = None
    has_poses = False

    if args.poses is not None:
        poses = get_images(args.poses)
        has_poses = True

    tforms = transforms.Compose([
            PadImage(),
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor()])

    if args.aug:
        tforms = transforms.Compose([
            PadImage(),
            transforms.RandomCrop(args.size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.ToTensor()])


    def txt_xforms(txt):
        # print(f"txt: {txt}")
        txt = txt.split(", ")
        if args.shuffle_tags:
            np.random.shuffle(txt)

        r = int(len(txt) * args.random_drop_tags)

        if r > 0:
            rand_range = random.randrange(r)

        if args.random_drop_tags > 0.0 and r > 0:
            txt.pop(rand_range)

        txt = ", ".join(txt)

        return txt

    tag_transform = txt_xforms

    if args.no_text_transform:
        tag_transform = None

    data = ImageLabelDataset(imgs, txts, None,
                             poses=poses,
                             dim=(args.size, args.size),
                             transform=tforms,
                             tag_transform=tag_transform,
                             channels_first=True,
                             return_raw_txt=True,
                             no_preload=True)

    dl = torch.utils.data.DataLoader(data,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.workers)
    
    rate = deque([1], maxlen=5)

    os.makedirs(args.samples_out, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        step = 0
        for data in dl:

            if has_poses:
                images, texts, poses = data
            else:
                images, texts = data
                poses = None

            step += 1

            t1 = time.monotonic()
            losses = []

            loss = trainer(
                images,
                cond_images=poses,
                texts=texts,
                unet_number=args.train_unet,
                max_batch_size=args.micro_batch_size
            )

            trainer.update(unet_number=args.train_unet)

            losses.append(loss)

            t2 = time.monotonic()
            rate.append(round(1.0 / (t2 - t1), 2))

            if step % 10 == 0:
                print("epoch {}/{} step {}/{} loss: {} - {}it/s".format(
                      epoch,
                      args.epochs,
                      step * args.batch_size,
                      len(imgs),
                      round(np.sum(losses), 5),
                      round(np.mean(rate), 2)))

            if step % 100 == 0:
                make_training_samples(poses, trainer, args, epoch, step)

                if args.imagen is not None:
                    trainer.save(args.imagen)    

        # END OF EPOCH
        make_training_samples(poses, trainer, args, epoch, step)

        if args.imagen is not None:
            trainer.save(args.imagen)


if __name__ == "__main__":
    main()
