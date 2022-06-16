import math
import os
import time
from collections import deque

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as VTF
from torchvision.utils import make_grid, save_image

from imagen_pytorch import Unet, Imagen, ImagenTrainer
from gan_utils import get_images, get_vocab
from data_generator import ImageLabelDataset

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=None)
    parser.add_argument('--tags_source', type=str, default=None)
    parser.add_argument('--tags', type=str, default=None)
    parser.add_argument('--vocab', default=None)
    parser.add_argument('--size', default=128, type=int)
    parser.add_argument('--num_unets', default=1, type=int, help="additional unet networks")
    parser.add_argument('--vocab_limit', default=None, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--imagen', default="imagen.pth")
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--replace', action='store_true', help="replace the output file")
    parser.add_argument('--unet_dims', default=32, type=int)
    # training
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--micro_batch_size', default=8, type=int)
    parser.add_argument('--samples_out', default="samples")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--shuffle_tags', action='store_true')
    parser.add_argument('--dp', action='store_true')

    args = parser.parse_args()

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

    imagen = get_imagen(args)

    if os.path.isfile(args.output) and not args.replace:
        return

    try:
        imagen = load(imagen, args.imagen)
    except:
        print(f"Error loading model: {args.imagen}")
        return

    sample_images = imagen.sample(texts=[args.tags],
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


def load(imagen, path):

    to_load = torch.load(path)

    try:
        imagen.load_state_dict(to_load["model"], strict=True)
    except RuntimeError:
        print("Failed loading state dict. Trying partial load")
        imagen.load_state_dict(restore_parts(imagen.state_dict(),
                                             to_load))

    return imagen


def get_imagen(args):
    unet1 = Unet(
    # unet for imagen
        dim = args.unet_dims,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 3,
        layer_attns = (False, True, True, True)
    )

    unets = [unet1]

    for i in range(args.num_unets):

        unet2 = Unet(
            dim = 32,
            cond_dim = 512,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = (2, 4, 8, 8),
            layer_attns = (False, False, False, True),
            layer_cross_attns = (False, False, False, True),
            memory_efficient=False
        )

        unets.append(unet2)

    image_sizes = []

    for i in range(len(unets)-1):
        image_sizes.append(int(math.pow(2, 6 + i)))

    image_sizes.append(args.size)

    imagen = Imagen(
        unets = unets,
        continuous_times=True,
        p2_loss_weight_gamma=1.0,
        text_encoder_name='t5-large',
        image_sizes=image_sizes,
        timesteps=1000
    ).cuda()

    return imagen


def train(args):

    imagen = get_imagen(args)

    trainer = ImagenTrainer(imagen)

    if args.dp:
        imagen = torch.nn.DataParallel(trainer)

    if args.imagen is not None and os.path.isfile(args.imagen):
        print(f"Loading model: {args.imagen}")
        trainer.load(args.imagen)

    imgs = get_images(args.source, verify=False)
    txts = get_images(args.tags_source, exts=".txt")
    vocab = get_vocab(args.vocab, top=args.vocab_limit)

    tforms = transforms.Compose([
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor()])
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation((-180, 180)),
            # transforms.ColorJitter(0.15, 0.15, 0.15, 0.15),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5,)*3, (0.5,)*3)])

    def txt_xforms(txt):
        # print(f"txt: {txt}")
        txt = txt.split(", ")
        if args.shuffle_tags:
            np.random.shuffle(txt)

        txt = ", ".join(txt)

        return txt

    data = ImageLabelDataset(imgs, txts, vocab,
                             dim=(args.size, args.size),
                             transform=tforms,
                             tag_transform=txt_xforms,
                             channels_first=True,
                             return_raw_txt=True,
                             no_preload=True)

    dl = torch.utils.data.DataLoader(data,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=8)


    disp_size = 4
    rate = deque([1], maxlen=5)

    os.makedirs(args.samples_out, exist_ok=True)

    sample_texts=['1girl, red_bikini, outdoors, pool',
                  '2girls, blue_dress, eyes_closed, off_shoulder',
                  '1girl, 1boy, long_hair, breasts, red_lingerie',
                  '1girl, from_behind, wristwatch']


    for epoch in range(1, args.epochs + 1):
        step = 0
        for images, texts in dl:
            step += 1

            t1 = time.monotonic()
            losses = []
            for i in range(len(imagen.unets)):
                loss = trainer(
                    images,
                    texts = texts,
                    unet_number = i + 1,
                    max_batch_size = args.micro_batch_size
                )

                trainer.update(unet_number=i + 1)

                losses.append(loss)

            t2 = time.monotonic()
            rate.append(round(1.0 / (t2 - t1), 2))

            if step % 100 == 0:
                print("epoch {}/{} step {}/{} loss: {} - {}it/s".format(
                      epoch,
                      args.epochs,
                      step * args.batch_size,
                      len(imgs),
                      round(np.sum(losses), 5),
                      round(np.mean(rate), 2)))

            if step % 1000 == 0:
                sample_images = trainer.sample(texts=sample_texts,
                                       cond_scale=7.,
                                       return_all_unet_outputs=True)

                sample_images0 = transforms.Resize(args.size)(sample_images[0])
                sample_images1 = transforms.Resize(args.size)(sample_images[1])
                sample_images = torch.cat([sample_images0, sample_images1])

                grid = make_grid(sample_images, nrow=disp_size, normalize=False, range=(-1, 1))
                VTF.to_pil_image(grid).save(os.path.join(args.samples_out, f"imagen_{epoch}_{int(step / epoch)}.png"))

                if args.imagen is not None:
                    trainer.save(args.imagen)

        sample_images = trainer.sample(texts=sample_texts,
                                       cond_scale=7.,
                                       return_all_unet_outputs=True)

        sample_images0 = transforms.Resize(args.size)(sample_images[0])
        sample_images1 = transforms.Resize(args.size)(sample_images[1])
        sample_images = torch.cat([sample_images0, sample_images1])

        grid = make_grid(sample_images, nrow=disp_size, normalize=False, range=(-1, 1))
        VTF.to_pil_image(grid).save(os.path.join(args.samples_out, f"imagen_{epoch}_{int(step / epoch)}.png"))

        if args.imagen is not None:
            trainer.save(args.imagen)


if __name__ == "__main__":
    main()