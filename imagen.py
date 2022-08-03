import math
import numbers
import os
import random
import re
import time
from collections import deque

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as VTF
from torchvision.utils import make_grid, save_image
from PIL import Image
from tqdm import tqdm

from imagen_pytorch import ImagenTrainer, ElucidatedImagenConfig, ImagenConfig
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
    parser.add_argument('--cond_images', type=str, default=None)
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
    parser.add_argument("--sample_unet", default=None, type=int)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--text_encoder', type=str, default="t5-large")
    parser.add_argument("--cond_scale", default=10, type=float, help="sampling conditional scale 0-10.0")
    parser.add_argument('--no_elu', action='store_true', help="don't use elucidated imagen")

    # training
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--micro_batch_size', default=8, type=int)
    parser.add_argument('--samples_out', default="samples")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--train_encoder', action='store_true')
    parser.add_argument('--shuffle_tags', action='store_true')
    parser.add_argument('--train_unet', type=int, default=1)
    parser.add_argument('--random_drop_tags', type=float, default=0.)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--no_text_transform', action='store_true')
    parser.add_argument('--start_epoch', default=1, type=int)

    args = parser.parse_args()

    if args.sample_steps is None:
        args.sample_steps = args.size

    if args.tags_source is None:
        args.tags_source = args.source

    if args.vocab is None:
        args.vocab = args.source
    else:
        assert os.path.isfile(args.vocab)

    if args.bf16:
        # probably (maybe) need to set TORCH_CUDNN_V8_API_ENABLED=1 in environment
        torch.set_float32_matmul_precision("medium")

    if args.train_encoder:
        train_encoder(args)

    if args.train:
        train(args)
    else:
        sample(args)

def sample(args):

    if os.path.isfile(args.output) and not args.replace:
        return

    try:
        imagen = load(args.imagen).to(args.device)
    except:
        print(f"Error loading model: {args.imagen}")
        return

    args.num_unets = len(imagen.unets) - 1

    image_sizes = get_image_sizes(args)
    print(f"image sizes: {image_sizes}")

    imagen.image_sizes = image_sizes

    cond_image = None

    if args.cond_images is not None and os.path.isfile(args.cond_images):
        tforms = transforms.Compose([PadImage(),
                                     transforms.Resize((args.size, args.size)),
                                     transforms.ToTensor()])
        cond_image = Image.open(args.cond_images)
        cond_image = tforms(cond_image).to(imagen.device)
        cond_image = cond_image.view(1, *cond_image.shape)

    sample_images = imagen.sample(texts=[args.tags],
                                  cond_images=cond_image,
                                  cond_scale=args.cond_scale,
                                  return_pil_images=True,
                                  stop_at_unet_number=args.sample_unet)

    final_image = sample_images[-1]
    final_image.resize((args.size, args.size)).save(args.output)


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


def get_image_sizes(args):
    image_sizes = [args.start_size]

    for i in range(0, args.num_unets):
        ns = image_sizes[-1] * 4
        if args.train:
            ns = ns // 4
        image_sizes.append(ns)

    image_sizes[-1] = args.size // 4 if args.train else args.size

    return image_sizes


def get_imagen(args, unet_dims=None, unet2_dims=None):

    if unet_dims is None:
        unet_dims = args.unet_dims

    if unet2_dims is None:
        unet2_dims = args.unet2_dims

    if args.cond_images is not None:
        cond_images_channels = 3
    else:
        cond_images_channels = 0

    # unet for imagen
    unet1 = dict(
        dim=unet_dims,
        cond_dim=512,
        dim_mults=(1, 2, 4, 6),
        cond_images_channels=cond_images_channels,
        num_resnet_blocks=3,
        layer_attns=(False, True, True, True),
        memory_efficient=True
    )

    unets = [unet1]

    for i in range(args.num_unets):

        unet2 = dict(
            dim=unet2_dims // (i + 1),
            cond_dim=512,
            dim_mults=(1, 2, 4, 4),
            cond_images_channels=cond_images_channels,
            num_resnet_blocks=(2, 4, 8, 8),
            layer_attns=(False, False, False, i < 2),
            layer_cross_attns=(False, False, True, True),
            final_conv_kernel_size=1,
            memory_efficient=True
        )

        unets.append(unet2)

    image_sizes = get_image_sizes(args)

    print(f"image_sizes={image_sizes}")

    sample_steps = [args.sample_steps] * (args.num_unets + 1)

    if not args.no_elu:
        imagen = ElucidatedImagenConfig(
            unets=unets,
            text_encoder_name=args.text_encoder,
            num_sample_steps=sample_steps,
            # noise_schedules=["cosine", "cosine"],
            # pred_objectives=["noise", "x_start"],
            image_sizes=image_sizes,
            per_sample_random_aug_noise_level=True,
            lowres_sample_noise_level=0.3
        ).create().to(args.device)

    else:
        imagen = ImagenConfig(
            unets=unets,
            text_encoder_name=args.text_encoder,
            noise_schedules=["cosine", "cosine"],
            pred_objectives=["noise", "x_start"],
            image_sizes=image_sizes,
            per_sample_random_aug_noise_level=True,
            lowres_sample_noise_level=0.3
        ).create().to(args.device)

    return imagen


def make_training_samples(poses, trainer, args, epoch, step, epoch_loss):
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

    # dup the sampler's image sizes temporarily:
    args.train = False
    sample_image_sizes = get_image_sizes(args)
    args.train = True

    train_image_sizes = trainer.imagen.image_sizes

    trainer.imagen.image_sizes = sample_image_sizes

    sample_images = trainer.sample(texts=sample_texts,
                                   cond_images=sample_poses,
                                   cond_scale=args.cond_scale,
                                   return_all_unet_outputs=True,
                                   stop_at_unet_number=args.train_unet)

    # restore train image sizes:
    trainer.imagen.image_sizes = train_image_sizes

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
    VTF.to_pil_image(grid).save(os.path.join(args.samples_out, f"imagen_{epoch}_{int(step / epoch)}_loss{epoch_loss}.png"))


def train(args):

    imagen = get_imagen(args)

    precision = None

    if args.fp16:
        precision = "fp16"
    elif args.bf16:
        precision = "bf16"

    trainer = ImagenTrainer(imagen, fp16=args.fp16)

    if args.imagen is not None and os.path.isfile(args.imagen):
        print(f"Loading model: {args.imagen}")
        trainer.load(args.imagen)

    print("Fetching image indexes...")

    imgs = get_images(args.source, verify=False)
    txts = get_images(args.tags_source, exts=".txt")

    print(f"{len(imgs)} images")
    print(f"{len(txts)} tags")

    cond_images = None
    has_poses = False

    if args.cond_images is not None:
        cond_images = get_images(args.cond_images)
        has_poses = True

    # get non-training sizes for image resizing/cropping
    args.train = False
    train_img_size = get_image_sizes(args)[args.train_unet - 1]

    tforms = transforms.Compose([
            PadImage(),
            transforms.Resize(train_img_size),
            transforms.ToTensor()])

    if args.train_unet > 1:
        tforms = transforms.Compose([
            transforms.Resize(args.size),
            transforms.RandomCrop(train_img_size),
            transforms.RandomHorizontalFlip(),
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
                             poses=cond_images,
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

    print(f"training on {len(data)} images")

    for epoch in range(args.start_epoch, args.epochs + 1):
        step = 0
        epoch_loss = 0
        with tqdm(dl, unit="batches") as tepoch:
            for data in tepoch:

                if has_poses:
                    images, texts, cond_images = data
                else:
                    images, texts = data
                    cond_images = None

                step += 1

                loss = trainer(
                    images,
                    cond_images=cond_images,
                    texts=texts,
                    unet_number=args.train_unet,
                    max_batch_size=args.micro_batch_size
                )

                trainer.update(unet_number=args.train_unet)

                epoch_loss += loss
                epoch_loss_disp = round(float(epoch_loss) / float(step), 5)

                tepoch.set_description(f"Epoch {epoch}")
                tepoch.set_postfix(loss=round(loss, 5), epoch_loss=epoch_loss_disp)

                if step % 100 == 0:
                    make_training_samples(cond_images, trainer, args, epoch,
                                          trainer.num_steps_taken(args.train_unet),
                                          epoch_loss_disp)

                    if args.imagen is not None:
                        trainer.save(args.imagen)
        # END OF EPOCH
        make_training_samples(cond_images, trainer, args, epoch,
                              trainer.num_steps_taken(args.train_unet),
                              epoch_loss_disp)

        if args.imagen is not None:
            trainer.save(args.imagen)


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer, file_paths: str, block_size=512):

        lines = []

        for file_path in tqdm(file_paths):
            assert os.path.isfile(file_path)
            with open(file_path, encoding="utf-8") as f:
                lines.extend([line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())])

        if tokenizer is not None:
            self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]
        else:
            self.examples = lines

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def train_tokenizer(args):
    import io
    import sentencepiece as spm

    if args.vocab is None:
        args.vocab = args.tags_source

    os.makedirs(args.text_encoder, exist_ok=True)

    print("Fetching vocab...")

    vocab = get_vocab(args.vocab, top=74270)

    vocab_file = os.path.join(args.text_encoder, "t5_vocab_input.vocab")

    with open(vocab_file, 'w') as f:
        f.write("\n".join(vocab))

    print(f"vocab size: {len(vocab)}")
    print("training tokenizer...")
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(input=vocab_file,
                                   model_writer=model, vocab_size=len(vocab))

    output_file = os.path.join(args.text_encoder, "t5_model.spm")

    with open(output_file, 'wb') as f:
        f.write(model.getvalue())

    return output_file


def train_encoder(args):

    from transformers import T5ForConditionalGeneration, TrainingArguments, Trainer
    from transformers import DataCollatorForLanguageModeling
    from transformers import T5Tokenizer

    assert args.text_encoder is not None

    pretrained = "t5-small"

    if os.path.exists(args.text_encoder):
        pretrained = args.text_encoder

    model = T5ForConditionalGeneration.from_pretrained(pretrained)

    t5_spm_model = train_tokenizer(args)

    tokenizer = T5Tokenizer(t5_spm_model)

    txts = get_images(args.tags_source, exts=".txt")
    tokenizer.save_pretrained(args.text_encoder)

    tokenizer.pad_token = tokenizer.eos_token

    lm_dataset = LineByLineTextDataset(tokenizer, txts)
    val_dataset = LineByLineTextDataset(tokenizer, txts[-2:])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
                                      output_dir=args.text_encoder,
                                      evaluation_strategy="epoch",
                                      learning_rate=2e-5,
                                      weight_decay=0.01,
                                      num_train_epochs=args.epochs,
                                      auto_find_batch_size=True,
                                      save_strategy="epoch",
                                      save_total_limit=3,
                                      bf16=args.bf16
                                     )

    trainer = Trainer(
                      model=model,
                      args=training_args,
                      train_dataset=lm_dataset,
                      eval_dataset=val_dataset,
                      data_collator=data_collator,
                     )

    trainer.train()
    trainer.save_model(args.text_encoder)


if __name__ == "__main__":
    main()
