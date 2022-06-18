import os
import numpy as np
import PIL
from PIL import Image

from gan_utils import txt_to_onehot, \
    txt_from_onehot, \
    get_images, \
    get_vocab, \
    load_image


class DataGenerator():
    def __init__(self, images, txts, vocab,
                 to_fit=True, batch_size=32, dim=(256, 256),
                 n_channels=3, shuffle=True, transform=None,
                 normalize=True, ret_filenames=False,
                 channels_first=False,
                 tag_transform=None,
                 silent=True,
                 as_array=False,
                 resize=False,
                 limit=None,
                 return_raw_txt=False):

        self.images = images
        self.txts = txts
        self.vocab = vocab
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.ret_filenames = ret_filenames
        self.channels_first = channels_first
        self.ndim = 1
        self.as_array = as_array
        self.resize = resize
        self.limit = limit
        self.return_raw_txt = return_raw_txt
        self.on_epoch_end()


        self.transform = transform
        self.tag_transform = tag_transform
        self.silent = silent

        if self.transform is None:
            self.transform = lambda x: x
        else:
            # cannot have a transform and normalize
            normalize = False

        self.normalize = normalize

        self.txts_oh = {}

        self._preload_txts()

    def _preload_txts(self):
        print("preloading txt files...")

        tot = len(self.images)
        count = 0

        txts = {}

        for txt in self.txts:
            bn = os.path.basename(txt)

            bn = bn.replace('.txt', '')

            txts[bn] = txt

        for img in self.images:
            count += 1
            print("\rprocessing {}/{}".format(count, tot), end='', flush=True)

            bn = os.path.basename(img)
            bn = os.path.splitext(bn)[0]

            try:
                txt = txts[bn]

                with open(txt, 'r') as f:
                    txt_data = f.read()

                if self.tag_transform is not None:
                    txt_data = self.tag_transform(txt_data)

                if not self.return_raw_txt:
                    oh = txt_to_onehot(self.vocab, txt_data)
                else:
                    oh = txt_data

                self.txts_oh[bn] = oh

            except KeyError:
                continue

        print("\ndone preloading txt onehots")

    def __len__(self):
        """Denotes the number of batches per epoch

        :return: number of batches per epoch
        """

        if self.limit:
            return (int(min(np.floor(len(self.images) / self.batch_size), self.limit / self.batch_size)))

        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # print(f"getting index: {index}")
        # Generate indexes of the batch
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.images[k] for k in indexes]

        # Generate data
        return self._generate_X(list_IDs_temp)

    def on_epoch_end(self):
        """Updates indexes after each epoch

        """
        self.indexes = np.arange(len(self.images))

        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images

        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        fn = []
        if self.channels_first:
            if self.batch_size > 1:
                X = np.zeros((self.batch_size, self.n_channels, *self.dim))
            else:
                X = np.zeros((self.n_channels, *self.dim))
        else:
            X = np.zeros((self.batch_size, *self.dim, self.n_channels))

        y = np.zeros((self.batch_size, len(self.vocab)))

        # Generate data
        for i, img in enumerate(list_IDs_temp):
            # Store sample

            bn = os.path.basename(img)
            bn = os.path.splitext(bn)[0]

            if bn not in self.txts_oh:
                if not self.silent:
                    print(f"could not find {bn} in preloaded txts")
                continue

            try:
                if self.normalize:
                    the_img = load_image(img, self.dim,
                                         normalize=self.normalize)
                else:
                    the_img = Image.open(img).convert("RGB")

                if self.resize:
                    the_img = the_img.resize(self.dim, Image.BICUBIC)

                the_img = self.transform(the_img)

                if self.as_array:
                    the_img = np.array(the_img)

            except ValueError as v:
                print(f"error processing {img}: {v}")
                self.images.remove(img)
                self.on_epoch_end()
                os.remove(img)

                continue

            except FileNotFoundError as er:
                print(f"error processing {img}: {er}")
                self.images.remove(img)
                self.on_epoch_end()

                continue

            fn.append(img)
            X[i, ] = the_img
            # except Exception as ex:
            #    print("failzors")
            #    print(ex)
            #    continue

            muh_oh = self.txts_oh[bn]
            y[i, ] = muh_oh
            # print(txt_from_onehot(self.vocab, self.txts_oh[bn]))

            # print(f"loading {img}")

        if self.ret_filenames:
            return fn, X, y

        return X, y


class ImageLabelDataset():
    def __init__(self, images, txts, vocab,
                 poses=None,
                 to_fit=True, 
                 dim=(256, 256),
                 n_channels=3,
                 shuffle=True,
                 transform=None,
                 normalize=True,
                 ret_filenames=False,
                 channels_first=False,
                 tag_transform=None,
                 silent=True,
                 as_array=False,
                 resize=False,
                 limit=None,
                 return_raw_txt=False,
                 no_preload=False):

        self.images = images
        self.txts = txts
        self.vocab = vocab
        self.poses = poses
        self.has_poses = True if poses is not None else False
        self.to_fit = to_fit
        self.batch_size = 1
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.ret_filenames = ret_filenames
        self.channels_first = channels_first
        self.ndim = 1
        self.as_array = as_array
        self.resize = resize
        self.limit = limit
        self.return_raw_txt = return_raw_txt
        self.on_epoch_end()

        self.transform = transform
        self.tag_transform = tag_transform
        self.silent = silent

        if self.transform is None:
            self.transform = lambda x: x
        else:
            # cannot have a transform and normalize
            normalize = False

        self.normalize = normalize

        self.txts_oh = {}
        self.poses_preload = {}

        if not no_preload:
            self._preload_txts()
            if self.has_poses:
                self._preload_poses()

    def _preload_txts(self, images=None):
        # print("preloading txt files...")

        if images is None:
            images = self.images

        tot = len(images)
        count = 0

        txts = {}

        for txt in self.txts:
            bn = os.path.basename(txt)

            bn = bn.replace('.txt', '')

            txts[bn] = txt

        for img in images:
            count += 1
            # print("\rprocessing {}/{}".format(count, tot), end='', flush=True)

            bn = os.path.basename(img)
            bn = os.path.splitext(bn)[0]

            try:
                txt = txts[bn]

                with open(txt, 'r') as f:
                    txt_data = f.read()

                if self.tag_transform is not None:
                    txt_data = self.tag_transform(txt_data)

                if not self.return_raw_txt:
                    oh = txt_to_onehot(self.vocab, txt_data)
                else:
                    oh = txt_data

                self.txts_oh[bn] = oh

            except KeyError:
                continue

    def _preload_poses(self, images=None):
        # print("preloading pose files...")

        if images is None:
            images = self.images

        tot = len(images)
        count = 0

        poses = {}

        for pose in self.poses:
            bn = os.path.basename(pose)

            bn = os.path.splitext(bn)[0]

            poses[bn] = pose

        for img in images:
            count += 1
            # print("\rprocessing {}/{}".format(count, tot), end='', flush=True)

            bn = os.path.basename(img)
            bn = os.path.splitext(bn)[0]

            try:
                pose = poses[bn]
                self.poses_preload[bn] = pose
            except KeyError:
                continue

    def __len__(self):
        """Denotes the number of batches per epoch

        :return: number of batches per epoch
        """

        if self.limit:
            return (int(min(np.floor(len(self.images) / self.batch_size), self.limit / self.batch_size)))

        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # print(f"getting index: {index}")
        # Generate indexes of the batch
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.images[k] for k in indexes]

        # Generate data
        return self._generate_X(list_IDs_temp)

    def on_epoch_end(self):
        """Updates indexes after each epoch

        """
        self.indexes = np.arange(len(self.images))

        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images

        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        img = list_IDs_temp[-1]
        fn = []
        bn = os.path.basename(img)
        bn = os.path.splitext(bn)[0]

        # if bn not in self.txts_oh:
        #     if not self.silent:
        #         print(f"could not find {bn} in preloaded txts")
        #     print(f"could not find {bn} in preloaded txts")
        #     return

        try:
            if self.normalize:
                the_img = load_image(img, self.dim,
                                     normalize=self.normalize)
            else:
                the_img = Image.open(img).convert("RGB")

            if self.resize:
                the_img = the_img.resize(self.dim, Image.BICUBIC)

            the_img = self.transform(the_img)

            if self.as_array:
                the_img = np.array(the_img)

        except ValueError as v:
            print(f"error processing {img}: {v}")
            self.images.remove(img)
            self.on_epoch_end()
            os.remove(img)

            return

        except FileNotFoundError as er:
            print(f"error processing {img}: {er}")
            self.images.remove(img)
            self.on_epoch_end()

            return

        except Image.DecompressionBombError as v:
            print(f"error processing {img}: {v}")
            self.images.remove(img)
            self.on_epoch_end()
            os.remove(img)

            return

        except PIL.UnidentifiedImageError as v:
            print(f"error processing {img}: {v}")
            self.images.remove(img)
            self.on_epoch_end()
            os.remove(img)

            return
        except OSError as v:
            print(f"error processing {img}: {v}")
            self.images.remove(img)
            self.on_epoch_end()
            os.remove(img)

            return

        fn.append(img)
        X = the_img
        # except Exception as ex:
        #    print("failzors")
        #    print(ex)
        #    continue

        # print(bn)
        muh_oh = self.txts_oh.get(bn, None)

        if muh_oh is None:
            self._preload_txts(images=[img])
            muh_oh = self.txts_oh[bn]

        y = muh_oh
        # print(txt_from_onehot(self.vocab, self.txts_oh[bn]))

        # print(f"loading {img}")

        if self.has_poses:
            pose = self.poses_preload.get(bn, None)

            if pose is None:
                self._preload_poses(images=[img])
                pose = self.poses_preload[bn]

            the_pose = Image.open(pose).convert("RGB")
            the_pose = self.transform(the_pose)

            return X, y, the_pose

        if self.ret_filenames:
            return fn, X, y

        # print(X.size())
        # print(y.size())
        return X, y
