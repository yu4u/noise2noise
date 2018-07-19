from pathlib import Path
import random
import numpy as np
import cv2
from keras.utils import Sequence


def add_noise(img, min_stddev=0, max_stddev=50):
    noise_img = img.astype(np.float)
    stddev = np.random.uniform(min_stddev, max_stddev)
    noise = np.random.randn(*img.shape) * stddev
    noise_img += noise
    noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)

    return noise_img


class NoisyImageGenerator(Sequence):
    def __init__(self, image_dir, batch_size=32, image_size=64, min_stddev=0, max_stddev=50):
        self.image_paths = list(Path(image_dir).glob("*.jpg"))
        self.image_num = len(self.image_paths)
        self.batch_size = batch_size
        self.image_size = image_size
        self.min_stddev = min_stddev
        self.max_stddev = max_stddev

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        sample_id = 0

        while True:
            image_path = random.choice(self.image_paths)
            image = cv2.imread(str(image_path))
            h, w, _ = image.shape

            if h >= image_size and w >= image_size:
                h, w, _ = image.shape
                i = np.random.randint(h - image_size + 1)
                j = np.random.randint(w - image_size + 1)
                clean_patch = image[i:i + image_size, j:j + image_size]

                y[sample_id] = add_noise(clean_patch, min_stddev=self.min_stddev, max_stddev=self.max_stddev)
                x[sample_id] = add_noise(clean_patch, min_stddev=self.min_stddev, max_stddev=self.max_stddev)

                sample_id += 1

                if sample_id == batch_size:
                    return x, y


class ValGenerator(Sequence):
    def __init__(self, image_dir, stddev=25):
        image_paths = list(Path(image_dir).glob("*.*"))
        self.image_num = len(image_paths)
        self.data = []

        for image_path in image_paths:
            y = cv2.imread(str(image_path))
            x = add_noise(y, min_stddev=stddev, max_stddev=stddev)
            self.data.append([np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)])

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        return self.data[idx]
