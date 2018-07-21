import argparse
import numpy as np
from pathlib import Path
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from model import get_srresnet_model, PSNR
from generator import NoisyImageGenerator, ValGenerator
from noise_model import get_noise_model


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.5
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.25
        return self.initial_lr * 0.125


def get_args():
    parser = argparse.ArgumentParser(description="train noise2noise model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="train image dir")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="test image dir")
    parser.add_argument("--image_size", type=int, default=64,
                        help="training patch size")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=60,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument("--steps", type=int, default=1000,
                        help="steps per epoch")
    parser.add_argument("--output_path", type=str, default="checkpoints",
                        help="checkpoint dir")
    parser.add_argument("--source_noise_model", type=str, default="gaussian,0,50",
                        help="checkpoint dir")
    parser.add_argument("--target_noise_model", type=str, default="gaussian,25,25",
                        help="checkpoint dir")
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    image_dir = args.image_dir
    test_dir = args.test_dir
    image_size = args.image_size
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    steps = args.steps
    output_path = Path(__file__).resolve().parent.joinpath(args.output_path)
    model = get_srresnet_model()
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss="mse", metrics=[PSNR])
    source_noise_model = get_noise_model(args.source_noise_model)
    target_noise_model = get_noise_model(args.target_noise_model)
    generator = NoisyImageGenerator(image_dir, source_noise_model, target_noise_model, batch_size=batch_size,
                                    image_size=image_size)
    val_generator = ValGenerator(test_dir, source_noise_model)
    output_path.mkdir(parents=True, exist_ok=True)
    callbacks = [
        LearningRateScheduler(schedule=Schedule(nb_epochs, lr)),
        ModelCheckpoint(str(output_path) + "/weights.{epoch:03d}-{val_loss:.3f}-{val_PSNR:.5f}.hdf5",
                        monitor="val_PSNR",
                        verbose=1,
                        mode="max",
                        save_best_only=True)
    ]

    hist = model.fit_generator(generator=generator,
                               steps_per_epoch=steps,
                               epochs=nb_epochs,
                               validation_data=val_generator,
                               verbose=1,
                               callbacks=callbacks,
                               use_multiprocessing=True,
                               workers=8
                               )

    np.savez(str(output_path.joinpath("history.npz")), history=hist.history)


if __name__ == '__main__':
    main()
