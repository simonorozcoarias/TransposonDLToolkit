import os
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

from utils.utils import (load_and_preprocess_fasta, load_and_preprocess_fasta_grouped, NUM_CLASSES, INITIAL_LR,
                         MAX_EPOCHS, ES_PATIENCE)
from utils.models import make_generator_model, make_critic_model, make_discriminator_lsgan
from utils.gans import WGAN_GP_C


def train_gan_with_es(gan_model, dataset, args, initial_lr, group=None):
    """
    Entrenamiento de GANc
    """
    d_optimizer = keras.optimizers.Adam(learning_rate=initial_lr,
                                        beta_1=0.5,
                                        beta_2=0.9)
    g_optimizer = keras.optimizers.Adam(learning_rate=initial_lr,
                                        beta_1=0.5,
                                        beta_2=0.9)
    gan_model.compile(d_optimizer=d_optimizer,
                        g_optimizer=g_optimizer,
                        d_loss_fn=None,
                        g_loss_fn=None)
    wait = 0
    best_loss = float('inf')
    
    prefix = f"{group}_" if group else ""

    # Entrenamiento
    for epoch in range(MAX_EPOCHS):
        start_time = time.time()
        epoch_d_loss = []
        epoch_g_loss = []

        for data in dataset:
            logs = gan_model.train_step(data)
            epoch_d_loss.append(logs['d_loss'].numpy())
            epoch_g_loss.append(logs['g_loss'].numpy())

        avg_d_loss = np.mean(epoch_d_loss)
        avg_g_loss = np.mean(epoch_g_loss)
        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{MAX_EPOCHS} ({epoch_time:.2f}s) - D_Loss={avg_d_loss:.4f}, G_Loss={avg_g_loss:.4f}"
        )

        # Earlystopping
        if avg_d_loss < best_loss:
            best_loss = avg_d_loss
            wait = 0
            generator_path = f'trained_models/{prefix}c_generator_best_{args.seq_len}_{args.noise_dim}_{args.batch_size}.weights.h5'
            gan_model.generator.save_weights(generator_path)
        else:
            wait += 1
            if wait >= ES_PATIENCE:
                print(
                    f"\n--- Early Stopping tras {epoch + 1} epochs. D_Loss no mejora tras {ES_PATIENCE} epochs. ---"
                )
                gan_model.generator.load_weights(generator_path)
                print(
                    f"Generator restaurado a mejores pesos ({generator_path}).")
                break

    return gan_model.generator


def get_args():
    """
    Parser de argumentos
    """
    parser = argparse.ArgumentParser(
        description="Entrenamiento de GANs para DNA sequences con Early Stopping.")
    parser.add_argument('--data_file',
                        type=str,
                        default='inpactordb2.fasta',
                        help='fichero FASTA.')
    parser.add_argument('--seq_len',
                        type=int,
                        default=600,
                        help='Longitud máxima de la secuencia.')
    parser.add_argument('--noise_dim',
                        type=int,
                        default=100,
                        help='Dimension del vector de ruido latente.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Tamaño del batch.')
    parser.add_argument('--grouping',
                        action='store_true',
                        help='Si se añade, se entrenan GANs separados para Transposones y Retrotransposones')
    return parser.parse_args()


def main():
    """
    Main
    """
    args = get_args()
    
    if args.grouping:
        print("----------- GROUPING GAN -----------")
        transposons, retrotransposons, num_classes = load_and_preprocess_fasta_grouped(
            args.data_file, max_len=args.seq_len)

        groups_data = [
            ("transposons", transposons),
            ("retrotransposons", retrotransposons)
        ]
        
        for group_name, (X_train, Y_train_oh) in groups_data:
            print("Empezando entrenamiento GAN para", group_name)
            channels = X_train.shape[-1]
            dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train_oh))
            dataset = dataset.shuffle(buffer_size=len(X_train)).batch(args.batch_size)

            generator = make_generator_model(seq_len=args.seq_len,
                                                channels=channels,
                                                noise_dim=args.noise_dim,
                                                num_classes=num_classes)
            discriminator = make_critic_model(seq_len=args.seq_len,
                                                channels=channels,
                                                num_classes=num_classes)
            gan_model = WGAN_GP_C(discriminator=discriminator,
                                    generator=generator,
                                    latent_dim=args.noise_dim,
                                    num_classes=num_classes)

            generator = train_gan_with_es(gan_model, dataset, args, INITIAL_LR, group=group_name)
            generator_final_path = f'trained_models/{group_name}_c_generator_{args.seq_len}_{args.noise_dim}_{args.batch_size}.weights.h5'
            generator.save(generator_final_path)
            print("Generator model for", group_name, "saved at:", generator_final_path)
            
    else:
        print("----------- SIMPLE GAN -----------")
        X_train, Y_train_oh, num_classes = load_and_preprocess_fasta(
            args.data_file, max_len=args.seq_len)
        channels = X_train.shape[-1]
        dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train_oh))
        dataset = dataset.shuffle(buffer_size=len(X_train)).batch(args.batch_size)

        generator = make_generator_model(seq_len=args.seq_len,
                                            channels=channels,
                                            noise_dim=args.noise_dim,
                                            num_classes=num_classes)
        discriminator = make_critic_model(seq_len=args.seq_len,
                                            channels=channels,
                                            num_classes=num_classes)
        gan_model = WGAN_GP_C(discriminator=discriminator,
                                generator=generator,
                                latent_dim=args.noise_dim,
                                num_classes=num_classes)

        generator = train_gan_with_es(gan_model, dataset, args, INITIAL_LR)
        generator_final_path = f'trained_models/c_generator_{args.seq_len}_{args.noise_dim}_{args.batch_size}.weights.h5'
        generator.save(generator_final_path)
        print("Generator model saved at:", generator_final_path)

if __name__ == "__main__":
    os.makedirs("trained_models", exist_ok=True)
    main()
