import csv
import glob
import math
import ntpath

import imageio
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

image_size = 28
image_chn = 1
input_shape = (image_size, image_size, image_chn)


# Logic for calculating reconstruction probability
def reconstruction_probability(dec, z_mean, z_log_var, X):
    """
    :param decoder: decoder model
    :param z_mean: encoder predicted mean value
    :param z_log_var: encoder predicted sigma square value
    :param X: input data
    :return: reconstruction probability of input
            calculated over L samples from z_mean and z_log_var distribution
    """
    sampled_zs = sampling([z_mean, z_log_var])
    mu_hat = dec(sampled_zs)

    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(X, mu_hat), axis=(-1)
        )
    )

    return reconstruction_loss


# Calculates and returns probability density of test input
def calculate_density(x_target_orig, enc, dec):
    x_target_orig = np.clip(x_target_orig, 0, 1)
    x_target = np.reshape(x_target_orig, (-1, 28 * 28))
    z_mean, z_log_var, _ = enc(x_target)
    reconstructed_prob_x_target = reconstruction_probability(dec, z_mean, z_log_var, x_target)
    return reconstructed_prob_x_target


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def compute_valid(sample, encoder, decoder, tshd):
    # fp = []
    # tn = []
    # for batch in anomaly_test:
    rec_loss = calculate_density(sample, encoder, decoder)
    if rec_loss > tshd or math.isnan(rec_loss):
        distr = 'ood'
        # tn.append(rec_loss)
    else:
        distr = 'id'
    return distr, rec_loss.numpy()
    # fp.append(rec_loss)

    # print("id: " + str(len(fp)))
    # print("ood: " + str(len(tn)))


def main():
    csv_file = r"losses/ood_analysis_xmutant_all_classes.csv"
    # VAE density threshold for classifying invalid inputs
    vae_threshold = 0.26608911681183206
    VAE = "mnist_vae_stocco_all_classes"

    decoder = tf.keras.models.load_model("trained/" + VAE + "/decoder", compile=False)
    encoder = tf.keras.models.load_model("trained/" + VAE + "/encoder", compile=False)

    RESULTS_PATH = r"../../generated_images/mnist_inputs/"

    with open(csv_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['TOOL', 'SAMPLE', 'ID/OOD', 'loss'])

        print("== XMutant ==")
        DJ_FOLDER = RESULTS_PATH + "mnist_xm/*.npy"
        filelist = [f for f in glob.glob(DJ_FOLDER)]

        # TODO: uniform the format and avoid code duplication
        # npy extension
        if len(filelist) != 0:
            print("Found samples: " + str(len(filelist)))
            for sample in filelist:
                s = np.load(sample)
                distr, loss = compute_valid(s, encoder, decoder, vae_threshold)
                sample_name = ntpath.split(sample)[-1]
                writer.writerow(['XMutant', sample_name, distr, loss])
        # png extension
        else:
            DJ_FOLDER = RESULTS_PATH + "mnist_xm/*.png"
            filelist = [f for f in glob.glob(DJ_FOLDER)]

            print("Found samples: " + str(len(filelist)))
            for sample in filelist:
                s = imageio.v2.imread(sample)
                distr, loss = compute_valid(s, encoder, decoder, vae_threshold)
                sample_name = ntpath.split(sample)[-1]
                writer.writerow(['XMutant', sample_name, distr, loss])


if __name__ == "__main__":
    '''
    Usage: place the MNIST digits for validation in the folder 'mnist/selforacle/generated_images/mnist_inputs/mnist_xm', either in npy or png format (other formats are currently not supported). 
    Then run the main function to calculate the reconstruction probability of the images and classify them as ID or OOD.
    The results are stored in a csv file 'ood_analysis_xmutant_all_classes.csv' within the 'mnist/selforacle/losses' folder.
    A different threshold value can be selected looking at the selforacle_thresholds_all_classes.json file.
    '''
    main()
