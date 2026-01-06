import tensorflow as tf
from tensorflow import keras


class WGAN_GP_C(keras.Model):
    """
    Conditional Wasserstein GAN with Gradient Penalty (WGAN-GP)
    """

    def __init__(self,
                 discriminator,
                 generator,
                 latent_dim,
                 num_classes,
                 gp_weight=10.0):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.gp_weight = gp_weight
        self.d_optimizer = None
        self.g_optimizer = None

    def compile(self,
                d_optimizer,
                g_optimizer,
                d_loss_fn=None,
                g_loss_fn=None):
        """
        Almacena los optimizadores sin llamar al compile de la clase base
        """
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_sequences, fake_sequences,
                         labels):
        """
        Calculo del gradient penalty para WGAN-GP
        """
        alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
        diff = fake_sequences - real_sequences
        interpolated = real_sequences + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator([interpolated, labels], training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0)**2)
        return gp

    def train_step(self, data):
        """
        Un paso de entrenamiento para discriminador y generador
        """
        real_sequences, labels = data
        batch_size = tf.shape(real_sequences)[0]

        random_labels_idx = tf.random.uniform((batch_size, ),
                                              maxval=self.num_classes,
                                              dtype=tf.int32)
        random_labels_oh = tf.one_hot(random_labels_idx, self.num_classes)

        # ENTRENAR DISCRIMINADOR (5 veces)
        for _ in range(5):
            random_latent_vectors = tf.random.normal(shape=(batch_size,
                                                            self.latent_dim))
            fake_sequences = self.generator(
                [random_latent_vectors, labels], training=True)

            with tf.GradientTape() as tape:

                fake_logits = self.discriminator(
                    [fake_sequences, labels], training=True)
                real_logits = self.discriminator([real_sequences, labels],
                                                 training=True)
                d_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(
                    real_logits)

                gp = self.gradient_penalty(batch_size, real_sequences,
                                           fake_sequences, labels)
                d_loss = d_cost + gp * self.gp_weight

            d_gradient = tape.gradient(d_loss,
                                       self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables))

        # ENTRENAR GENERADOR (1 vez)
        random_latent_vectors = tf.random.normal(shape=(batch_size,
                                                        self.latent_dim))

        with tf.GradientTape() as tape:
            fake_sequences = self.generator([random_latent_vectors, labels],
                                            training=True)
            fake_logits = self.discriminator([fake_sequences, labels],
                                             training=True)
            g_loss = -tf.reduce_mean(fake_logits)

        g_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradient, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}