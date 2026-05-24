import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="Evalify")
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True,
            name="token_embedding",
        )
        self.pos_emb = layers.Embedding(
            input_dim=maxlen,
            output_dim=embed_dim,
            name="position_embedding",
        )
        self.supports_masking = True

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        return self.token_emb(inputs) + self.pos_emb(positions)

    def compute_mask(self, inputs, mask=None):
        return self.token_emb.compute_mask(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"maxlen": self.maxlen, "vocab_size": self.vocab_size, "embed_dim": self.embed_dim})
        return config


@tf.keras.utils.register_keras_serializable(package="Evalify")
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            name="multi_head_attention",
        )
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="gelu"),
                layers.Dense(embed_dim),
            ],
            name="feed_forward_network",
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.supports_masking = True

    def call(self, inputs, training=False, mask=None):
        attention_mask = None
        if mask is not None:
            attention_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.bool)

        attention_output = self.att(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=attention_mask,
            training=training,
        )
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layernorm1(inputs + attention_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="Evalify")
class MaskedGlobalAveragePooling1D(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is None:
            return tf.reduce_mean(inputs, axis=1)
        mask = tf.cast(mask, inputs.dtype)
        mask = tf.expand_dims(mask, axis=-1)
        summed = tf.reduce_sum(inputs * mask, axis=1)
        counts = tf.reduce_sum(mask, axis=1)
        return summed / tf.maximum(counts, tf.keras.backend.epsilon())

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="Evalify")
class AbsoluteDifference(layers.Layer):
    def call(self, inputs):
        left, right = inputs
        return tf.abs(left - right)


@tf.keras.utils.register_keras_serializable(package="Evalify")
class BinaryFocalLossWithSmoothing(keras.losses.Loss):
    def __init__(
        self,
        gamma=2.0,
        alpha=0.25,
        label_smoothing=0.0,
        name="binary_focal_loss_with_smoothing",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_true = tf.reshape(y_true, tf.shape(y_pred))

        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_factor = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        return tf.reduce_mean(alpha_factor * modulating_factor * bce)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha, "label_smoothing": self.label_smoothing})
        return config


class V5ModelAdapter:
    """Loads the V5 Keras model and applies the same numeric feature scaling as the notebook."""

    def __init__(self, model_path, feature_config_path):
        self.model = keras.models.load_model(
            model_path,
            custom_objects={
                "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
                "TransformerEncoderBlock": TransformerEncoderBlock,
                "MaskedGlobalAveragePooling1D": MaskedGlobalAveragePooling1D,
                "AbsoluteDifference": AbsoluteDifference,
                "BinaryFocalLossWithSmoothing": BinaryFocalLossWithSmoothing,
            },
            compile=False,
        )

        self.feature_config = joblib.load(feature_config_path)
        self.numeric_features = self.feature_config["numeric_features"]
        self.feature_mean = pd.Series(self.feature_config["feature_mean"])
        self.feature_std = pd.Series(self.feature_config["feature_std"]).replace(0, 1.0)

    def predict(self, frame: pd.DataFrame, batch_size: int = 128) -> np.ndarray:
        frame = frame.copy()

        for col in self.numeric_features:
            if col not in frame.columns:
                frame[col] = 0.0

        frame[self.numeric_features] = frame[self.numeric_features].apply(
            pd.to_numeric,
            errors="coerce",
        ).fillna(0.0)

        frame[self.numeric_features] = (
            (frame[self.numeric_features] - self.feature_mean[self.numeric_features])
            / self.feature_std[self.numeric_features]
        ).astype(np.float32)

        inputs = {
            "resume_text": tf.constant(frame["resume_clean"].fillna("").astype(str).tolist(), dtype=tf.string),
            "job_text": tf.constant(frame["job_text_clean"].fillna("").astype(str).tolist(), dtype=tf.string),
            "numeric_features": tf.convert_to_tensor(
                frame[self.numeric_features].to_numpy(dtype=np.float32),
                dtype=tf.float32,
            ),
        }

        return self.model.predict(inputs, batch_size=batch_size, verbose=0).reshape(-1)
