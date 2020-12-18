import tensorflow as tf
import numpy as np

from ..config import Config as config

import matplotlib.pyplot as plt


# from .ymLayers import BatchNorm


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(1e4, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
      q, k, v must have matching leading dimensions.
      k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
      The mask has different shapes depending on its type(padding or look ahead)
      but it must be broadcastable for addition.

      Args:
        q: query shape == (..., seq_len_q, depth) / (batch, num_heads, hq, wq, depth]
        k: key shape == (..., seq_len_k, depth) / (batch, num_heads, hk, wk, depth]
        v: value shape == (..., seq_len_v, depth_v) / (batch, num_heads, hv, wv, depth]
        mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None. / [batch, num_heads, hq, wq, wk]

      Returns:
        output, attention_weights
      """

    # [batch, num_heads, hq, wq, wk]
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, sen_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor
    if mask is not None:
        scaled_attention_logits += (mask * 1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1
    # [batch, num_heads, hq, wq, wk]
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    # assert wk == wv

    # (batch, num_heads, hq, wq, depth)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
        q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, ..., depth)
        """
        shape = x.shape
        x = tf.reshape(x, (batch_size, shape[1], shape[2], self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 3, 1, 2, 4])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        # (batch, hq, wq, d_model)
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        # (batch, hk, wk, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model
        # (batch, hv, wv, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # (batch, num_heads, hq, wq, depth)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        # (batch, num_heads, hk, wk, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        # (batch, num_heads, hv, wv, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth) / (batch, num_heads, hq, wq, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k) / (batch, num_heads, hq, wq, wk)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth) / (batch, hq, wq, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 3, 1, 4])

        temp_shape = scaled_attention.shape
        # (batch_size, seq_len_q, d_model) / (batch, hq, wq, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, temp_shape[1], temp_shape[2], -1))

        output = self.dense(concat_attention)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        # (batch_size, input_seq_len, d_model) / (batch, hq, wq, d_model)
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model) / (batch, hq, wq, d_model)
        out1 = self.layernorm1(x + attn_output)

        # (batch_size, input_seq_len, d_model) / (batch, hq, wq, d_model)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch_size, input_seq_len, d_model) / (batch, hq, wq, d_model)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderLayer(tf.keras.layers.Layer):

    # TODO: check out whether decoder layer needs x as input or just the output of encoder as input.

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # (batch_size, target_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # (batch_size, target_seq_len, d_model)
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        # (batch_size, target_seq_len, d_model)
        out2 = self.layernorm2(attn2 + attn1)

        # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        # (batch_size, target_seq_len, d_model)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    """
    input x: (batch, h, w, 3)
    pos_encoding: (batch
    """

    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.embedding = tf.keras.layers.Dense(d_model)

        # TODO: Adjust the out shape of pos_encoding or check the necessaries of pos encoding
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        # seq_len = tf.shape(x)[1]

        # adding embedding and position encoding (batch, h, w, d_model)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # [batch_size, input_seq_len, d_model] / [batch, h, w, d_model]


class Decoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == [batch, h, w, d_model]
        # seq_len = tf.shape(x)[1]
        attention_weights = {}

        # x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # x += self.pos_encoding[:, :seq_len, :]

        # x = self.dropout(x, training=training)
        x = enc_output
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == [batch_size, target_seq_len, d_model]
        return x, attention_weights


class Transformer(tf.keras.Model):

    def __init__(self, num_layers, d_model, num_heads, dff, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(config.NUM_MODALS * 3)

    def call(self, inp, training, enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # [batch_size, inp_seq_len, d_model]

        # dec_output.shape == [batch_size, target_seq_len, d_model]
        dec_output, attention_weights = self.decoder(enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # [batch_size, tar_seq_len, tar_vocab_size]

        return final_output, attention_weights


if __name__ == '__main__':
    test_tensor = tf.random.normal([1, 96, 96, 3])
    transformer = Transformer(num_layers=1, d_model=256, num_heads=8, dff=1024, pe_input=96, pe_target=96)
    final_output, attention_weights = transformer(test_tensor, False, None, None, None)
    print(final_output.shape)
    print(attention_weights)
