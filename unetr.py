import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model

def transformer_encoder(x, cf):
    skip1 = x
    x = L.LayerNormalization()(x)
    x = L.MultiHeadAttention(num_heads=cf['num_heads'], key_dim=cf['hidden_dim'])(x, x)
    x = L.Add()([x, skip1])
    
    skip2 = x
    x = L.LayerNormalization()(x)
    x = L.Dense(cf['hidden_dim'] * 4, activation='relu')(x)
    x = L.Dense(cf['hidden_dim'])(x)
    x = L.Add()([x, skip2])
    return x

def convdecoder(x, filters):
    x = L.Conv2D(filters, kernel_size=3, padding='same', activation='relu')(x)
    return x

def deconv_block(x, filters):
    x = L.Conv2DTranspose(filters, kernel_size=2, strides=2, padding='same', activation='relu')(x)
    return x

def unetr2D(cf):
    input_shape = (cf['num_patch'], cf['patch_size'] * cf['patch_size'] * cf['num_channels'])
    inputs = L.Input(input_shape)

    patch_embed = L.Dense(cf['hidden_dim'])(inputs)
    position = tf.range(start=0, limit=cf['num_patch'], delta=1)
    position_embed = L.Embedding(input_dim=cf['num_patch'], output_dim=cf['hidden_dim'])(position)

    x = patch_embed + position_embed

    skip_connection_index = [3, 6, 9, 12]
    skip_connection = []
    for i in range(1, cf['num_layers'] + 1):
        x = transformer_encoder(x, cf)
        if i in skip_connection_index:
            skip_connection.append(x)

    z3, z6, z9, z12 = skip_connection
    z0 = L.Reshape((cf['image_size'], cf['image_size'], cf['num_channels']))(inputs)
    size = cf['image_size'] // cf['patch_size']
    z3 = L.Reshape((size, size, cf['hidden_dim']))(z3)
    z6 = L.Reshape((size, size, cf['hidden_dim']))(z6)
    z9 = L.Reshape((size, size, cf['hidden_dim']))(z9)
    z12 = L.Reshape((size, size, cf['hidden_dim']))(z12)

    x = deconv_block(z12, 512)

    s = deconv_block(z9, 512)
    s = convdecoder(s, 512)

    x = L.Concatenate()([x, s])
    x = convdecoder(x, 512)
    x = convdecoder(x, 512)

    x = deconv_block(x, 256)

    s = deconv_block(z6, 256)
    s = convdecoder(s, 256)
    s = deconv_block(s, 256)
    s = convdecoder(s, 256)

    x = L.Concatenate()([x, s])
    x = convdecoder(x, 256)
    x = convdecoder(x, 256)

    x = deconv_block(x, 128)

    s = deconv_block(z6, 128)
    s = convdecoder(s, 128)
    s = deconv_block(s, 128)
    s = convdecoder(s, 128)
    s = deconv_block(s, 128)
    s = convdecoder(s, 128)

    x = L.Concatenate()([x, s])
    x = convdecoder(x, 128)
    x = convdecoder(x, 128)

    x = deconv_block(x, 64)

    s = convdecoder(z0, 64)
    s = convdecoder(s, 64)
    x = L.Concatenate()([x, s])
    x = convdecoder(x, 64)
    x = convdecoder(x, 64)

    outputs = L.Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')(x)

    return Model(inputs, outputs, name="UNETR_2D")


config = {
    'image_size': 256,
    'num_layers': 12,
    'hidden_dim': 768,
    'mlp_dim': 3072,
    'num_heads': 12,
    'dropout_rate': 0.1,
    'patch_size': 16,
    'num_patch': 256,
    'num_channels': 3,
}

model = unetr2D(config)
model.summary()