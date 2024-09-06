import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore

def create_input_layers():
    inp = layers.Input(shape=(30, 30), name="sinar_input") # 30x30 input
    x = layers.Reshape((30, 30, 1))(inp)
    x = layers.ZeroPadding2D(padding=(1, 1))(x) # pad input to 32x32
    return inp, x

def create_output_layers(model_name, inp):
    x = layers.GlobalAveragePooling2D()(inp)
    x = layers.Dense(512, activation='relu', name=f'{model_name}_dense_1')(x)
    out = layers.Dense(1, activation='sigmoid', name=f'{model_name}_output')(x)
    return out


def create_sinar_vgg16(compile=True):
    inp, x = create_input_layers()
    vgg16 = tf.keras.applications.VGG16(include_top=False, weights=None, input_shape=(32, 32, 1))(x)
    out = create_output_layers("sinar_vgg16", vgg16)
    sinar_vgg16 = models.Model(inputs=inp, outputs=out, name='sinar-vgg16')

    if compile:
        sinar_vgg16.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    return sinar_vgg16


def create_sinar_resnet50(compile=True):
    inp, x = create_input_layers()
    resnet = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(32, 32, 1))(x)
    out = create_output_layers("sinar_resnet50", resnet)
    sinar_resnet50 = models.Model(inputs=inp, outputs=out, name='sinar-resnet50')

    if compile:
        sinar_resnet50.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    return sinar_resnet50


def create_sinar_mobilenet(compile=True):
    inp, x = create_input_layers()
    mobilenet = tf.keras.applications.MobileNetV2(include_top=False, weights=None, input_shape=(32, 32, 1))(x)
    out = create_output_layers("sinar_mobilenet", mobilenet)
    sinar_mobilenet = models.Model(inputs=inp, outputs=out, name='sinar-mobilenet')

    if compile:
        sinar_mobilenet.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    return sinar_mobilenet

