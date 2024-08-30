import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore

def create_tops(model_name):
    top = [
            layers.Dense(512, activation='relu', name=f'{model_name}_dense_1'),
            layers.Dense(128, activation='relu', name=f'{model_name}_dense_2'),
            layers.Dense(32, activation='relu', name=f'{model_name}_dense_3'),
            layers.Dense(1, activation='sigmoid', name=f'{model_name}_output')
        ]
    return top


def create_sinar_vgg16(compile=True):
    sinar_top = create_tops('sinar_vgg16')
    sinar_vgg16 = models.Sequential(
        [
            tf.keras.applications.VGG16(include_top=False, weights=None, input_shape=(32, 32, 1)),
            layers.GlobalAveragePooling2D(),
            *sinar_top
        ],
        name='sinar-vgg16'
    )

    if compile:
        sinar_vgg16.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    return sinar_vgg16


def create_sinar_resnet50(complie=True):
    sinar_top = create_tops('sinar_resnet50')
    sinar_resnet50 = models.Sequential(
        [
            tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(32, 32, 1)),
            layers.GlobalAveragePooling2D(),
            *sinar_top
        ],
        name='sinar-resnet50'
    )

    if complie:
        sinar_resnet50.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    return sinar_resnet50


def create_sinar_mobilenet(complie=True):
    sinar_top = create_tops('sinar_mobilenet')
    sinar_mobilenet = models.Sequential(
        [
            tf.keras.applications.MobileNet(include_top=False, weights=None, input_shape=(32, 32, 1)),
            layers.GlobalAveragePooling2D(),
            *sinar_top
        ],
        name='sinar-mobilenet'
    )

    if complie:
        sinar_mobilenet.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    return sinar_mobilenet

