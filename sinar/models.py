from typing import Union
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from abc import ABC, abstractmethod

import sinar.logger
from sinar.utils import flatten_matrix, fill_square

logger = sinar.logger.get(__name__)

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


class BaseClassifierModel(ABC):
    """Base class for classifier models
    
    Methods:
        - predict(x): predict the class of input x
        - presquare(x): ensure x is square matrix, pad with zeros if not
        - load(path): factory method to load model from file

    """
    @abstractmethod
    def predict(self, x):
        pass

    @staticmethod
    def presquare(x: Union[np.ndarray, list]):
        if (isinstance(x, np.ndarray) and len(x.shape) == 3) or isinstance(x, list): # batch of matrices
            x = np.array([fill_square(xi) for xi in x])
        else:
            x = fill_square(x) # single matrix
            x = np.expand_dims(x, axis=0) # add batch dim
        return x
    
    def predict_batch(self, x):
        # implement in subclass if needed
        # alias for predict
        # x is a list of matrices, convert to numpy array
        x = self.presquare(x)
        x = np.array(x, dtype=np.float32)
        return self.predict(x).flatten()

    # factory method
    @classmethod
    def load(cls, path, *args, **kwargs):
        ext = path.split(".")[-1]
        match ext:
            case "h5" | "keras":
                return TFClassifierModel.load(path, *args, **kwargs)
            case "pth" | "pt":
                return TorchClassifierModel.load(path, *args, **kwargs)
            case "pkl" | "pickle":
                return ClassicMLModel.load(path, *args, **kwargs)
            case _:
                raise ValueError(f"Unsupported model format: {ext}")
        

class TFClassifierModel(BaseClassifierModel):
    """TensorFlow/Keras Classifier Model
    
    Args:
        model: a compiled keras model
    """
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        # ensure x shape is (N, 30, 30)
        x = self.presquare(x)
        return self.model.predict(x)

    @classmethod
    def load(cls, path, *args, **kwargs):
        model = tf.keras.models.load_model(path)
        return cls(model)
    
    def __repr__(self):
        self.model.summary()
        return f"<TFClassifierModel {self.model.name}>"


class TorchClassifierModel(BaseClassifierModel, nn.Module):
    """PyTorch Classifier Model
    Args:
        device: device to run the model on, default is "cpu"
    
    Note:
        The model architecture is fixed to a simple CNN for now.
        This model can accept variable input shape, as long as the second dimension is 30.
        Input shape is (batch, 30, N) or (30, N) for single sample.
        Output is class labels (0 or 1).
    """
    def __init__(self, device="cpu"):
        super().__init__()
        # validate device
        if isinstance(device, int):
            self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            try:
                self.device = torch.device(device)
            except Exception as e:
                logger.error(f"Invalid device string: {device} falling back to cpu")
                self.device = torch.device("cpu")
        self.to(self.device)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1,2), stride=(1,2), padding=(0,0)),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(2,1), stride=(2,1), padding=(0,0)),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=(5,5), stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        # self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(96, 2)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.gap(h).flatten(1)
        # h = self.dropout(h)
        return self.fc(h)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            # ensure x shape is (N, 1, 30, M)
            if len(x.shape) == 3:
                x = x.unsqueeze(1) # add channel dim
            elif len(x.shape) == 2:
                x = x.unsqueeze(0).unsqueeze(0) # add batch and channel dim
            x = x.to(self.device)
            logits = self.forward(x) # this will be logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs[:, 1] # return positive class probability
    
    def predict_batch(self, x):
        # x is a list of matrices
        # process one by one to avoid memory issue
        results = []
        for xi in x:
            preds = self.predict(xi)
            results.extend(preds)
        return np.array(results)
        
    
    @classmethod
    def load(cls, path, device="cpu", *args, **kwargs):
        model = cls(device=device)
        model.load_state_dict(torch.load(path, map_location=model.device))
        # model.to(device)
        model.eval()
        return model
    

class ClassicMLModel(BaseClassifierModel):
    """Classic ML Model using pickle serialization

    Note:
        added just for backward compatibility from past experiments
    """
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        x = self.presquare(x)
        x = flatten_matrix(x)
        # expand dims if 1D
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)
        return self.model.predict(x)

    @classmethod
    def load(cls, path, *args, **kwargs):
        with open(path, "rb") as f:
            model = pickle.load(f)
        return cls(model)
    
    def __repr__(self):
        return self.model.__repr__()


class OVClassifierModel(BaseClassifierModel):
    """OpenVINO Classifier Model
    Args:
        model: an OpenVINO model

    Note:
        Added for future compatibility with OpenVINO models
    """
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        # ensure x is 4D
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=(0,1)) # add batch and channel dim
        elif len(x.shape) == 3:
            x = np.expand_dims(x, axis=1) # add channel dim
        
        
        return self.model.predict(x)

    @classmethod
    def load(cls, path, *args, **kwargs):
        import openvino as ov
        core = ov.Core()
        model = core.read_model(model=path)
        compiled_model = core.compile_model(model=model)
        return cls(compiled_model)
    