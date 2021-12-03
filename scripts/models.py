from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19


def load_vgg(model="vgg16", verbose=False):
    """Load pre-trained """
    if model == "vgg16":
        vgg = VGG16(weights='imagenet', include_top=False)
    elif model == "vgg19":
        vgg = VGG19(weights='imagenet', include_top=False)
    else:
        raise ValueError("""Unknown model name! Acceptable names: {"vgg16", "vgg19"}""")
    if verbose:
        vgg.summary()
    return vgg


def load_depth_model(model_path="../models/monocular-depth-estimation-model/", verbose=False):
    """Load pre-trained depth estimation model."""
    depth_model = load_model(model_path)
    if verbose:
        depth_model.summary()
    return depth_model

