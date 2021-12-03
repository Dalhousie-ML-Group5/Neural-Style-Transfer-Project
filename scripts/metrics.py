import cv2
import numpy as np
import tensorflow as tf


def gram_matrix(image):
    """Generate Gram Matrix of a given image."""
    channels = image.shape[3]
    x = tf.reshape(image, [-1, channels])
    gram = tf.matmul(tf.transpose(x), x)
    return gram


def content_fidelity(target_img, content_img, vgg) -> float:
    """
    Compute the Content Fidelity score of the target image.

    Parameters
    ----------
    target_img : tensorflow Tensor or Numpy array
        The resultant stylized image after combining content and style.

    content_img : tensorflow Tensor or Numpy array
        The content image that was used for neural style transfer.

    vgg : tensorflow Model
        A VGG16 pretrained model to be used for feature map extraction.

    Returns
    -------
    CF : float
        The Content Fidelity score.
    """
    N = len(vgg.layers)
    layer_outputs = [layer.output for layer in vgg.layers]
    cf_model = tf.keras.models.Model(inputs=vgg.input, outputs=layer_outputs)
    fmap_target = [cf_model.predict(target_img)[i] for i in range(N)]
    fmap_content = [cf_model.predict(content_img)[i] for i in range(N)]

    CF = 0
    for i in range(N):
        cosine_similarity = tf.keras.metrics.CosineSimilarity()
        cosine_similarity.update_state(fmap_target[i], fmap_content[i])
        CF += cosine_similarity.result().numpy()
    CF = CF/N
    return CF


def global_colors(target_img, style_img, verbose=False) -> float:
    """
    Compute the Global Colors score of the target image.

    Parameters
    ----------
    target_img : tensorflow Tensor or Numpy array
        The resultant stylized image after combining content and style.

    style_img : tensorflow Tensor or Numpy array
        The style image that was used for neural style transfer.

    verbose : bool
        If True, print out the shapes of input images.

    Returns
    -------
    GC : float
        The Global Colors score.
    """
    if verbose:
        print("target_img.shape :", target_img.shape)
        print("style_img.shape :", style_img.shape)

    X = np.array(target_img).reshape(target_img.shape[1:])
    S = np.array(style_img).reshape(style_img.shape[1:])

    GC = 0
    for cx, cs in zip(cv2.split(X), cv2.split(S)):
        histX = cv2.calcHist([cx], [0], None, [256], [0, 256])
        histX = cv2.normalize(histX, histX).flatten()

        histS = cv2.calcHist([cs], [0], None, [256], [0, 256])
        histS = cv2.normalize(histS, histS).flatten()

        cosine_similarity = tf.keras.metrics.CosineSimilarity()
        cosine_similarity.update_state(histX, histS)
        GC += cosine_similarity.result().numpy()
    GC = GC / len(cv2.split(X))
    return GC


def holistic_textures(target_img, style_img, vgg) -> float:
    """
    Compute the Holistic Texture score of the target image.

    Parameters
    ----------
    target_img : tensorflow Tensor or Numpy array
        The resultant stylized image after combining content and style.

    style_img : tensorflow Tensor or Numpy array
        The style image that was used for neural style transfer.

    vgg : tensorflow Model
        A VGG16 pretrained model to be used for feature map extraction.

    Returns
    -------
    HT : float
        The Holistic Texture score.
    """
    N = len(vgg.layers)
    layer_outputs = [layer.output for layer in vgg.layers]
    ht_model = tf.keras.models.Model(inputs=vgg.input, outputs=layer_outputs)
    # Feature maps
    fmap_target = [ht_model.predict(target_img)[i] for i in range(N)]
    fmap_style = [ht_model.predict(style_img)[i] for i in range(N)]
    # Gram matrices
    gm_target = [gram_matrix(fm) for fm in fmap_target]
    gm_style = [gram_matrix(fm) for fm in fmap_style]

    HT = 0
    for i in range(N):
        cosine_similarity = tf.keras.metrics.CosineSimilarity()
        cosine_similarity.update_state(gm_target[i], gm_style[i])
        HT += cosine_similarity.result().numpy()
    HT = HT/N
    return HT


def global_effects(target_img, style_img, vgg):
    """The average of the Global Colors and Holistic Textures."""
    GC = global_colors(target_img, style_img)
    HT = holistic_textures(target_img, style_img, vgg)
    GE = 1/2*(GC + HT)
    return GE


if __name__ == '__main__':
    print("MAIN!")
