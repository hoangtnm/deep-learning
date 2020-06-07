from research.action_recognition.models import TFC3D, LRMobileNetV2


def build_model(name=None, classes=101, dropout=0.5, weights=None, **kwargs):
    """Instantiates model with the given arguments.

    Args:
        name: One of `c3d` or `lrmobilenet_v2`.
        classes: Number of classes to classify videos into.
        dropout: Float between 0 and 1. Fraction of the input units to drop.
        weights: Path to the weights file to load.

    Returns:
        A `keras.Model` instance.
    """
    model = None

    if name == 'c3d':
        model = TFC3D(classes, dropout, **kwargs)
    elif name == 'lrmobilenet_v2':
        model = LRMobileNetV2(classes=classes, dropout=dropout, **kwargs)
    assert model is not None, 'This model is not supported'

    if (weights is not None) and (weights != 'imagenet'):
        model.load_weights(weights)

    return model
