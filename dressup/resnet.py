from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

rn50 = ResNet50(weights='imagenet')


def resnet_classifier(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = rn50.predict(x)
    return set([x[1] for x in decode_predictions(preds, top=2)[0]])


def resnet_filter(image_path, allowed_classes=set(['gown', 'mosquito_net', 'hoopskirt', 'groom'])):
    predictions = resnet_classifier(image_path)
    if len(allowed_classes.union(predictions)) > 0:
        return True
    else:
        return False


def resnet_classifier_img(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = rn50.predict(x)
    return set([x[1] for x in decode_predictions(preds, top=3)[0]])


def resnet_filter_img(img, allowed_classes=set(['gown', 'mosquito_net', 'hoopskirt', 'groom'])):
    predictions = resnet_classifier_img(img)
    if len(allowed_classes.intersection(predictions)) > 0:
        return " | ".join(list(predictions))
    else:
        return "nope"