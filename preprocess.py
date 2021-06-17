import numpy as np
import sys
import os 
import json
from sklearn.utils import shuffle
from tqdm import tqdm_notebook as tqdm


annotation_file = '/content/annotations/captions_train2014.json'
PATH = '/content/train2014/'

with open(annotation_file, 'r') as f:
    annotations = json.load(f)

captions = []
vectors = []

for annot in annotations['annotations']:
    caption = 'ssss ' + annot['caption'] + ' eeee'
    image_id = annot['image_id']
    path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

    vectors.append(path)
    captions.append(caption)

train_captions, img_name_vector = shuffle(captions, vectors,
                                          random_state=1)

num_examples = 10000
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]

target_size = (299, 299, 3)

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (target_size[0], target_size[1]))
    
    img = efn.preprocess_input(img)
    
    return img, image_path
  
  #Feature Extractor

image_model = image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                 weights='imagenet')

new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# Get unique images
encode_train = sorted(set(img_name_vector))

feature_dict = {}

# Feel free to change batch_size according to your system configuration
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

for img, path in tqdm(image_dataset):
  batch_features = image_features_extract_model(img)
  batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))

  for bf, p in zip(batch_features, path):
    path_of_feature = p.numpy().decode("utf-8")
    feature_dict[path_of_feature] =  bf.numpy()
