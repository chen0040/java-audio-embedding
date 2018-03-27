from random import shuffle
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

from demo.gtzan_utils import gtzan_labels


def load_audio_path_label_pairs(max_allowed_pairs=None):
    audio_paths = []
    with open('./data/lists/test_songs_gtzan_list.txt', 'rt') as file:
        for line in file:
            audio_path = '../' + line.strip()
            audio_paths.append(audio_path)
    pairs = []
    with open('./data/lists/test_gt_gtzan_list.txt', 'rt') as file:
        for line in file:
            label = int(line)
            if max_allowed_pairs is None or len(pairs) < max_allowed_pairs:
                pairs.append((audio_paths[len(pairs)], label))
            else:
                break
    return pairs


def melgram(audio_path):
    img_path = audio_path + '.png'
    img = img_to_array(load_img(img_path, grayscale=True))
    return img


def main():
    audio_path_label_pairs = load_audio_path_label_pairs()
    shuffle(audio_path_label_pairs)
    print('loaded: ', len(audio_path_label_pairs))

    with tf.gfile.FastGFile('./models/tensorflow_models/cifar10/cifar10.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        [print(n.name) for n in sess.graph.as_graph_def().node]
        predict_op = sess.graph.get_tensor_by_name('output_node0:0')
        for i in range(0, 20):
            audio_path, actual_label_id = audio_path_label_pairs[i]

            mg = melgram(audio_path)
            mg = np.expand_dims(mg, axis=0)

            predicted = sess.run(predict_op, feed_dict={"conv2d_1_input:0": mg})

            predicted_label_idx = np.argmax(predicted, axis=1)[0]
            predicted_label = gtzan_labels[predicted_label_idx]
            actual_label = gtzan_labels[actual_label_id]

            print('predicted: ', predicted_label, 'actual: ', actual_label)


if __name__ == '__main__':
    main()
