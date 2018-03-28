from keras_audio.library.resnet_v2 import ResNetV2AudioClassifier


def main():

    classifier = ResNetV2AudioClassifier()
    classifier.load_model(model_dir_path='./models')

    classifier.export_tensorflow_model(output_fld='./models/tensorflow_models/resnet_v2')


if __name__ == '__main__':
    main()
