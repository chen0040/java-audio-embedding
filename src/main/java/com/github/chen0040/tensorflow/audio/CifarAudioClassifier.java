package com.github.chen0040.tensorflow.audio;

import com.github.chen0040.tensorflow.audio.consts.MelSpectrogramDimension;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

@Getter
@Setter
public class CifarAudioClassifier {
    int height = MelSpectrogramDimension.Height;
    int width = MelSpectrogramDimension.Width;
    int channels = 3;
    int numTrainSamples = 100;
    int numTestSamples = 100;
    int batchSize = 30;

    int outputNum = 10;
    int iterations = 5;
    int epochs = 5;
    int seed = 123;
    int listenerFreq = 5;

    private int numLabels = 10;

    private static final Logger logger = LoggerFactory.getLogger(CifarAudioClassifier.class);

    public CifarAudioClassifier(){

    }
    

    private MultiLayerNetwork createModel() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .cacheMode(CacheMode.DEVICE)
                .updater(Updater.ADAM)
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l1(1e-4)
                .regularization(true)
                .l2(5 * 1e-4)
                .list()
                .layer(0, new ConvolutionLayer.Builder(new int[]{4, 4}, new int[]{1, 1}, new int[]{0, 0}).name("cnn1").convolutionMode(ConvolutionMode.Same)
                        .nIn(3).nOut(64).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)//.learningRateDecayPolicy(LearningRatePolicy.Step)
                        .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2*2).build())
                .layer(1, new ConvolutionLayer.Builder(new int[]{4,4}, new int[] {1,1}, new int[] {0,0}).name("cnn2").convolutionMode(ConvolutionMode.Same)
                        .nOut(64).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                        .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2*2).build())
                .layer(2, new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}).name("maxpool2").build())

                .layer(3, new ConvolutionLayer.Builder(new int[]{4,4}, new int[] {1,1}, new int[] {0,0}).name("cnn3").convolutionMode(ConvolutionMode.Same)
                        .nOut(96).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                        .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2*2).build())
                .layer(4, new ConvolutionLayer.Builder(new int[]{4,4}, new int[] {1,1}, new int[] {0,0}).name("cnn4").convolutionMode(ConvolutionMode.Same)
                        .nOut(96).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                        .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2*2).build())

                .layer(5, new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {0,0}).name("cnn5").convolutionMode(ConvolutionMode.Same)
                        .nOut(128).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                        .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2*2).build())
                .layer(6, new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {0,0}).name("cnn6").convolutionMode(ConvolutionMode.Same)
                        .nOut(128).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                        .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2*2).build())

                .layer(7, new ConvolutionLayer.Builder(new int[]{2,2}, new int[] {1,1}, new int[] {0,0}).name("cnn7").convolutionMode(ConvolutionMode.Same)
                        .nOut(256).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                        .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2*2).build())
                .layer(8, new ConvolutionLayer.Builder(new int[]{2,2}, new int[] {1,1}, new int[] {0,0}).name("cnn8").convolutionMode(ConvolutionMode.Same)
                        .nOut(256).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                        .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2*2).build())
                .layer(9, new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}).name("maxpool8").build())

                .layer(10, new DenseLayer.Builder().name("ffn1").nOut(1024).learningRate(1e-3).biasInit(1e-3).biasLearningRate(1e-3*2).build())
                .layer(11,new DropoutLayer.Builder().name("dropout1").dropOut(0.2).build())
                .layer(12, new DenseLayer.Builder().name("ffn2").nOut(1024).learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2*2).build())
                .layer(13,new DropoutLayer.Builder().name("dropout2").dropOut(0.2).build())
                .layer(14, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

    public MultiLayerNetwork saveModel(MultiLayerNetwork model, File locationModelFile) {
        boolean saveUpdater = false;
        try {
            ModelSerializer.writeModel(model,locationModelFile,saveUpdater);
        } catch (Exception e) {
            logger.error("Saving model is not success !",e);
        }
        return model;
    }
}
