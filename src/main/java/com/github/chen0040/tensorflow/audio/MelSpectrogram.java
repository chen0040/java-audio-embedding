package com.github.chen0040.tensorflow.audio;

import be.tarsos.dsp.AudioDispatcher;
import be.tarsos.dsp.AudioEvent;
import be.tarsos.dsp.AudioProcessor;
import be.tarsos.dsp.io.jvm.AudioDispatcherFactory;
import be.tarsos.dsp.io.jvm.AudioPlayer;
import be.tarsos.dsp.pitch.PitchDetectionHandler;
import be.tarsos.dsp.pitch.PitchDetectionResult;
import be.tarsos.dsp.pitch.PitchProcessor;
import be.tarsos.dsp.util.fft.FFT;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.UnsupportedAudioFileException;
import java.io.File;
import java.io.IOException;

public class MelSpectrogram  implements PitchDetectionHandler {

    private AudioDispatcher dispatcher;

    private float sampleRate = 44100;
    private int bufferSize = 1024 * 4;
    private int overlap = 768 * 4 ;

    private double pitch;

    private PitchProcessor.PitchEstimationAlgorithm algo;

    AudioProcessor fftProcessor = new AudioProcessor(){

        FFT fft = new FFT(bufferSize);
        float[] amplitudes = new float[bufferSize/2];

        public void processingFinished() {
            // TODO Auto-generated method stub
        }

        public boolean process(AudioEvent audioEvent) {
            float[] audioFloatBuffer = audioEvent.getFloatBuffer();
            float[] transformbuffer = new float[bufferSize*2];
            System.arraycopy(audioFloatBuffer, 0, transformbuffer, 0, audioFloatBuffer.length);
            fft.forwardTransform(transformbuffer);
            fft.modulus(transformbuffer, amplitudes);
            drawFFT(pitch, amplitudes,fft);
            return true;
        }

    };

    private void drawFFT(double pitch, float[] amplitudes, FFT fft) {

    }

    public void loadAudio(File audioFile, String name) throws IOException, UnsupportedAudioFileException, LineUnavailableException {

        if(dispatcher!= null){
            dispatcher.stop();
        }

        PitchProcessor.PitchEstimationAlgorithm newAlgo = PitchProcessor.PitchEstimationAlgorithm.valueOf(name);
        algo = newAlgo;


        dispatcher = AudioDispatcherFactory.fromFile(audioFile, bufferSize, overlap);
        AudioFormat format = AudioSystem.getAudioFileFormat(audioFile).getFormat();
        dispatcher.addAudioProcessor(new AudioPlayer(format));

        // add a processor, handle pitch event.
        dispatcher.addAudioProcessor(new PitchProcessor(algo, sampleRate, bufferSize, this));
        dispatcher.addAudioProcessor(fftProcessor);

        // run the dispatcher (on a new thread).
        new Thread(dispatcher,"Audio dispatching").start();
    }

    public void handlePitch(PitchDetectionResult pitchDetectionResult,AudioEvent audioEvent) {
        if(pitchDetectionResult.isPitched()){
            pitch = pitchDetectionResult.getPitch();
        } else {
            pitch = -1;
        }

    }
}
