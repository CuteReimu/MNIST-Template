package neural.impl;

import neural.BiasNeuronLayer;
import neural.NeuronLayer;
import neural.NeuronNetwork;
import neural.Trainingset;
import neural.activationfunctions.SigmoidFunction;
import neural.costfunctions.CrossEntropyCostFunction;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

public class Main {
    public static void main(String[] args) throws IOException {
        long time = System.currentTimeMillis();
        String dir = "./";
        List<Trainingset> trainingsets = new ArrayList<>(60000);
        try (DataInputStream is = new DataInputStream(new BufferedInputStream(new FileInputStream(dir + "train-labels.idx1-ubyte")))) {
            try (DataInputStream is2 = new DataInputStream(new BufferedInputStream(new FileInputStream(dir + "train-images.idx3-ubyte")))) {
                is.readInt();
                is.readInt();
                is2.readInt();
                is2.readInt();
                is2.readInt();
                is2.readInt();
                int label;
                while ((label = is.read()) != -1) {
                    double[] inputs = new double[28 * 28];
                    for (int i = 0; i < 28 * 28; i++)
                        inputs[i] = is2.read() / 255.0;
                    double[] outputs = new double[10];
                    outputs[label] = 1;
                    trainingsets.add(new Trainingset(inputs, outputs));
                }
            }
        }

        List<Trainingset> testingsets = new ArrayList<>(10000);
        try (DataInputStream is = new DataInputStream(new BufferedInputStream(new FileInputStream(dir + "t10k-labels.idx1-ubyte")))) {
            try (DataInputStream is2 = new DataInputStream(new BufferedInputStream(new FileInputStream(dir + "t10k-images.idx3-ubyte")))) {
                is.readInt();
                is.readInt();
                is2.readInt();
                is2.readInt();
                is2.readInt();
                is2.readInt();
                int label;
                while ((label = is.read()) != -1) {
                    double[] inputs = new double[28 * 28];
                    for (int i = 0; i < 28 * 28; i++)
                        inputs[i] = is2.read() / 255.0;
                    double[] outputs = new double[10];
                    outputs[label] = 1;
                    testingsets.add(new Trainingset(inputs, outputs));
                }
            }
        }

        NeuronNetwork nn = new NeuronNetwork(new CrossEntropyCostFunction(), 0.5, 0, new Function<>() {
            private final Random rand = new Random();

            @Override
            public Double apply(Integer inputCount) {
                return rand.nextGaussian();
            }
        });
        NeuronLayer inputLayer = new NeuronLayer(28 * 28, new SigmoidFunction());
        nn.addLayerToBack(inputLayer);
        NeuronLayer hiddenBiasLayer = new BiasNeuronLayer(30);
        NeuronLayer hiddenLayer = new NeuronLayer(30, new SigmoidFunction());
        NeuronLayer outputLayer = new NeuronLayer(10, new SigmoidFunction());
        NeuronLayer outputBiasLayer = new BiasNeuronLayer(10);
        nn.oneToOneConnect(hiddenBiasLayer, hiddenLayer);
        nn.oneToOneConnect(outputBiasLayer, outputLayer);
        nn.fullConnect(inputLayer, hiddenLayer);
        nn.fullConnect(hiddenLayer, outputLayer);
        nn.addLayerToBack(hiddenLayer);
        nn.addLayerToBack(outputLayer);
        nn.setInputLayer(inputLayer);
        nn.setOutputLayer(outputLayer);

        Random rand = new Random();
        for (int i = 0; i < 6000; i++) {
            List<Trainingset> trainingsets2 = new ArrayList<>(10);
            for (int j = 0; j < 10; j++)
                trainingsets2.add(trainingsets.get(rand.nextInt(trainingsets.size())));
            nn.train(trainingsets2);
            if ((i + 1) % 10 == 0) System.out.println("progress: " + (i + 1) + "/" + 6000);
        }

        int correct = 0;
        for (Trainingset trainingset : testingsets) {
            double[] output = nn.apply(trainingset.input);
            int i, j;
            for (j = 0; j < trainingset.output.length; j++)
                if (trainingset.output[j] == 1) break;
            double max = -1.0;
            int maxi = -1;
            for (i = 0; i < output.length; i++) {
                if (output[i] > max) {
                    max = output[i];
                    maxi = i;
                }
            }
            if (j == maxi) correct++;
        }
        System.out.println("result:" + correct + "/" + testingsets.size());
        System.out.println("time:" + (System.currentTimeMillis() - time));
    }
}
