package neural;

import neural.activationfunctions.ReLUFunction;

import java.util.*;
import java.util.Map.Entry;
import java.util.function.Function;

public class NeuronNetwork {
    private final List<NeuronLayer> layers = new ArrayList<>();
    private final CostFunction costFunction;
    private final double learningRate;
    private final double L2Lambda;
    private final Function<Integer, Double> randomGenerator;
    private NeuronLayer inputLayer;
    private NeuronLayer outputLayer;

    public NeuronNetwork(CostFunction costFunction, double learningRate) {
        this(costFunction, learningRate, 0, new GaussianRandomGenerator());
    }

    public NeuronNetwork(CostFunction costFunction, double learningRate, double L2Lambda) {
        this(costFunction, learningRate, L2Lambda, new GaussianRandomGenerator());
    }

    public NeuronNetwork(CostFunction costFunction, double learningRate, Function<Integer, Double> randomGenerator) {
        this(costFunction, learningRate, 0, randomGenerator);
    }

    public NeuronNetwork(CostFunction costFunction, double learningRate, double L2Lambda, Function<Integer, Double> randomGenerator) {
        this.costFunction = costFunction;
        this.learningRate = learningRate;
        this.L2Lambda = L2Lambda;
        this.randomGenerator = randomGenerator;
    }

    public void addLayerToBack(NeuronLayer layer) {
        layers.add(layer);
    }

    public void setInputLayer(NeuronLayer layer) {
        this.inputLayer = layer;
    }

    public void setOutputLayer(NeuronLayer layer) {
        this.outputLayer = layer;
    }

    public double[] apply(double[] inputs) {
        assert (inputs.length == inputLayer.size());
        // input
        for (int j = 0; j < inputs.length; j++)
            inputLayer.get(j).output = inputs[j];
        // forward propagation
        for (NeuronLayer layer : layers) {
            if (layer != inputLayer) {
                for (Neuron neuron : layer)
                    neuron.apply();
            }
        }
        // the end
        double[] outputs = new double[outputLayer.size()];
        for (int j = 0; j < outputLayer.size(); j++)
            outputs[j] = outputLayer.get(j).output;
        return outputs;
    }

    public void train(List<Trainingset> trainingsets) {
        Map<Weight, Double> weightDelta = new HashMap<>();
        for (Trainingset trainingset : trainingsets) {
            double[] inputs = trainingset.input;
            double[] outputs = trainingset.output;
            assert (inputs.length == inputLayer.size());
            // input
            for (int j = 0; j < inputs.length; j++)
                inputLayer.get(j).output = inputs[j];
            // forward propagation
            for (NeuronLayer layer : layers) {
                if (layer != inputLayer) {
                    for (Neuron neuron : layer)
                        neuron.apply();
                }
            }
            // calculate output layer delta
            for (int j = 0; j < outputLayer.size(); j++) {
                Neuron neuron = outputLayer.get(j);
                neuron.delta = costFunction.partialDerivative(j, outputLayer, outputs) * neuron.activationFunction.derivative(neuron.input);
            }
            // backward propagation
            for (int j = layers.size() - 1; j >= 0; j--) {
                NeuronLayer layer = layers.get(j);
                if (layer != outputLayer && layer != inputLayer) {
                    for (Neuron neuron : layer)
                        neuron.calDelta();
                }
            }
            // the end
            for (NeuronLayer layer : layers) {
                if (layer != outputLayer) {
                    for (Neuron neuron : layer) {
                        for (Line line : neuron.outputLines) {
                            double delta = line.left.output * line.right.delta;
                            if (weightDelta.containsKey(line.weight))
                                weightDelta.put(line.weight, weightDelta.get(line.weight) + delta);
                            else
                                weightDelta.put(line.weight, delta);
                        }
                    }
                }
            }
        }
        // modify the weights
        for (Entry<Weight, Double> entry : weightDelta.entrySet()) {
            if (L2Lambda != 0 && entry.getKey().enableL2)
                entry.getKey().reducePer(L2Lambda * learningRate);
            entry.getKey().reduce(entry.getValue() / trainingsets.size() * learningRate);
        }
    }

    public void fullConnect(NeuronLayer layer1, NeuronLayer layer2) {
        int inCount = layer1.size();
        for (Neuron neuron1 : layer1) {
            for (Neuron neuron2 : layer2) {
                Line line = new Line(neuron1, neuron2, new Weight(randomGenerator.apply(inCount), true));
                neuron1.outputLines.add(line);
                neuron2.inputLines.add(line);
            }
        }
    }

    public void oneToOneConnect(NeuronLayer layer1, NeuronLayer layer2) {
        assert (layer1.size() == layer2.size());
        int len = layer1.size();
        for (int i = 0; i < len; i++) {
            Neuron neuron1 = layer1.get(i);
            Neuron neuron2 = layer2.get(i);
            Line line = new Line(neuron1, neuron2, new Weight(randomGenerator.apply(1), false));
            neuron1.outputLines.add(line);
            neuron2.inputLines.add(line);
        }
    }

    public TwoDimensionNeuronLayer convolute(TwoDimensionNeuronLayer layer1, int width, int height) {
        Weight[][] weights = new Weight[height][width];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                weights[i][j] = new Weight(randomGenerator.apply(width * height), true);
            }
        }
        BiasNeuron bias = new BiasNeuron();
        TwoDimensionNeuronLayer layer2 = new TwoDimensionNeuronLayer(layer1.width - width + 1, layer1.height - height + 1, new ReLUFunction());
        for (int ii = 0; ii < layer2.height; ii++) {
            for (int jj = 0; jj < layer2.width; jj++) {
                Neuron neuron2 = layer2.get(ii, jj);
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        Neuron neuron1 = layer1.get(i + ii, j + jj);
                        Line line = new Line(neuron1, neuron2, weights[i][j]);
                        neuron1.outputLines.add(line);
                        neuron2.inputLines.add(line);
                    }
                }
                Line line = new Line(bias, neuron2, new Weight(randomGenerator.apply(1), false));
                bias.outputLines.add(line);
                neuron2.inputLines.add(line);
            }
        }
        return layer2;
    }

    private static class GaussianRandomGenerator implements Function<Integer, Double> {
        private final Random rand = new Random();

        @Override
        public Double apply(Integer t) {
            return rand.nextGaussian();
        }

    }
}
