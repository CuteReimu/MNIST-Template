package neural;

/**
 * an 2-dimension Neuron Layer
 */
public class TwoDimensionNeuronLayer extends NeuronLayer {
    final int width;
    final int height;

    /**
     * create an 2-dimension Neuron Layer with specific width, height and activation function
     *
     * @param width
     * @param height
     * @param activationFunction
     */
    public TwoDimensionNeuronLayer(int width, int height, ActivationFunction activationFunction) {
        super(width * height, activationFunction);
        this.width = width;
        this.height = height;
    }

    /**
     * create an 2-dimension Neuron Layer with specific neurons
     *
     * @param neurons
     */
    public TwoDimensionNeuronLayer(Neuron[][] neurons) {
        super(new Neuron[neurons.length * neurons[0].length]);
        height = neurons.length;
        width = neurons[0].length;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                list.set(i * width + j, neurons[i][j]);
            }
        }
    }

    public Neuron get(int i, int j) {
        return get(i * width + j);
    }

}
