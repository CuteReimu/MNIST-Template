package neural;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * an 1-dimension Neuron Layer
 */
public class NeuronLayer implements Iterable<Neuron> {
    protected final List<Neuron> list;

    /**
     * create an 1-dimension Neuron Layer with specific length and activation function
     *
     * @param length
     * @param activationFunction
     */
    public NeuronLayer(int length, ActivationFunction activationFunction) {
        Neuron[] neurons = new Neuron[length];
        for (int i = 0; i < neurons.length; i++)
            neurons[i] = new Neuron(activationFunction);
        list = Arrays.asList(neurons);
    }

    /**
     * create an 1-dimension Neuron Layer with specific neurons
     *
     * @param neurons
     */
    public NeuronLayer(Neuron... neurons) {
        list = Arrays.asList(neurons);
    }

    @Override
    public Iterator<Neuron> iterator() {
        return list.iterator();
    }

    public int size() {
        return list.size();
    }

    public Neuron get(int index) {
        return list.get(index);
    }

}
