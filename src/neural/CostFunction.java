package neural;

/**
 * the function to calculate the deviation of the output
 */
public interface CostFunction {
    /**
     * calculate the deviation
     *
     * @param outputLayer   the output layer of the Neuron Network
     * @param outputs       the outputs of the training sets
     * @return
     */
    double apply(NeuronLayer outputLayer, double[] outputs);

    /**
     * calculate the partial derivative of the function at the specific output value</br>
     * you'd better override it
     *
     * @param index
     * @param outputLayer
     * @param outputs
     * @return the partial derivative at specific output value
     */
    default double partialDerivative(int index, NeuronLayer outputLayer, double[] outputs) {
        final double smallValue = 1.0 / (1 << 24);
        outputLayer.get(index).output += smallValue;
        double apply1 = apply(outputLayer, outputs);
        outputLayer.get(index).output -= smallValue * 2;
        double apply2 = apply(outputLayer, outputs);
        outputLayer.get(index).output += smallValue;
        return (apply1 - apply2) / smallValue / 2.0;
    }
}
