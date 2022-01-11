package neural.costfunctions;

import neural.CostFunction;
import neural.NeuronLayer;

/**
 * the function of 'Least Square Method'
 */
public class SquareCostFunction implements CostFunction {

    @Override
    public double apply(NeuronLayer outputLayer, double[] outputs) {
        assert (outputLayer.size() == outputs.length);
        double result = 0.0;
        int len = outputs.length;
        for (int i = 0; i < len; i++) {
            double delta = outputLayer.get(i).getOutput() - outputs[i];
            result += delta * delta;
        }
        return result / 2.0;
    }

    @Override
    public double partialDerivative(int index, NeuronLayer outputLayer, double[] outputs) {
        assert (outputLayer.size() == outputs.length);
        return outputLayer.get(index).getOutput() - outputs[index];
    }

}
