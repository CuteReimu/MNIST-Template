package neural.costfunctions;

import neural.CostFunction;
import neural.NeuronLayer;

public class CrossEntropyCostFunction implements CostFunction {

    @Override
    public double apply(NeuronLayer outputLayer, double[] outputs) {
        assert (outputLayer.size() == outputs.length);
        double result = 0.0;
        int len = outputs.length;
        for (int i = 0; i < len; i++) {
            double a = outputLayer.get(i).getOutput(), y = outputs[i];
            assert (y == 1 || y == 0);
            result -= y * Math.log(a) + (1 - y) * Math.log(1 - a);
        }
        return result;
    }

    @Override
    public double partialDerivative(int index, NeuronLayer outputLayer, double[] outputs) {
        assert (outputLayer.size() == outputs.length);
        double a = outputLayer.get(index).getOutput(), y = outputs[index];
        assert (y == 1 || y == 0);
        return -y / a - (1 - y) / (a - 1);
    }

}
