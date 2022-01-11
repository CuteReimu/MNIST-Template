package neural.activationfunctions;

import neural.ActivationFunction;

/**
 * the function y = 1 / (1 + exp(-x))
 * <li>when x is -infinity, y equals 0</li>
 * <li>when x is 0, y equals 0.5</li>
 * <li>when x is +infinity, y equals 1</li>
 */
public class SigmoidFunction implements ActivationFunction {

    @Override
    public double apply(double input) {
        return 1.0 / (1.0 + Math.exp(-input));
    }

    @Override
    public double derivative(double input) {
        double y = apply(input);
        return y * (1 - y);
    }
}
