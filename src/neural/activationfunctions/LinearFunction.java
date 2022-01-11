package neural.activationfunctions;

import neural.ActivationFunction;

/**
 * the function y = x
 */
public class LinearFunction implements ActivationFunction {

    @Override
    public double apply(double input) {
        return input;
    }

    @Override
    public double derivative(double input) {
        return 1;
    }

}
