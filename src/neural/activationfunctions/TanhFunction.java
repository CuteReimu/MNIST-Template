package neural.activationfunctions;

import neural.ActivationFunction;

/**
 * the function {@code: y = tanh(x) = (exp(x)-exp(-x)) / (exp(x)+exp(-x))}
 * <li>when x is -infinity, y equals -1</li>
 * <li>when x is 0, y equals 0</li>
 * <li>when x is +infinity, y equals 1</li>
 */
public class TanhFunction implements ActivationFunction {

    @Override
    public double apply(double input) {
        double a = Math.exp(input), b = 1.0 / a;
        return (a - b) / (a + b);
    }

    @Override
    public double derivative(double input) {
        double y = apply(input);
        return 1 - y * y;
    }
}
