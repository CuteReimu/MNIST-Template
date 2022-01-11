package neural.activationfunctions;

import neural.ActivationFunction;

/**
 * the function {@code: y = max(0, x)}
 * <li>when x <= 0, y = 0</li>
 * <li>when x >= 0, y = x</li>
 */
public class ReLUFunction implements ActivationFunction {

    @Override
    public double apply(double input) {
        return input > 0 ? input : 0;
    }

    @Override
    public double derivative(double input) {
        return input > 0 ? 1 : (input < 0 ? 0 : 0.5);
    }
}
