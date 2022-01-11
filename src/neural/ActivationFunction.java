package neural;

/**
 * the function {@code: y = f(x)}, which has one input and one output
 */
public interface ActivationFunction {
    /**
     * calculate the output of the function
     *
     * @param input
     * @return the output
     */
    double apply(double input);

    /**
     * calculate the derivative of the function</br>
     * you'd better override it
     *
     * @param input
     * @return the derivative at specific input
     */
    default double derivative(double input) {
        final double smallValue = 1.0 / (1 << 24);
        return (apply(input + smallValue) - apply(input - smallValue)) / smallValue / 2.0;
    }
}
