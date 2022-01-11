package neural;

import java.util.ArrayList;
import java.util.List;

public class Neuron {
    protected final List<Line> outputLines = new ArrayList<>();
    protected final List<Line> inputLines = new ArrayList<>();
    final ActivationFunction activationFunction;
    protected double input;
    protected double output;
    protected double delta;

    public Neuron(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    protected void apply() {
        input = 0;
        for (Line line : inputLines) {
            input += line.left.output * line.weight.get();
        }
        output = activationFunction.apply(input);
    }

    protected void calDelta() {
        delta = 0;
        for (Line line : outputLines) {
            delta += line.right.delta * line.weight.get();
        }
        delta *= activationFunction.derivative(input);
    }

    public double getInput() {
        return input;
    }

    public double getOutput() {
        return output;
    }
}
