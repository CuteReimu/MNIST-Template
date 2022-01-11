package neural;

public class BiasNeuron extends Neuron {

    public BiasNeuron() {
        super(new ActivationFunction() {
            @Override
            public double apply(double input) {
                return 1.0;
            }

            @Override
            public double derivative(double input) {
                return 0.0;
            }
        });
    }

    @Override
    public double getOutput() {
        return 1.0;
    }

    @Override
    protected void apply() {
    }

    @Override
    protected void calDelta() {
    }

}
