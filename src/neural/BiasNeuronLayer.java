package neural;

public class BiasNeuronLayer extends NeuronLayer {
    public BiasNeuronLayer(int length) {
        super(length, new ActivationFunction() {
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
}
