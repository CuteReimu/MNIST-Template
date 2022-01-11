package neural;

public final class Trainingset {
    public final double[] input;
    public final double[] output;

    public Trainingset(double[] input, double[] output) {
        this.input = input;
        this.output = output;
    }
}
