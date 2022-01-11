package neural;

final class Line {
    final Weight weight;
    final Neuron left;
    final Neuron right;

    Line(Neuron left, Neuron right, Weight weight) {
        this.left = left;
        this.right = right;
        this.weight = weight;
    }
}
