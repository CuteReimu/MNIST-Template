package neural;

class Weight {
    public final boolean enableL2;
    private double val;

    public Weight(double val, boolean enableL2) {
        this.val = val;
        this.enableL2 = enableL2;
    }

    public double get() {
        return val;
    }

    public void reduce(double val) {
        this.val -= val;
    }

    public void reducePer(double per) {
        this.val *= 1 - per;
    }
}
