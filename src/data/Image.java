package data;

public class Image {
    
    private final double[][] data;
    private final int label;
    
    public Image(double[][] data, int label) {
        this.data = data;
        this.label = label;
    }

    public double[][] getData() {
        return data;
    }

    public int getLabel() {
        return label;
    }

    @Override
    public String toString() {
        
        String string = label + "\n";

        for (double[] row : data) {
            for (double value : row) {
                string += value + ", ";
            }
            string += "\n";
        }
        return string;
    }    

}