package layers;

import java.util.ArrayList;
import java.util.List;

public abstract class Layer {

    protected Layer _nextLayer;
    protected Layer _previousLayer;


    public abstract double[] getOutput(List<double[][]> input);
    public abstract double[] getOutput(double[] input);
    
    public abstract void backPropagation(List<double[][]> dLd0);
    public abstract void backPropagation(double[] dLd0);

    public abstract int getOutputLength();
    public abstract int getOutputRows();
    public abstract int getOutputColumns();
    public abstract int getOutputElements();

                                                                // Getters and Setters
    public Layer get_nextLayer() {
        return _nextLayer;
    }
    public void set_nextLayer(Layer _nextLayer) {
        this._nextLayer = _nextLayer;
    }
    public Layer get_previousLayer() {
        return _previousLayer;
    }
    public void set_previousLayer(Layer _previousLayer) {
        this._previousLayer = _previousLayer;
    }

                                                                // Converters
    public double[] matrixToVector(List<double[][]> input) {
        
        int length = input.size();
        int rows = input.get(0).length;
        int coulmns = input.get(0)[0].length;
        
        double[] output = new double[length * rows * coulmns];

        int index = 0;
        for (double[][] matrix : input) {
            for (double[] row : matrix) {
                for (double value : row) {
                    output[index] = value;
                    index++;
                }
            }
        }
        return output;
    }

    public List<double[][]> vectorToMatrix(double[] input, int length, int rows, int columns) {
        
        List<double[][]> output = new ArrayList<>();

        int index = 0;
        for (int i = 0; i < length; i++) {
            double[][] matrix = new double[rows][columns];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    matrix[r][c] = input[index];
                    index++;
                }
            }
            output.add(matrix);
        }
        return output;
    }
    
}
