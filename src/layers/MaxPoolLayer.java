package layers;

import java.util.ArrayList;
import java.util.List;

public class MaxPoolLayer extends Layer {

    private final int _stepSize;
    private final int _windowSize;

    private final int _inputLengths;
    private final int _inputRows;
    private final int _inputColumns;

    List<int[][]> _lastMaxRow;
    List<int[][]> _lastMaxColumn;

    

    public MaxPoolLayer(int _stepSize, int _windowSize, int _inputLengths, int _inputRows, int _inputColumns) {
        this._stepSize = _stepSize;
        this._windowSize = _windowSize;
        this._inputLengths = _inputLengths;
        this._inputRows = _inputRows;
        this._inputColumns = _inputColumns;
    }

    public List<double[][]> maxPoolForwardPass(List<double[][]> input) {

        List<double[][]> output = new ArrayList<>();
        _lastMaxRow = new ArrayList<>();
        _lastMaxColumn = new ArrayList<>();

        for(int i = 0; i < input.size(); i++) {
            output.add(pool(input.get(i)));
        }

        return output;
    }

    public double[][] pool(double[][] input) {

        double[][] output = new double[getOutputRows()][getOutputColumns()];

        int[][] maxRows = new int[getOutputRows()][getOutputColumns()];
        int[][] maxColumns = new int[getOutputRows()][getOutputColumns()];

        for (int row = 0; row < getOutputRows(); row += _stepSize) {
            for (int column = 0; column < getOutputColumns(); column += _stepSize) {
                
                double max = 0.0;
                maxRows[row][column] = -1;
                maxColumns[row][column] = -1;

                for (int i = 0; i < _windowSize; i++) {
                    for (int j = 0; j < _windowSize; j++) {
                        if (max < input[row + i][column + j]) {
                            max = input[row + i][column + j];
                            maxRows[row][column] = row + i;
                            maxColumns[row][column] = column + j;
                        }
                    }
                }

                output[row][column] = max;
            }
        }

        _lastMaxRow.add(maxRows);
        _lastMaxColumn.add(maxColumns);

        return output;
    }

    @Override
    public void backPropagation(double[] dLd0) {
        List<double[][]> matrixList = vectorToMatrix(dLd0, getOutputLength(), getOutputRows(), getOutputColumns());
        backPropagation(matrixList);
    }

    @Override
    public void backPropagation(List<double[][]> dLd0) {

        List<double[][]> dXdL = new ArrayList<>();
        
        int l = 0;
        for (double[][] array: dLd0){
            double[][] error = new double[_inputRows][_inputColumns];

            for (int row = 0; row < getOutputRows(); row += _stepSize) {
                for (int column = 0; column < getOutputColumns(); column += _stepSize) {
                    int max_i = _lastMaxRow.get(l)[row][column];
                    int max_J = _lastMaxColumn.get(l)[row][column];

                    if (max_i != -1) {
                        error[max_i][max_J] += array[row][column];
                    }
                }
            }

            dXdL.add(error);
            l++;
        }

        if(_nextLayer != null) {
            _nextLayer.backPropagation(dXdL);
        }
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> outputPool = maxPoolForwardPass(input);
        return _nextLayer.getOutput(outputPool);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixList = vectorToMatrix(input, _inputLengths, _inputRows, _inputColumns);
        return getOutput(matrixList);
    }

    @Override
    public int getOutputLength() {
        return _inputLengths;
    }

    @Override
    public int getOutputRows() {
        return (_inputRows - _windowSize) / _stepSize + 1;
    }
    
    @Override
    public int getOutputColumns() {
        return (_inputColumns - _windowSize) / _stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return getOutputLength() * getOutputRows() * getOutputColumns();
    }

}
