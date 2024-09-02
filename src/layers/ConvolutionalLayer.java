package layers;

import static data.MatrixUtility.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ConvolutionalLayer extends Layer{

    private final long SEED;

    private List<double[][]> _filters;
    private final int _filterSize;
    private final int _stepSize;
    private final int _inputLength;
    private final int _inputRows;
    private final int _inputColumns;
    private final double _learningRate;

    private List<double[][]> _lastInput;

    public ConvolutionalLayer(int _filterSize, int _stepSize, int _inputLength, int _inputRows, int _inputColumns, long SEED, int numFilters, double learningRate) {
        this._filterSize = _filterSize;
        this._stepSize = _stepSize;
        this._inputLength = _inputLength;
        this._inputRows = _inputRows;
        this._inputColumns = _inputColumns;
        this.SEED = SEED;
        this._learningRate = learningRate;
    }
    
    public void initialize(int numFilters) {
        generateRandomFilters(numFilters);
    }

    public void generateRandomFilters(int numFilters){
        List<double[][]> filters = new ArrayList<>();
        Random random = new Random(SEED);

        for(int i = 0; i < numFilters; i++){

            double[][] filter = new double[_filterSize][_filterSize];

            for(int j = 0; j < _filterSize; j++){
                for(int k = 0; k < _filterSize; k++){
                    
                    double value = random.nextGaussian();
                    filter[j][k] = value;

                }
            }

            filters.add(filter);
        }

        _filters = filters;
    }

    public List<double[][]> convolutionalForwardPass(List<double[][]> input) {

        _lastInput = input;
        
        List<double[][]> output = new ArrayList<>();

        for(int n = 0; n < input.size(); n++){
            for(double[][] filter : _filters){
                output.add(convolve(input.get(n), filter, _stepSize));
            }
        }
        return output;
    }

    private double[][] convolve(double[][] input, double[][] filter, int stepSize){

        int outRows = (input.length - filter.length) / stepSize + 1;
        int outColumns = (input[0].length - filter[0].length) / stepSize + 1;
        
        int inRows = input.length;
        int inColumns = input[0].length;

        int fRows = filter.length;
        int fColumns = filter[0].length;

        double[][] output = new double[outRows][outColumns];

        int outRow = 0;
        int outColumn;

        for(int i = 0; i <= inRows - fRows; i += stepSize){

            outColumn = 0;

            for(int j = 0; j < inColumns - fColumns; j += stepSize){

                    double sum = 0;

                for(int x = 0; x < fRows; x++){
                    for(int y = 0; y < fColumns; y++){

                        int inputRowIndex = i + x;
                        int inputColumnIndex = j + y;

                        double value = filter[x][y] * input[inputRowIndex][inputColumnIndex];
                        sum += value;
                    }
                }

                output[outRow][outColumn] = sum;
                outColumn++;
            }

            outRow++;
        }

        return output;
    }

    public double[][] spaceArray(double[][] input) {

        if (_stepSize == 1) {
            return input;
        }

        int outRows = (input.length - 1) * _stepSize + 1;
        int outColumns = (input[0].length - 1) * _stepSize + 1;

        double[][] output = new double[outRows][outColumns];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                output[i * _stepSize][j * _stepSize] = input[i][j];
            }
        }

        return output;
    }

    @Override
    public void backPropagation(List<double[][]> dLd0) {
        
        List<double[][]> filtersDelta = new ArrayList<>();
        List<double[][]> dLd0PreviousLayer = new ArrayList<>();

        for (double[][] _filter : _filters) {
            _filter[0][0] = 0;
            filtersDelta.add(new double[_filterSize][_filterSize]);
        }

        for(int i = 0; i < _lastInput.size(); i++){

            double[][] errorForInput = new double[_inputRows][_inputColumns];

            for(int f = 0; f < _filters.size(); f++){

                double[][] filter = _filters.get(f);
                double[][] error = dLd0.get(i * _filters.size() + f);

                double[][] spacedError = spaceArray(error);
                double[][] dLdF = convolve(_lastInput.get(i), spacedError, 1);

                double[][] delta = multiply(dLdF, _learningRate * -1);
                double[][] newTotalDelta = add(filtersDelta.get(f), delta);
                filtersDelta.set(f, newTotalDelta);

                double[][] flippedError = flipArrayHorizontal(flipArrayVertical(spacedError));
                errorForInput = add(errorForInput, fullConvolve(filter, flippedError));
            }

            dLd0PreviousLayer.add(errorForInput);
        }
        
        for(int f = 0; f < _filters.size(); f++){
            double[][] modifiedFilter = add(filtersDelta.get(f), _filters.get(f));
            _filters.set(f, modifiedFilter);
        }

        if(_previousLayer != null){
            _previousLayer.backPropagation(dLd0PreviousLayer);
        }
    }

    public double[][] flipArrayHorizontal(double[][] input) {

        int rows = input.length;
        int columns = input[0].length;

        double[][] output = new double[rows][columns];

        for (int i = 0; i < rows; i++) {
            System.arraycopy(input[i], 0, output[rows - i - 1], 0, columns);
        }

        return output;
    }
    
    public double[][] flipArrayVertical(double[][] input) {

        int rows = input.length;
        int columns = input[0].length;

        double[][] output = new double[rows][columns];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                output[i][columns - j - 1] = input[i][j];
            }
        }

        return output;
    }

    private double[][] fullConvolve(double[][] input, double[][] filter){

        int outRows = (input.length + filter.length) + 1;
        int outColumns = (input[0].length + filter[0].length) + 1;
        
        int inRows = input.length;
        int inColumns = input[0].length;

        int fRows = filter.length;
        int fColumns = filter[0].length;

        double[][] output = new double[outRows][outColumns];

        int outRow = 0;
        int outColumn;

        for(int i = -fRows + 1; i < inRows; i ++){

            outColumn = 0;

            for(int j = -fColumns + 1; j < inColumns; j ++){

                    double sum = 0;

                for(int x = 0; x < fRows; x++){
                    for(int y = 0; y < fColumns; y++){

                        int inputRowIndex = i + x;
                        int inputColumnIndex = j + y;
                        
                        if(inputRowIndex >= 0 && inputRowIndex < inRows && inputColumnIndex >= 0 && inputColumnIndex < inColumns){
                            double value = filter[x][y] * input[inputRowIndex][inputColumnIndex];
                            sum += value;
                        }
                    }
                }
                output[outRow][outColumn] = sum;
                outColumn++;
            }

            outRow++;
        }

        return output;
    }


    @Override
    public void backPropagation(double[] dLd0) {
        
        List<double[][]> matrixInput = vectorToMatrix(dLd0, _inputLength, _inputRows, _inputColumns);
        backPropagation(matrixInput);
        
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        
        List<double[][]> output = convolutionalForwardPass(input);
        return _nextLayer.getOutput(output);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> inputList = vectorToMatrix(input, _inputLength, _inputRows, _inputColumns);
        return getOutput(inputList);
    }

    @Override
    public int getOutputLength() {
        return _filters.size() * _inputLength;
    }

    @Override
    public int getOutputRows() {
        return (_inputRows - _filterSize) / _stepSize + 1;
    }

    @Override
    public int getOutputColumns() {
        return (_inputColumns - _filterSize) / _stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return getOutputLength() * getOutputRows() * getOutputColumns();
    }

}
