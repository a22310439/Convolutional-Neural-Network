package layers;

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

    

    public ConvolutionalLayer(int _filterSize, int _stepSize, int _inputLength, int _inputRows, int _inputColumns, long SEED, int numFilters) {
        this._filterSize = _filterSize;
        this._stepSize = _stepSize;
        this._inputLength = _inputLength;
        this._inputRows = _inputRows;
        this._inputColumns = _inputColumns;
        this.SEED = SEED;
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

        for(int i = 0; i < inRows - fRows + 1; i += stepSize){

            outColumn = 0;

            for(int j = 0; j < inColumns - fColumns + 1; j += stepSize){

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

    @Override
    public void backPropagation(List<double[][]> dLd0) {
        // TODO Auto-generated method stub
        
    }

    @Override
    public void backPropagation(double[] dLd0) {
        // TODO Auto-generated method stub
        
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
