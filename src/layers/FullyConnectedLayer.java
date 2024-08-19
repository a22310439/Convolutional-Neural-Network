package layers;

import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer{

    private final double[][] _weights;
    private final int _inputLength;
    private final int _outputLength;
    private final long SEED;
    private final double _learningRate;
    private final double leak = 0.01;

    private double[] lastZ;
    private double[] lastInput;

    private void setRandomWeights() {
        Random random = new Random(SEED);

        for (int i = 0; i < _inputLength; i++) {
            for (int j = 0; j < _outputLength; j++) {
                _weights[i][j] = random.nextGaussian();
            }
        }
    }
    
    public FullyConnectedLayer(int _inputLength, int _outputLength, long SEED, double _learningRate) {
        this._inputLength = _inputLength;
        this._outputLength = _outputLength;
        this._learningRate = _learningRate;
        this.SEED = SEED;

        _weights = new double[_inputLength][_outputLength];
        setRandomWeights();
    }

    public double[] fullyConnectedForwardPass(double[] input) {
        double[] z = new double[_outputLength];
        double[] output = new double[_outputLength];

        lastInput = input;

        for (int i = 0; i < _outputLength; i++) {
            for (int j = 0; j < _inputLength; j++) {
                z[i] += input[j] * _weights[j][i];
            }
        }

        lastZ = z;

        for (int i = 0; i < _outputLength; i++) {
            for (int j = 0; j < _inputLength; j++) {
                output[i] = reLu(z[j]);
            }
        }

        return output;
    }

    public double reLu(double input) {
        if (input > 0) {
            return input;
        } else {
            return 0;
        }
    }

    public double derivativeReLu(double input) {
        if (input > 0) {
            return 1;
        } else {
            return leak;
        }
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vector = matrixToVector(input);
        return getOutput(vector);
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPass = fullyConnectedForwardPass(input);

        if(_nextLayer != null) {
            return _nextLayer.getOutput(forwardPass);
        } else {
            return forwardPass;
        }
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        double[] vector = matrixToVector(dLdO);
        backPropagation(vector);
    }

    @Override
    public void backPropagation(double[] dLdO) {
        double[] dLdX = new double[_inputLength];
        double dOdz;
        double dzdw;
        double dLdw;
        double dzdx;

        for (int i = 0; i < _inputLength; i++) {
            double dLdX_sum = 0;
            for (int j = 0; j < _outputLength; j++) {
                dOdz = derivativeReLu(lastZ[j]);
                dzdw = lastInput[i];
                dzdx = _weights[i][j];

                dLdw = dLdO[j] * dOdz * dzdw;
                _weights[i][j] -= _learningRate * dLdw;
                dLdX_sum += dLdO[j] * dOdz * dzdx;
            }
            dLdX[i] = dLdX_sum;
        }

        if (_previousLayer != null){
            _previousLayer.backPropagation(dLdX); 
        }       
    }

    @Override
    public int getOutputLength() {
        return 0;
    }

    @Override
    public int getOutputRows() {
        return 0;
    }

    @Override
    public int getOutputColumns() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return _outputLength;
    }

}
