using System;
using System.Drawing.Text;
using MathNet.Numerics.LinearAlgebra;

namespace MultipleLinearRegressionWithGradientDescent;

public class LogisticalRegressionModel : RegressionModel
{
    protected override Vector<double> ComputePrediction(Matrix<double> input, Vector<double> weight, double bias)
        => Sigmoid(input.Multiply(weight) + bias);

    protected override double ComputePrediction(Vector<double> input, Vector<double> weight, double bias)
        => Sigmoid(input.DotProduct(weight) + bias);

    protected override double ComputeCost(Vector<double> weight, double bias, Matrix<double>? input = null,
        Vector<double>? output = null)
    {
        input ??= TrainingInput;
        output ??= TrainingOutput;

        var predictions = ComputePrediction(input, weight, bias);
        var inverseTrainingOutput = output * -1;
        var positiveComponent = inverseTrainingOutput.PointwiseMultiply(predictions.PointwiseLog());
        var negativeComponent = (1 + inverseTrainingOutput).PointwiseMultiply((1 - predictions).PointwiseLog());
        return (positiveComponent - negativeComponent).Sum() / input.RowCount;
    }

    private static Vector<double> Sigmoid(Vector<double> baseValue)
    {
        var divisor = (1 + baseValue.Multiply(-1).PointwiseExp());
        return 1 / divisor;
    }

    private double Sigmoid(double input) => 1 / (1 + Math.Exp(-input));
}