using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Windows.Media;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Providers.LinearAlgebra;

namespace MultipleLinearRegressionWithGradientDescent;

//TODO: Feature Scaling
//TODO: Proper Normalization
//TODO: Regularization
//TODO: Logistic Regression

public class NormalMatrix
{
    public NormalMatrix(Matrix<double> original)
    {
        Original = original;
        Normalize();
    }

    public MathNet.Numerics.LinearAlgebra.Vector<double> Mean { get; private set; }
    public MathNet.Numerics.LinearAlgebra.Vector<double> Deviation { get; private set; }
    public Matrix<double> Original { get; }
    public Matrix<double> Normal { get; private set; }
    
    private void Normalize()
    {
        Deviation = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(Original.ColumnCount);
        Mean = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(Original.ColumnCount);

        Normal = Matrix<double>.Build.DenseOfColumns(Original.EnumerateColumns().Select((c, i) =>
        {
            var normal = Normalize(c, out double deviation, out double mean);
            Deviation[i] = deviation;
            Mean[i] = mean;
            return normal;
        }));
    }

    public MathNet.Numerics.LinearAlgebra.Vector<double> NormalizeRow(
        MathNet.Numerics.LinearAlgebra.Vector<double> input)
    {
        return (input - Mean).PointwiseDivide(Deviation);
    }

    private MathNet.Numerics.LinearAlgebra.Vector<double> Normalize(MathNet.Numerics.LinearAlgebra.Vector<double> input, out double deviation, out double mean)
    {
        double deviationValue = deviation = StandardDeviation(input);
        double meanValue = mean = input.Average();
        return (input - meanValue).Divide(deviationValue);
    }

    private double StandardDeviation(MathNet.Numerics.LinearAlgebra.Vector<double> input)
    {
        double mean = input.Average();
        double deviation = input.Sum(x => Math.Pow(x - mean, 2));
        return deviation / input.Count;
    }
}


public class RegressionModel
{
    public RegressionModel()
    {
    }

    public MathNet.Numerics.LinearAlgebra.Vector<double> Weight { get; set; }
    public double Bias { get; set; }
    public Matrix<double> TrainingInput => NormalizedInput.Normal;
    public NormalMatrix NormalizedInput { get; private set; }
    public MathNet.Numerics.LinearAlgebra.Vector<double> TrainingOutput { get; private set; }
    public double LearningRate { get; set; } = 0.1;
    public double TrainingThreshold { get; set; } = 3E2;

    public IEnumerable<double> Fit(Matrix<double> trainingInput, MathNet.Numerics.LinearAlgebra.Vector<double> trainingOutput)
    {
        NormalizedInput = new NormalMatrix(trainingInput);
        TrainingOutput = trainingOutput;

        Weight = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(trainingInput.ColumnCount);
        Bias = 0;

        double lastCost = GradientStep();
        double cost = GradientStep();

        yield return cost;
        while (Math.Abs(cost-lastCost) > TrainingThreshold)
        {
            lastCost = cost;
            cost = GradientStep();
            yield return cost;
        }
    }

    private double GradientStep()
    {
        var (weightGradient, biasGradient) = ComputeGradient(Weight, Bias);
        Weight -= LearningRate * weightGradient;
        Bias -= LearningRate * biasGradient;
        var cost = ComputeCost(Weight, Bias);
        return cost;
    }

    private (MathNet.Numerics.LinearAlgebra.Vector<double> WeightGradient, double BiasGradient) ComputeGradient(MathNet.Numerics.LinearAlgebra.Vector<double> weight, double bias)
    {
        var predictions = TrainingInput.Multiply(weight) + bias;
        var scaledPredictions = (predictions - TrainingOutput);
        
        var biasGradient = scaledPredictions.Sum() / predictions.Count;

        var weightGradient = TrainingInput.LeftMultiply(scaledPredictions).Divide(predictions.Count);

        return (weightGradient, biasGradient);
    }

    private double ComputeCost(MathNet.Numerics.LinearAlgebra.Vector<double> weight, double bias)
    {
        var predictions = TrainingInput.Multiply(weight) + bias;
        var sqrtError = (predictions - TrainingOutput);
        return sqrtError.DotProduct(sqrtError);
    }

    public double Predict(MathNet.Numerics.LinearAlgebra.Vector<double> input)
    {
        return ComputePrediction(NormalizedInput.NormalizeRow(input), Weight, Bias);
    }

    private double ComputePrediction(MathNet.Numerics.LinearAlgebra.Vector<double> input, MathNet.Numerics.LinearAlgebra.Vector<double>? weight, double? bias)
    {
        weight ??= Weight;
        bias ??= Bias;
        return input.DotProduct(weight)+bias.Value;
    }
}