using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace MultipleLinearRegressionWithGradientDescent;

//TODO: Feature Scaling
//TODO: Proper Normalization
//TODO: Regularization
//TODO: Logistic Regression

public class RegressionModel
{
    public Vector<double> Weight { get; set; }
    public double Bias { get; set; }
    public Matrix<double> TrainingInput => NormalizedInput.Normal;
    public NormalMatrix NormalizedInput { get; private set; }
    public Vector<double> TrainingOutput { get; private set; }
    public double LearningRate { get; set; } = 1;
    public double TrainingThreshold { get; set; } = 3E2;

    public IEnumerable<double> Fit(Matrix<double> trainingInput, Vector<double> trainingOutput)
    {
        NormalizedInput = new NormalMatrix(trainingInput);
        TrainingOutput = trainingOutput;

        Weight = Vector<double>.Build.Dense(trainingInput.ColumnCount);
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

    private (Vector<double> WeightGradient, double BiasGradient) ComputeGradient(Vector<double> weight, double bias)
    {
        var predictions = TrainingInput.Multiply(weight) + bias;
        var scaledPredictions = (predictions - TrainingOutput);
        
        var biasGradient = scaledPredictions.Sum() / predictions.Count;

        var weightGradient = TrainingInput.LeftMultiply(scaledPredictions).Divide(predictions.Count);

        return (weightGradient, biasGradient);
    }

    private double ComputeCost(Vector<double> weight, double bias)
    {
        var predictions = TrainingInput.Multiply(weight) + bias;
        var sqrtError = (predictions - TrainingOutput);
        return sqrtError.DotProduct(sqrtError);
    }

    public double Predict(Vector<double> input)
    {
        return ComputePrediction(NormalizedInput.NormalizeRow(input), Weight, Bias);
    }

    private double ComputePrediction(Vector<double> input, Vector<double>? weight, double? bias)
    {
        weight ??= Weight;
        bias ??= Bias;
        return input.DotProduct(weight)+bias.Value;
    }
}