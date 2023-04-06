using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Windows.Media;
using MathNet.Numerics.LinearAlgebra;

namespace MultipleLinearRegressionWithGradientDescent;

//TODO: Feature Scaling
//TODO: Proper Normalization
//TODO: Regularization
//TODO: Logistic Regression

public class RegressionModel
{
    public RegressionModel()
    {
    }

    public Vector<double> Weight { get; set; }
    public double Bias { get; set; }
    public Matrix<double> TrainingInput { get; private set; }
    public Vector<double> TrainingOutput { get; private set; }
    public double LearningRate { get; set; } = 0.000001;
    public double TrainingThreshold { get; set; } = 3E2;

    public IEnumerable<double> Fit(Matrix<double> trainingInput, Vector<double> trainingOutput)
    {
        TrainingInput = trainingInput;
        TrainingOutput = trainingOutput;
        Weight = Vector<double>.Build.Dense(trainingInput.ColumnCount);
        Bias = 0;

        double lastCost = GradientStep();
        double cost = GradientStep();
        double epsilon = 3E2;

        yield return cost;
        while (Math.Abs(cost-lastCost) > epsilon)
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
        double GradientComponent(int featureIndex)
        {
            var featureValues = TrainingInput.Column(featureIndex);
            return (predictions - TrainingOutput).DotProduct(featureValues) / predictions.Count;
        }

        var biasGradient = (predictions - TrainingOutput).Sum() / predictions.Count;

        var weightGradient = Vector<double>.Build.DenseOfEnumerable(Enumerable.Range(0, weight.Count).Select(GradientComponent));
        return (weightGradient, biasGradient);
    }

    private double ComputeCost(Vector<double> weight, double bias)
    {
        double InstanceCost(Vector<double> rowData, int rowIndex) 
            => Math.Pow(ComputePrediction(rowData, weight, bias) - TrainingOutput[rowIndex], 2);

        return TrainingInput.EnumerateRows().Select(InstanceCost).Sum();
    }

    public double ComputePrediction(Vector<double> input, Vector<double>? weight = null, double? bias = null)
    {
        weight ??= Weight;
        bias ??= Bias;
        return input.DotProduct(weight)+bias.Value;
    }
}