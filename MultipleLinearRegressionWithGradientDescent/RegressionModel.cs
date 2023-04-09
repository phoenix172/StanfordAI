using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Media.Media3D;
using MathNet.Numerics.LinearAlgebra;

namespace MultipleLinearRegressionWithGradientDescent;

//TODO: Feature Scaling
//TODO: Regularization
//TODO: Logistic Regression

public class RegressionModel
{
    public Vector<double> Weight { get; set; }
    public double Bias { get; set; }
    public Matrix<double> TrainingInput => NormalizedInput.Normal;
    public Matrix<double> OriginalTrainingInput { get; private set; }
    public NormalMatrix NormalizedInput { get; private set; }
    public Vector<double> TrainingOutput { get; private set; }
    public double LearningRate { get; set; } = 1;
    public double TrainingThreshold { get; set; } = 3E2;
    public double RegularizationTerm { get; set; } = 0;

    public Func<Vector<double>, Vector<double>> FeatureMap { get; set; } = x => x;

    public double Predict(Vector<double> input) 
        => ComputePrediction(NormalizedInput.NormalizeRow(FeatureMap(input)), Weight, Bias);

    public static Func<Vector<double>, Vector<double>> MapFeatureDegree(int degree) => input =>
    {
        IEnumerable<double> EnumerateFeatures()
        {
            for (int i = 1; i <= degree; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    yield return Math.Pow(input[0], i - j) * Math.Pow(input[1], j);
                }
            }
        }

        return Vector<double>.Build.DenseOfEnumerable(EnumerateFeatures());
    };

    public IEnumerable<double> Fit(Matrix<double> trainingInput, Vector<double> trainingOutput)
    {
        OriginalTrainingInput = trainingInput;
        NormalizedInput = new NormalMatrix(FeatureMapMatrix(trainingInput));
        TrainingOutput = trainingOutput;

        Weight = Vector<double>.Build.Dense(TrainingInput.ColumnCount);
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

    private Matrix<double> FeatureMapMatrix(Matrix<double> input) 
        => Matrix<double>.Build.DenseOfRows(input.EnumerateRows().Select(FeatureMap));

    private double GradientStep()
    {
        var (weightGradient, biasGradient) = ComputeGradient(Weight, Bias);
        Weight -= LearningRate * weightGradient;
        Bias -= LearningRate * biasGradient;
        var cost = ComputeCost(Weight, Bias) + CostRegularizationTerm(Weight);
        return cost;
    }

    private (Vector<double> WeightGradient, double BiasGradient) ComputeGradient(Vector<double> weight, double bias)
    {
        var predictions = ComputePrediction(TrainingInput, weight, bias);
        var scaledPredictions = (predictions - TrainingOutput);
        
        var biasGradient = scaledPredictions.Sum() / predictions.Count;

        var weightGradient = TrainingInput.LeftMultiply(scaledPredictions).Divide(predictions.Count) + GradientRegularizationTerm(weight);

        return (weightGradient, biasGradient);
    }

    protected virtual double ComputeCost(Vector<double> weight, double bias)
    {
        var predictions = ComputePrediction(TrainingInput, weight, bias);
        var sqrtError = (predictions - TrainingOutput);
        return sqrtError.DotProduct(sqrtError) / TrainingInput.RowCount;
    }

    private Vector<double> GradientRegularizationTerm(Vector<double> weight)
        => (RegularizationTerm / TrainingInput.RowCount) * weight;

    private double CostRegularizationTerm(Vector<double> weight)
        => (RegularizationTerm / (2 * TrainingInput.RowCount)) * weight.PointwisePower(2).Sum();

    protected virtual Vector<double> ComputePrediction(Matrix<double> input, Vector<double> weight, double bias)
    {
        return input.Multiply(weight) + bias;
    }

    protected virtual double ComputePrediction(Vector<double> input, Vector<double> weight, double bias)
    {
        return input.DotProduct(weight)+bias;
    }
}