using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Windows.Media.Media3D;
using MathNet.Numerics.LinearAlgebra;

namespace MultipleLinearRegressionWithGradientDescent;

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
    public int BatchSize { get; set; } = -1;

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

        IEnumerable<double> enumerable;
        if (input.Count == 1)
        {
            enumerable = Enumerable.Range(1, degree).Select(d => Math.Pow(input[0], d));
        }
        else
        {
            enumerable = EnumerateFeatures();
        }

        return Vector<double>.Build.DenseOfEnumerable(enumerable.Reverse());
    };

    public IEnumerable<double> Fit(Matrix<double> trainingInput, Vector<double> trainingOutput)
    {
        OriginalTrainingInput = trainingInput;
        NormalizedInput = new NormalMatrix(FeatureMapMatrix(trainingInput));
        TrainingOutput = trainingOutput;

        Weight = Vector<double>.Build.Dense(TrainingInput.ColumnCount);
        Bias = 0;

        Range range = 0..trainingInput.RowCount;
        if (BatchSize > 0) 
            range = 0..BatchSize;

        double lastCost = GradientStep(range);
        double cost = GradientStep(range);

        yield return cost;
        while (Math.Abs(cost-lastCost) > TrainingThreshold)
        {
            lastCost = cost;
            cost = GradientStep(range);
            if(BatchSize > 0)
                range = range.End..Math.Min(range.End.Value+BatchSize, TrainingInput.RowCount);

            if (range.Start.Value == range.End.Value)
                range = 0..BatchSize;

            yield return cost;
        }
    }

    private Matrix<double> FeatureMapMatrix(Matrix<double> input) 
        => Matrix<double>.Build.DenseOfRows(input.EnumerateRows().Select(FeatureMap));

    private double GradientStep(Range range)
    {
        (int offset, int length) = range.GetOffsetAndLength(TrainingInput.RowCount);
        Matrix<double> input = TrainingInput.SubMatrix(offset, length, 0, TrainingInput.ColumnCount);
        Vector<double> output = TrainingOutput.SubVector(offset, length);

        var (weightGradient, biasGradient) = ComputeGradient(Weight, Bias,input, output);
        Weight -= LearningRate * weightGradient;
        Bias -= LearningRate * biasGradient;
        var cost = ComputeCost(Weight, Bias, input, output) + CostRegularizationTerm(Weight);
        return cost;
    }

    private (Vector<double> WeightGradient, double BiasGradient) ComputeGradient(Vector<double> weight, double bias, Matrix<double>? input = null, Vector<double>? output = null)
    {
        input ??= TrainingInput;
        output ??= TrainingOutput;

        var predictions = ComputePrediction(input, weight, bias);
        var errors = (predictions - output);
        
        var biasGradient = errors.Sum() / predictions.Count;

        var weightGradient = input.LeftMultiply(errors).Divide(predictions.Count) + GradientRegularizationTerm(weight);

        var weightGradient1 = input.Transpose().Multiply(errors).Divide(predictions.Count) + GradientRegularizationTerm(weight);

        Debug.Assert(weightGradient.SequenceEqual(weightGradient1));

        return (weightGradient, biasGradient);
    }

    protected virtual double ComputeCost(Vector<double> weight, double bias, Matrix<double>? input = null, Vector<double>? output = null)
    {
        input ??= TrainingInput;
        output ??= TrainingOutput;
        
        var predictions = ComputePrediction(input, weight, bias);
        var squaredError = (predictions - output).PointwisePower(2).Sum();
        return squaredError / (2*input.RowCount);
    }

    private Vector<double> GradientRegularizationTerm(Vector<double> weight, Matrix<double>? data= null)
    {
        data ??= TrainingInput;
        return (RegularizationTerm / data.RowCount) * weight;
    }

    private double CostRegularizationTerm(Vector<double> weight, Matrix<double>? data = null)
    {
        data ??= TrainingInput;
        return (RegularizationTerm / (2 * data.RowCount)) * weight.PointwisePower(2).Sum();
    }

    protected virtual Vector<double> ComputePrediction(Matrix<double> input, Vector<double> weight, double bias)
    {
        return input.Multiply(weight) + bias;
    }

    protected virtual double ComputePrediction(Vector<double> input, Vector<double> weight, double bias)
    {
        return input.DotProduct(weight)+bias;
    }
}