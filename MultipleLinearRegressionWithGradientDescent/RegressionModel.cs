using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
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

    public string GetModelEquation(int decimalPlaces = 5)
    {
        string funcStr;
        if (OriginalTrainingInput.ColumnCount == 1)
        {
            int degree = TrainingInput.ColumnCount;
            string[] variables = new[] { "x" }.Concat(Enumerable.Range(2, degree - 1).Select(x => "x^" + x)).ToArray();

            funcStr = string.Join("+",
                          variables.Zip(Weight.Select(x => Math.Round(x, decimalPlaces)))
                              .Select(x => x.First + "*" + x.Second)) +
                      "+" + Bias;

            return funcStr;
        }

        throw new NotImplementedException();
    }

    public string GetNormalizationEquation(int featureIndex, int decimalPlaces = 5)
    {
        return
            $"(x_{featureIndex} - {Math.Round(NormalizedInput.Mean[featureIndex], decimalPlaces)})/{Math.Round(NormalizedInput.Deviation[featureIndex], decimalPlaces)}";
    }

    //x*14.375091062986135+x^2*22.112854084327648+x^3*36.05371065910582+x^4*50.29443085140673+x^5*49.75632603888832+x^6*12.084475937412718+186.79686666537364
    //(x_0 - 159534034100018)/241860085626095.12

    public Expression<Func<double, double>> GetModelExpression()
    {
        if (OriginalTrainingInput.ColumnCount != 1) throw new NotImplementedException();
        var param = Expression.Parameter(typeof(double), "x");
        int degree = TrainingInput.ColumnCount;

        Expression NormalParamDegree(double degree)
        {
            BinaryExpression normalParam =
                Expression.Divide(Expression.Subtract(param, Expression.Constant(NormalizedInput.Mean[0])),
                    Expression.Constant(NormalizedInput.Deviation[0]));
            var pesho = typeof(Math).GetMethod(nameof(Math.Pow), BindingFlags.Static | BindingFlags.Public);
            var expression =
                Expression.Call(null, pesho, new Expression[] { normalParam, Expression.Constant(degree) });
            return expression;
            //return Expression.Power(normalParam, Expression.Constant(degree));
        }

        var terms = Weight.Select((d, i) => Expression.Multiply(NormalParamDegree(i + 1), Expression.Constant(d)))
            .Aggregate(Expression.Add);
        var modelEquation = Expression.Add(terms, Expression.Constant(Bias));
        return Expression.Lambda<Func<double, double>>(modelEquation, param);
    }

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

        double lastCost = GradientStep();
        double cost = GradientStep();

        yield return cost;
        while (Math.Abs(cost - lastCost) / lastCost > TrainingThreshold)
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
        if (BatchSize <= 0) return GradientStep(0..TrainingInput.RowCount);

        double cost = 0;
        int batch = 0;
        for (; batch < TrainingInput.RowCount / BatchSize; batch++)
        {
            cost += GradientStep((batch * BatchSize)..Math.Min((batch * BatchSize + BatchSize - 1),
                TrainingInput.RowCount));
        }

        return cost;
    }

    private double GradientStep(Range range)
    {
        (int offset, int length) = range.GetOffsetAndLength(TrainingInput.RowCount);
        Matrix<double> input = TrainingInput.SubMatrix(offset, length, 0, TrainingInput.ColumnCount);
        Vector<double> output = TrainingOutput.SubVector(offset, length);

        var (weightGradient, biasGradient) = ComputeGradient(Weight, Bias, input, output);
        Weight -= LearningRate * weightGradient;
        Bias -= LearningRate * biasGradient;
        var cost = ComputeCost(Weight, Bias, input, output) + CostRegularizationTerm(Weight);
        return cost;
    }

    private (Vector<double> WeightGradient, double BiasGradient) ComputeGradient(Vector<double> weight, double bias,
        Matrix<double>? input = null, Vector<double>? output = null)
    {
        input ??= TrainingInput;
        output ??= TrainingOutput;

        var predictions = ComputePrediction(input, weight, bias);
        var errors = (predictions - output);

        var biasGradient = errors.Sum() / predictions.Count;

        var weightGradient = input.LeftMultiply(errors).Divide(predictions.Count) + GradientRegularizationTerm(weight);

        var weightGradient1 = input.Transpose().Multiply(errors).Divide(predictions.Count) +
                              GradientRegularizationTerm(weight);

        Debug.Assert(weightGradient.SequenceEqual(weightGradient1));

        return (weightGradient, biasGradient);
    }

    protected virtual double ComputeCost(Vector<double> weight, double bias, Matrix<double>? input = null,
        Vector<double>? output = null)
    {
        input ??= TrainingInput;
        output ??= TrainingOutput;

        var predictions = ComputePrediction(input, weight, bias);
        var squaredError = (predictions - output).PointwisePower(2).Sum();
        return squaredError / (2 * input.RowCount);
    }

    private Vector<double> GradientRegularizationTerm(Vector<double> weight, Matrix<double>? data = null)
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
        return input.DotProduct(weight) + bias;
    }
}