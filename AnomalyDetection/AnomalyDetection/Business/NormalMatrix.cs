using MathNet.Numerics.LinearAlgebra;

namespace AnomalyDetection.Business;

public class NormalMatrix
{
    public NormalMatrix(Matrix<double> original)
    {
        Original = original;
        Normalize();
    }

    public Vector<double> Mean { get; private set; }
    public Vector<double> Deviation { get; private set; }
    public Matrix<double> Original { get; }
    public Matrix<double> Normal { get; private set; }

    private void Normalize()
    {
        Deviation = Vector<double>.Build.Dense(Original.ColumnCount);
        Mean = Vector<double>.Build.Dense(Original.ColumnCount);

        Normal = Matrix<double>.Build.DenseOfColumns(Original.EnumerateColumns().Select((c, i) =>
        {
            var normal = Normalize(c, out double deviation, out double mean);
            Deviation[i] = deviation;
            Mean[i] = mean;
            return normal;
        }));
    }

    public Vector<double> NormalizeRow(Vector<double> input)
    {
        return (input - Mean).PointwiseDivide(Deviation);
    }

    private Vector<double> Normalize(Vector<double> input,
        out double deviation, out double mean)
    {
        double deviationValue = deviation = StandardDeviation(input);
        double meanValue = mean = input.Average();
        return (input - meanValue).Divide(deviationValue);
    }

    private double StandardDeviation(Vector<double> input)
    {
        double mean = input.Average();
        double variance = input.Sum(x => Math.Pow(x - mean, 2)) / input.Count;
        return Math.Sqrt(variance);
    }
}