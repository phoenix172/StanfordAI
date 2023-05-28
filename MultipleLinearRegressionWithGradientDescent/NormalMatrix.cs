using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace MultipleLinearRegressionWithGradientDescent;

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

    public Vector<double> NormalizeRow(
        Vector<double> input)
    {
        return (input - Mean).PointwiseDivide(Deviation);
    }

    private MathNet.Numerics.LinearAlgebra.Vector<double> Normalize(MathNet.Numerics.LinearAlgebra.Vector<double> input,
        out double deviation, out double mean)
    {
        double deviationValue = deviation = StandardDeviation(input);
        double meanValue = mean = input.Average();
        return (input - meanValue).Divide(deviationValue);
    }

    private double StandardDeviation(MathNet.Numerics.LinearAlgebra.Vector<double> input)
    {
        double mean = input.Average();
        double variance = input.Sum(x => Math.Pow(x - mean, 2)) / input.Count;
        return Math.Sqrt(variance);
    }
}