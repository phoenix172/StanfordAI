using MathNet.Numerics.LinearAlgebra;

namespace AnomalyDetection.Core;

public static class Utilities
{
    public static Vector<double> VectorArrange(double start, double end, double stepSize)
    {
        double stepsCount = (end - start) / stepSize;
        var steps = Enumerable.Range(0, (int)Math.Floor(stepsCount)).Select(x => start + x * stepSize);
        return Vector<double>.Build.DenseOfEnumerable(steps);
    }
}