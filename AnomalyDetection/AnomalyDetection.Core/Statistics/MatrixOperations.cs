using MathNet.Numerics.LinearAlgebra;

namespace AnomalyDetection.Core.Statistics;

public static class MatrixOperations
{
    public static Matrix<double> ToRowMatrix(this Vector<double> vector, int numberOfRows)
    {
        var matrix = Matrix<double>.Build.Dense(numberOfRows, vector.Count);
        for (int i = 0; i < numberOfRows; i++)
        {
            matrix.SetRow(i, vector);
        }
        return matrix;
    }
}