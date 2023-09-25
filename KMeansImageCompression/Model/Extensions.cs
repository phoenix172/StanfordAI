using System;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using KMeansImageCompression.Data;
using MathNet.Numerics.LinearAlgebra;

namespace KMeansImageCompression.Model;

public static class Extensions
{
    public static Matrix<double> ColumnExpand(this Vector<double> input, int count)
        => Matrix<double>.Build.DenseOfColumns(Enumerable.Repeat(input, count));

    public static Matrix<double> RowExpand(this Vector<double> input, int count)
        => Matrix<double>.Build.DenseOfRows(Enumerable.Repeat(input, count));

    public static Matrix<double> ToMatrix(this PixelColor[] pixels) =>
        Matrix<double>.Build.DenseOfRows(
            pixels.Select(ToDoubleArray));

    public static double[] ToDoubleArray(PixelColor pixel) =>
        new double[] { pixel.Red, pixel.Green, pixel.Blue };

    //ft. ChatGPT 4
    public static Matrix<double> CalculateEuclideanDistances(Matrix<double> a, Matrix<double> b)
    {
        int n = a.RowCount;
        int m = b.RowCount;
        int d = a.ColumnCount;

        if (b.ColumnCount != d)
            throw new ArgumentException("b must have the same number of columns as a");

        var aSquared = a.PointwisePower(2).RowSums();
        var bSquared = b.PointwisePower(2).RowSums();

        var mUnitVector = Vector<double>.Build.Dense(m, 1.0);
        var nUnitVector = Vector<double>.Build.Dense(n, 1.0);

        var term1 = nUnitVector.OuterProduct(bSquared);
        var term2 = aSquared.OuterProduct(mUnitVector);

        var term3 = ParallelTransposeAndMultiply(a, b);

        var squaredDistanceMatrix = term1 + term2 - term3;

        return squaredDistanceMatrix;
    }

    public static Matrix<double> ParallelTransposeAndMultiply(Matrix<double> a, Matrix<double> b)
    {
        var bTransposed = b.Transpose();

        int n = a.RowCount;
        int m = b.RowCount;

        int p = Environment.ProcessorCount;
        var chunks = ChunkMatrix(a, p);

        var results = new Matrix<double>[p];
        Parallel.For(0, p, i =>
        {
            results[i] = 2 * chunks[i].Multiply(bTransposed);
        });

        var term3 = Matrix<double>.Build.Dense(n, m, (i, j) =>
        {
            int chunkIndex = Math.Min(i / (n / p), results.Length-1);
            return results[chunkIndex].At(i - chunkIndex * (n / p), j);
        });

        return term3;
    }

    public static Matrix<double>[] ChunkMatrix(this Matrix<double> matrix, int chunksCount)
    {
        var chunks = new Matrix<double>[chunksCount];
        for (int i = 0; i < chunksCount; i++)
        {
            int chunkSize = matrix.RowCount / chunksCount;
            int remainder = matrix.RowCount % chunksCount;
            int start = i * chunkSize;
            int end = (i == chunksCount - 1) ? start + chunkSize + remainder : start + chunkSize;
            chunks[i] = matrix.SubMatrix(start, end - start, 0, matrix.ColumnCount);
        }

        return chunks;
    }
}