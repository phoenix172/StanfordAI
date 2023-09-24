using System.Linq;
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

    public static Vector<double> DistanceToCentroid(Matrix<double> pixels, Vector<double> centroid)
    {
        var centroidRepeatRows = centroid.RowExpand(pixels.RowCount);
        return (pixels - centroidRepeatRows).PointwiseAbs().RowSums();
    }
}