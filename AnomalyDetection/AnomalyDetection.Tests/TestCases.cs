using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace AnomalyDetection.Tests;

public class TestCases
{
    private const double DefaultPrecision = 1d / 7d;
    public record EstimateGaussianParametersTestCase(
        double[,] Data,
        double Precision = DefaultPrecision);

    public record MultivariateGaussianTestCase(
        double[,] Data,
        double[] ExpectedMean,
        double[] ExpectedVariance,
        double[] ExpectedProbabilities,
        double Precision = DefaultPrecision);

    public static IEnumerable<MultivariateGaussianTestCase> MultivariateGaussianTestCases =>
    [
        new MultivariateGaussianTestCase(
            Data: new double[,]
            {
                { -3, -1, 0 },
                { -2, 0, 1 },
                { -1, 1, -1 },
                { 0, -1, -1 },
                { 1, 0, -2 },
                { 2, 1, 0 },
                { 3, -1, 1 }
            },
            ExpectedMean: [0d, -0.14285714, -0.28571429],
            ExpectedVariance: [4d, 0.69387755, 1.06122449],
            ExpectedProbabilities: [0.00999, 0.00887, 0.00766, 0.00645, 0.01017, 0.00817, 0.00749]
        ),
        new MultivariateGaussianTestCase(
            Data: new double[,]
            {
                { -3 },
                { -2 },
                { -1 },
                { -0 },
                { 1 },
                { 2 },
                { 3 }
            },
            ExpectedMean: [0d],
            ExpectedVariance: [4d],
            ExpectedProbabilities: [0.0647588, 0.12098536, 0.17603266, 0.19947114, 0.17603266, 0.12098536, 0.0647588]
        )
    ];

    public static IEnumerable<EstimateGaussianParametersTestCase> EstimateGaussianTestCases =>
    [
        new EstimateGaussianParametersTestCase(
            new[,]
            {
                { 1d, 1, 1 },
                { 2, 2, 2 },
                { 3, 3, 3 }
            }
        ),
        new EstimateGaussianParametersTestCase(
            new[,]
            {
                { 1d, 2, 3 },
                { 2, 4, 6 },
                { 3, 6, 9 }
            }
        ),
        new EstimateGaussianParametersTestCase(
            Matrix<double>.Build.DenseOfRowVectors
            (
                Vector<double>.Build.Random(500, new Normal(0, Math.Sqrt(1))),
                Vector<double>.Build.Random(500, new Normal(1, Math.Sqrt(2))),
                Vector<double>.Build.Random(500, new Normal(3, Math.Sqrt(1.5)))
            ).ToArray()
        ),
        new EstimateGaussianParametersTestCase(
            new[,]
            {
                { 4d, 4d, 4d },
                { 5d, 5d, 5d },
                { 6d, 6d, 6d }
            }
        ),
        new EstimateGaussianParametersTestCase(
            new[,]
            {
                { 1d, 2d, 3d },
                { 2d, 3d, 4d },
                { 3d, 4d, 5d }
            }
        ),
        new EstimateGaussianParametersTestCase(
            new[,]
            {
                { 2d, 2d, 2d },
                { 10d, 12d, 14d }
            }
        ),
        new EstimateGaussianParametersTestCase(
            new[,]
            {
                { 7d, 7d, 7d, 7d },
                { 8d, 8d, 8d, 8d },
                { 9d, 9d, 9d, 9d }
            }
        ),
        new EstimateGaussianParametersTestCase(
            new[,]
            {
                { 1d, 2d, 1d },
                { 4d, 5d, 6d },
                { 7d, 8d, 9d }
            }
        ),
        new EstimateGaussianParametersTestCase(
            new[,]
            {
                { 3d, 3d, 3d },
                { 2d, 4d, 6d },
                { 1d, 5d, 9d }
            }
        ),
    ];

    public const int Part1DataRowsCount = 307;
    public const int Part1DataColumnsCount = 2;
}