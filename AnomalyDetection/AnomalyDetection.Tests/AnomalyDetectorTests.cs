using AnomalyDetection.Client.Business;
using AnomalyDetection.Client.ServiceContracts;
using FluentAssertions;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace AnomalyDetection.Tests
{
    [TestFixture(typeof(CsvMatrixLoader))]
    [TestFixture(typeof(NumPyMatrixLoader))]
    public class AnomalyDetectorTests<T> where T : IMatrixLoader, new()
    {
        private const int Part1DataRowsCount = 307;
        private const int Part1DataColumnsCount = 2;

        readonly AnomalyDetector _detector;

        public AnomalyDetectorTests()
        {
            _detector = new(new T());
        }

        [SetUp]
        public async Task SetUp()
        {
            await _detector.LoadFrom("TestData/Part1");
        }

        [Test]
        public void LoadFrom_TrainingData_Loaded()
        {
            var actual = _detector.TrainingData.ToRowArrays();
            double[][] expected = [[13.04681517, 14.74115241],
                [13.40852019, 13.7632696],
                [14.19591481, 15.85318113],
                [14.91470077, 16.17425987],
                [13.57669961, 14.04284944]];

            actual.Take(5).Should().BeRoundedEquivalentTo(expected);
            actual.Should().HaveCount(Part1DataRowsCount);
            actual.Should().AllSatisfy(x => x.Should().HaveCount(Part1DataColumnsCount));
        }

        [Test]
        public void LoadFrom_ValidationInput_Loaded()
        {
            var actual = _detector.ValidationInput.ToRowArrays();
            double[][] expected =
            [
                [15.79025979, 14.9210243],
                [13.63961877, 15.32995521],
                [14.86589943, 16.47386514],
                [13.58467605, 13.98930611],
                [13.46404167, 15.63533011]
            ];


            actual.Take(5).Should().BeRoundedEquivalentTo(expected);
            actual.Should().HaveCount(Part1DataRowsCount);
            actual.Should().AllSatisfy(x => x.Should().HaveCount(Part1DataColumnsCount));
        }

        [Test]
        public void LoadFrom_ValidationTarget_Loaded()
        {
            var actual = _detector.ValidationTarget;

            actual.Take(5).Should().BeRoundedEquivalentTo([0, 0, 0, 0, 0]);
            actual.Should().HaveCount(Part1DataRowsCount);
        }

        [TestCaseSource(nameof(EstimateGaussianTestCases))]
        public async Task EstimateGaussian_ReturnsCorrect_Results(EstimateGaussianTestCase testCase)
        {
            var matrix = Matrix<double>.Build.DenseOfArray(testCase.Data).Transpose();

            var actual = await _detector.EstimateGaussianParameters(matrix);

            actual.Mean.Should().BeRoundedEquivalentTo(testCase.ExpectedMean, precision: testCase.Precision);
            actual.Variance.Should().BeRoundedEquivalentTo(testCase.ExpectedVariance, precision: testCase.Precision);
                
        }

        public record EstimateGaussianTestCase(
            double[,] Data,
            double[] ExpectedMean,
            double[] ExpectedVariance,
            double Precision = 0.1);

        public static IEnumerable<EstimateGaussianTestCase> EstimateGaussianTestCases =>
            [
                new EstimateGaussianTestCase(
                    new [,] {
                        {1d, 1, 1},
                        {2, 2, 2},
                        {3, 3, 3}
                    },
                    [1, 2, 3],
                    [0, 0, 0]
                ),
                new EstimateGaussianTestCase(
                    new [,] {
                        {1d, 2, 3},
                        {2, 4, 6},
                        {3, 6, 9}
                    },
                    [2, 4, 6],
                    [2d/3, 8d/3, 18d/3]
                ),
                new EstimateGaussianTestCase(
                    Matrix<double>.Build.DenseOfRowVectors
                    (
                        Vector<double>.Build.Random(500, new Normal(0, Math.Sqrt(1))),
                        Vector<double>.Build.Random(500, new Normal(1, Math.Sqrt(2))),
                        Vector<double>.Build.Random(500, new Normal(3, Math.Sqrt(1.5)))
                    ).ToArray(),
                    [0, 1, 3],
                    [1, 2, 1.5],
                    Precision: 0.2
                )
            ];
    }
}