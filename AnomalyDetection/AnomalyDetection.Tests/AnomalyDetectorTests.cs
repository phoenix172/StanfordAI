using System.Data.Common;
using AnomalyDetection.Core;
using AnomalyDetection.Core.IO;
using FluentAssertions;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Hosting;

namespace AnomalyDetection.Tests
{
    public record MockHostEnvironment() : IHostEnvironment
    {
        public string EnvironmentName { get; set; }
        public string ApplicationName { get; set; }
        public string ContentRootPath { get; set; } = Directory.GetCurrentDirectory();
        public IFileProvider ContentRootFileProvider { get; set; }
    }

    [TestFixture(typeof(CsvMatrixLoader))]
    [TestFixture(typeof(NumPyMatrixLoader))]
    public class AnomalyDetectorTests<T> where T : class, IMatrixLoader
    {
        private const int Part1DataRowsCount = 307;
        private const int Part1DataColumnsCount = 2;

        readonly AnomalyDetector _detector;

        public AnomalyDetectorTests()
        {
            ServiceCollection services = new ServiceCollection();
            var provider = services
                .AddSingleton( new DataConfiguration(Directory.GetCurrentDirectory()))
                .RegisterAnomalyDetection()
                .AddScoped<T>()
                .BuildServiceProvider();
            var matrixLoader = provider.GetRequiredService<T>();
            _detector = new AnomalyDetector(matrixLoader);
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
            double[][] expected =
            [
                [13.04681517, 14.74115241],
                [13.40852019, 13.7632696],
                [14.19591481, 15.85318113],
                [14.91470077, 16.17425987],
                [13.57669961, 14.04284944]
            ];

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

            var expectedMean = matrix.EnumerateColumns().Select(x => x.Mean()).ToArray();
            var expectedVariance = matrix.EnumerateColumns().Select(x => x.PopulationVariance()).ToArray();

            var actual = _detector.EstimateGaussianParameters(matrix);

            Console.WriteLine($"Actual Mean: {actual.Mean}");
            Console.WriteLine($"Expected Variance: {expectedMean}");

            Console.WriteLine($"Actual Variance: {actual.Variance}");
            Console.WriteLine($"Expected Variance: {expectedVariance}");

            actual.Mean.Should().BeRoundedEquivalentTo(expectedMean, precision: testCase.Precision);
            actual.Variance.Should().BeRoundedEquivalentTo(expectedVariance, precision: testCase.Precision);
        }

        public record EstimateGaussianTestCase(
            double[,] Data,
            double Precision = 0.1);

        public static IEnumerable<EstimateGaussianTestCase> EstimateGaussianTestCases =>
        [
            new EstimateGaussianTestCase(
                new[,]
                {
                    { 1d, 1, 1 },
                    { 2, 2, 2 },
                    { 3, 3, 3 }
                }
            ),
            new EstimateGaussianTestCase(
                new[,]
                {
                    { 1d, 2, 3 },
                    { 2, 4, 6 },
                    { 3, 6, 9 }
                }
            ),
            new EstimateGaussianTestCase(
                Matrix<double>.Build.DenseOfRowVectors
                (
                    Vector<double>.Build.Random(500, new Normal(0, Math.Sqrt(1))),
                    Vector<double>.Build.Random(500, new Normal(1, Math.Sqrt(2))),
                    Vector<double>.Build.Random(500, new Normal(3, Math.Sqrt(1.5)))
                ).ToArray()
            ),
            new EstimateGaussianTestCase(
                new[,]
                {
                    { 4d, 4d, 4d },
                    { 5d, 5d, 5d },
                    { 6d, 6d, 6d }
                }
            ),
            new EstimateGaussianTestCase(
                new[,]
                {
                    { 1d, 2d, 3d },
                    { 2d, 3d, 4d },
                    { 3d, 4d, 5d }
                }
            ),
            new EstimateGaussianTestCase(
                new[,]
                {
                    { 2d, 2d, 2d },
                    { 10d, 12d, 14d }
                }
            ),
            new EstimateGaussianTestCase(
                new[,]
                {
                    { 7d, 7d, 7d, 7d },
                    { 8d, 8d, 8d, 8d },
                    { 9d, 9d, 9d, 9d }
                }
            ),
            new EstimateGaussianTestCase(
                new[,]
                {
                    { 1d, 2d, 1d },
                    { 4d, 5d, 6d },
                    { 7d, 8d, 9d }
                }
            ),
            new EstimateGaussianTestCase(
                new[,]
                {
                    { 3d, 3d, 3d },
                    { 2d, 4d, 6d },
                    { 1d, 5d, 9d }
                }
            ),
        ];

        [Test]
        public void TestMultivariateGaussian()
        {
            // Define a simple dataset
            Matrix<double> testData = Matrix<double>.Build.DenseOfArray(new double[,]
            {
                { 1, 2 },
                { 2, 3 },
                { 3, 4 }
            });

            var parameters = _detector.EstimateGaussianParameters(testData);
            Vector<double> probabilities = _detector.MultivariateGaussian(parameters);

            // Define expected probabilities (replace with the actual expected values)
            Vector<double> expectedProbabilities = Vector<double>.Build.Dense([0.05855d, 0.15915, 0.05855]);

            Console.WriteLine(
                $"Expected: {expectedProbabilities.ToVectorString()} \n Actual: {probabilities.ToVectorString()}");
            probabilities.Should().BeRoundedEquivalentTo(expectedProbabilities.ToArray(), 1e-4);
        }

        [Test]
        public void TestMultivariateGaussian1()
        {
            // Define a simple dataset
            Matrix<double> testData = Matrix<double>.Build.DenseOfArray(new double[,]
            {
                { -3 },
                { -2 },
                { -1 },
                { -0 },
                { 1 },
                { 2 },
                { 3 }
            });

            var parameters = _detector.EstimateGaussianParameters(testData);
            parameters.Mean.Should().ContainSingle(value => value == 0d);
            parameters.Variance.Should().ContainSingle(value => value == 4d);
            
            Vector<double> probabilities = _detector.MultivariateGaussian(parameters);

            probabilities.Sum().Should().Be(1);
            
            // Define expected probabilities (replace with the actual expected values)
            Vector<double> expectedProbabilities = Vector<double>.Build.Dense([1]);

            Console.WriteLine(
                $"Expected: {expectedProbabilities.ToVectorString()} \n Actual: {probabilities.ToVectorString()}");
            probabilities.Should().BeRoundedEquivalentTo(expectedProbabilities.ToArray(), 1d/7d);
        }
    }
}