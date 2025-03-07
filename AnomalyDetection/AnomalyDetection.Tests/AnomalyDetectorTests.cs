using AnomalyDetection.Core;
using AnomalyDetection.Core.IO;
using FluentAssertions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;
using Microsoft.Extensions.DependencyInjection;

namespace AnomalyDetection.Tests;

[TestFixture(typeof(CsvMatrixLoader))]
[TestFixture(typeof(NumPyMatrixLoader))]
public class AnomalyDetectorTests<T> where T : class, IMatrixLoader
{
    private readonly AnomalyDetector _detector;

    public AnomalyDetectorTests()
    {
        ServiceCollection services = new ServiceCollection();
        var provider = services
            .AddSingleton(new DataConfiguration(Directory.GetCurrentDirectory()))
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
        actual.Should().HaveCount(TestCases.Part1DataRowsCount);
        actual.Should().AllSatisfy(x => x.Should().HaveCount(TestCases.Part1DataColumnsCount));
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
        actual.Should().HaveCount(TestCases.Part1DataRowsCount);
        actual.Should().AllSatisfy(x => x.Should().HaveCount(TestCases.Part1DataColumnsCount));
    }

    [Test]
    public void LoadFrom_ValidationTarget_Loaded()
    {
        var actual = _detector.ValidationTarget;

        actual.Take(5).Should().BeRoundedEquivalentTo([0, 0, 0, 0, 0]);
        actual.Should().HaveCount(TestCases.Part1DataRowsCount);
    }

    [TestCaseSource(typeof(TestCases), nameof(TestCases.EstimateGaussianTestCases))]
    public async Task EstimateGaussian_ReturnsCorrect_Results(TestCases.EstimateGaussianParametersTestCase parametersTestCase)
    {
        var matrix = Matrix<double>.Build.DenseOfArray(parametersTestCase.Data).Transpose();

        var expectedMean = matrix.EnumerateColumns().Select(x => x.Mean()).ToArray();
        var expectedVariance = matrix.EnumerateColumns().Select(x => x.PopulationVariance()).ToArray();

        var actual = _detector.EstimateGaussianParameters(matrix);

        Console.WriteLine($"Actual Mean: {actual.Mean}");
        Console.WriteLine($"Expected Variance: {expectedMean}");

        Console.WriteLine($"Actual Variance: {actual.Variance}");
        Console.WriteLine($"Expected Variance: {expectedVariance}");

        actual.Mean.Should().BeRoundedEquivalentTo(expectedMean, precision: parametersTestCase.Precision);
        actual.Variance.Should().BeRoundedEquivalentTo(expectedVariance, precision: parametersTestCase.Precision);
    }

    [TestCaseSource(typeof(TestCases), nameof(TestCases.MultivariateGaussianTestCases))]
    public void TestMultivariateGaussian(TestCases.MultivariateGaussianTestCase testCase)
    {
        var testData = Matrix<double>.Build.DenseOfArray(testCase.Data);

        var parameters = _detector.EstimateGaussianParameters(testData);
        parameters.Mean.Should().BeRoundedEquivalentTo(testCase.ExpectedMean, testCase.Precision);
        parameters.Variance.Should().BeRoundedEquivalentTo(testCase.ExpectedVariance, testCase.Precision);

        var probabilities = _detector.MultivariateGaussian(parameters);

        probabilities.Should().BeRoundedEquivalentTo(testCase.ExpectedProbabilities, testCase.Precision);
    }
}