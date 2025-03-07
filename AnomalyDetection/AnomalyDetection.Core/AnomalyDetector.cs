using AnomalyDetection.Core.IO;
using AnomalyDetection.Core.Statistics;
using MathNet.Numerics.LinearAlgebra;

namespace AnomalyDetection.Core;

public class AnomalyDetector : IAnomalyDetector
{
    private readonly IMatrixLoader _loader;

    public AnomalyDetector(IMatrixLoader loader)
    {
        _loader = loader;
    }

    public Matrix<double> TrainingData { get; set; }
    public Matrix<double> ValidationInput { get; set; }
    public Vector<double> ValidationTarget { get; set; }

    public async Task LoadFrom(string basePath, string trainingDataFileName = "train",
        string validationInputFileName = "validate_input", string validationTargetFileName = "validate_target")
    {
        await LoadData(
            (basePath + "/" + trainingDataFileName + _loader.FileExtension).TrimStart('/'),
            (basePath + "/" + validationInputFileName + _loader.FileExtension).TrimStart('/'),
            (basePath + "/" + validationTargetFileName + _loader.FileExtension).TrimStart('/')
        );
    }

    public async Task LoadData(string trainingDataPath, string validationInputPath, string validationTargetPath)
    {
        TrainingData = await _loader.LoadMatrix(trainingDataPath);
        ValidationInput = await _loader.LoadMatrix(validationInputPath);
        ValidationTarget = await _loader.LoadVector(validationTargetPath);
    }


    public GaussianParameters EstimateGaussianParameters(Matrix<double>? data = null)
    {
        data ??= TrainingData;
        var featureSums = data.ColumnSums();
        var mean = featureSums.Divide(data.RowCount);

        var repeatedMeanMatrix = mean.ToRowMatrix(data.RowCount);

        var variance = (data - repeatedMeanMatrix).PointwisePower(2).ColumnSums().Divide(data.RowCount);
        return new(mean, variance, data);
    }

    public Vector<double> MultivariateGaussian(GaussianParameters parameters)
    {
        var data = parameters.Data;
        var meanRowMatrix = parameters.Mean.ToRowMatrix(parameters.Data.RowCount);
        var varianceMatrix = Matrix<double>.Build.DiagonalOfDiagonalVector(parameters.Variance);

        var diffMatrix = data - meanRowMatrix;
        Vector<double> exponents =
            -0.5*(diffMatrix * varianceMatrix.PseudoInverse() * diffMatrix.Transpose()).Diagonal();

        var factor = Math.Pow(2 * Math.PI, 0.5 * parameters.Data.ColumnCount) * Math.Sqrt(varianceMatrix.Determinant());
        return (1 / factor) * exponents.PointwiseExp();
    }
}