using AnomalyDetection.Client.ServiceContracts;
using MathNet.Numerics.LinearAlgebra;
using static AnomalyDetection.Client.Business.CrossPlatform;

namespace AnomalyDetection.Client.Business;

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
            (basePath +"/"+ trainingDataFileName + _loader.FileExtension).TrimStart('/'),
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

    public async Task<(Vector<double> Mean, Vector<double> Variance)> EstimateGaussian(Matrix<double>? data = null)
    {
        data ??= TrainingData;
        var featureSums = data.ColumnSums();
        var mean = featureSums.Divide(data.RowCount);

        var repeatedMeanMatrix = mean.ToRowMatrix(data.RowCount);
        
        var variance = (data - repeatedMeanMatrix).PointwisePower(2).ColumnSums().Divide(data.RowCount);
        return (mean, variance);
    }

}

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