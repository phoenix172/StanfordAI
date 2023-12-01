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

    public void LoadFrom(string basePath, string trainingDataFileName = "train", string validationInputFileName = "validate_input", string validationTargetFileName = "validate_target")
    {
        LoadData(
            basePath +"/"+ trainingDataFileName + _loader.FileExtension,
            basePath + "/" + validationInputFileName + _loader.FileExtension,
            basePath + "/" + validationTargetFileName + _loader.FileExtension
        );
    }

    public void LoadData(string trainingDataPath, string validationInputPath, string validationTargetPath)
    {
        TrainingData = _loader.LoadMatrix(trainingDataPath);
        ValidationInput = _loader.LoadMatrix(validationInputPath);
        ValidationTarget = _loader.LoadVector(validationTargetPath);
    }

    public Matrix<double> TrainingData { get; set; }
    public Matrix<double> ValidationInput { get; set; }
    public Vector<double> ValidationTarget { get; set; }

}