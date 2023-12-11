using AnomalyDetection.Core.Statistics;
using MathNet.Numerics.LinearAlgebra;

namespace AnomalyDetection.Core;

public interface IAnomalyDetector
{
    Task LoadFrom(string basePath, string trainingDataFileName = "train",
        string validationInputFileName = "validate_input", string validationTargetFileName = "validate_target");
    Matrix<double> TrainingData { get; set; }
    Matrix<double> ValidationInput { get; set; }
    Vector<double> ValidationTarget { get; set; }
    GaussianParameters EstimateGaussianParameters(Matrix<double>? data = null);
    Vector<double> MultivariateGaussian(GaussianParameters parameters);
}