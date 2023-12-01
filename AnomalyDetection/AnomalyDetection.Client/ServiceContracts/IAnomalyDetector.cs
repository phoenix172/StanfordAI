﻿using MathNet.Numerics.LinearAlgebra;

namespace AnomalyDetection.Client.ServiceContracts;

public interface IAnomalyDetector
{
    void LoadFrom(string basePath, string trainingDataFileName = "train", string validationInputFileName = "validate_input", string validationTargetFileName = "validate_target");
    Matrix<double> TrainingData { get; set; }
    Matrix<double> ValidationInput { get; set; }
    Vector<double> ValidationTarget { get; set; }
}