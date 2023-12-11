using MathNet.Numerics.LinearAlgebra;

namespace AnomalyDetection.Core.Statistics;

public record GaussianParameters(Vector<double> Mean, Vector<double> Variance, Matrix<double> Data);