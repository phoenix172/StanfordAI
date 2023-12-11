using MathNet.Numerics.LinearAlgebra;

namespace AnomalyDetection.Core.IO;

public interface IMatrixLoader
{
    string FileExtension { get; }
    Task<Matrix<double>> LoadMatrix(string dataPath);
    Task<Vector<double>> LoadVector(string dataPath, int columnIndex = 0);
}