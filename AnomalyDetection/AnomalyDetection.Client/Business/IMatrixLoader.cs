using MathNet.Numerics.LinearAlgebra;

namespace AnomalyDetection.Client.Business;

public interface IMatrixLoader
{
    string FileExtension { get; }
    Matrix<double> LoadMatrix(string dataPath);
    Vector<double> LoadVector(string dataPath, int columnIndex = 0);
}