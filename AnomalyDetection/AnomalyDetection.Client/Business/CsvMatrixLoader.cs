using MathNet.Numerics.LinearAlgebra;

namespace AnomalyDetection.Client.Business;

public class CsvMatrixLoader : IMatrixLoader
{
    public string FileExtension => ".csv";

    public Matrix<double> LoadMatrix(string dataPath)
    {
        var splitData = File.ReadAllLines(dataPath).Select(x => x.Split(',').Select(double.Parse));
        var matrix = Matrix<double>.Build.DenseOfRows(splitData);
        return matrix;
    }

    public Vector<double> LoadVector(string dataPath, int columnIndex = 0)
    {
        return LoadMatrix(dataPath).EnumerateColumns().Skip(columnIndex).First();
    }
}