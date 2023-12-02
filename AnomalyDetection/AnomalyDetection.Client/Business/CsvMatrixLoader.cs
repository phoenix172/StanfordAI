using MathNet.Numerics.LinearAlgebra;

namespace AnomalyDetection.Client.Business;

public class CsvMatrixLoader : IMatrixLoader
{
    private readonly FileReader _fileReader;

    public CsvMatrixLoader(FileReader? fileReader = null)
    {
        _fileReader = fileReader ?? new FileReader();
    }
    
    public string FileExtension => ".csv";

    public async Task<Matrix<double>> LoadMatrix(string dataPath)
    {
        var data = await _fileReader.ReadFileLines(dataPath);
        
        var splitData = data.Select(x => x.Split(',').Select(double.Parse));
        var matrix = Matrix<double>.Build.DenseOfRows(splitData);
        return matrix;
    }

    public async Task<Vector<double>> LoadVector(string dataPath, int columnIndex = 0)
    {
        var matrix = (await LoadMatrix(dataPath));
        return matrix.EnumerateColumns().Skip(columnIndex).First();
    }
}