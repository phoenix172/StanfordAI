﻿using System.Runtime.CompilerServices;
using MathNet.Numerics.LinearAlgebra;

namespace AnomalyDetection.Client.Business;

public class CsvMatrixLoader(FileReader fileReader) : IMatrixLoader
{
    public CsvMatrixLoader()
        :this(new FileReader())
    {
    }

    public string FileExtension => ".csv";

    public async Task<Matrix<double>> LoadMatrix(string dataPath)
    {
        var data = await fileReader.ReadFileLines(dataPath);
        
        var splitData = data.Select(x => x.Split(',').Select(double.Parse));
        
        var matrix = Matrix<double>.Build.DenseOfRows(splitData);
        return matrix;
        return new NormalMatrix(matrix).Normal;
    }

    public async Task<Vector<double>> LoadVector(string dataPath, int columnIndex = 0)
    {
        var matrix = (await LoadMatrix(dataPath));
        var result = matrix.SubMatrix(0, matrix.RowCount, 0, 1);
        return result.Column(0);
        return new NormalMatrix(result).Normal.Column(0);
    }
}