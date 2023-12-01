using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using MathNet.Numerics.LinearAlgebra;
using Numpy;

namespace AnomalyDetection.Client.Business;

public class NumPyMatrixLoader : IMatrixLoader
{
    public string FileExtension => ".npy";

    public Vector<double> LoadVector(string dataPath, int columnIndex = 0)
    {
        using NDarray? npyData = np.load(dataPath);
        double[]? data = npyData.GetData<double>();
        if (npyData.shape.Dimensions.Length != 1)
            throw new ArgumentException(".npy file does not contain a Vector. Needs to be 1D", nameof(dataPath));
        var vector = Vector<double>.Build.Dense(data);
        return vector;
    }

    public Matrix<double> LoadMatrix(string dataPath)
    {
        using NDarray? npyData = np.load(dataPath);
        double[]? data = npyData.GetData<double>();
        if (npyData.shape.Dimensions.Length != 2)
            throw new ArgumentException(".npy file does not contain a Matrix. Needs to be 2D", nameof(dataPath));
        var matrix = Matrix<double>.Build.Dense(npyData.shape[1], npyData.shape[0], data).Transpose();
        return matrix;
    }

    private static Array LoadNumPyArray<T>(string dataPath)
    {
        using NDarray? npyData = np.load(dataPath);

        double[]? flatData = npyData.GetData<double>();
        Array array = Array.CreateInstance(typeof(T), npyData.shape.Dimensions);
        var span = MemoryMarshal.CreateSpan(ref Unsafe.As<byte, double>(ref MemoryMarshal.GetArrayDataReference(array)),
            array.Length);
        flatData.AsSpan().CopyTo(span);

        return array;
    }
}