using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using MathNet.Numerics.LinearAlgebra;
using Numpy;

namespace AnomalyDetection.Business;

public class AnomalyDetector
{
    public AnomalyDetector()
    {
        
    }

    public void LoadFrom(string basePath, string trainingDataFileName = "train.npy", string validationInputFileName = "validate_input.npy", string validationTargetFileName = "validate_target.npy")
    {
        LoadData(
            Path.GetFullPath(Path.Combine(basePath, trainingDataFileName)),
            Path.Combine(basePath, validationInputFileName),
            Path.Combine(basePath, validationTargetFileName)
        );
    }

    public void LoadData(string trainingDataPath, string validationInputPath, string validationTargetPath)
    {
        TrainingData = LoadMatrix(trainingDataPath);
        ValidationInput = LoadMatrix(validationInputPath);
        ValidationTarget = LoadVector(validationTargetPath);
    }

    private static Vector<double> LoadVector(string dataPath)
    {
        NDarray? npyData = np.load(dataPath);
        double[]? data = npyData.GetData<double>();
        if (npyData.shape.Dimensions.Length != 1)
            throw new ArgumentException(".npy file does not contain a Vector. Needs to be 1D", nameof(dataPath));
        var vector = Vector<double>.Build.Dense(data);
        return vector;
    }

    private static Matrix<double> LoadMatrix(string dataPath)
    {
        NDarray? npyData = np.load(dataPath);
        double[]? data = npyData.GetData<double>();
        if (npyData.shape.Dimensions.Length != 2)
            throw new ArgumentException(".npy file does not contain a Matrix. Needs to be 2D", nameof(dataPath));
        var matrix = Matrix<double>.Build.Dense(npyData.shape[1], npyData.shape[0], data).Transpose();
        return matrix;
    }

    private static Array LoadNumPyArray<T>(string dataPath)
    {
        NDarray? npyData = np.load(dataPath);

        double[]? flatData = npyData.GetData<double>();
        Array array = Array.CreateInstance(typeof(T), npyData.shape.Dimensions);
        var span = MemoryMarshal.CreateSpan(ref Unsafe.As<byte, double>(ref MemoryMarshal.GetArrayDataReference(array)),
            array.Length);
        flatData.AsSpan().CopyTo(span);

        return array;
    }

    public Matrix<double> TrainingData { get; set; }

    public Matrix<double> ValidationInput { get; set; }
    public Vector<double> ValidationTarget { get; set; }

}