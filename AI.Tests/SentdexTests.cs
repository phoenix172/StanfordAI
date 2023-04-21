using FluentAssertions;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetworks;

namespace AI.Tests;

[TestFixture]
public class SentdexTests
{
    static Matrix<double> TestInput = Matrix<double>.Build.DenseOfArray(new[,]
    {
        { 1, 2, 3, 2.5 } ,
        {2.0,5.0,-1.0,2.0},
        {-1.5, 2.7, 3.3, -0.8}
    });

    static Matrix<double>? Layer1Weights = Matrix<double>.Build.DenseOfArray(new[,]
    {
        { 0.2, 0.8, -0.5, 1.0 },
        { 0.5, -0.91, 0.26, -0.5 },
        { -0.26, -0.27, 0.17, 0.87 }
    });

    static Vector<double> Layer1Bias = Vector<double>.Build.Dense(new[] { 2, 3, 0.5 });

    static Matrix<double>? Layer2Weights = Matrix<double>.Build.DenseOfArray(new[,]
    {
        { 0.1, -0.14, 0.5 },
        { -0.5, 0.12, -0.33},
        { -0.44, 0.73, -0.13}
    });

    static Vector<double> Layer2Bias = Vector<double>.Build.Dense(new[] {-1, 2, -0.5});

    [Test]
    public void ForwardTest()
    {
        var layer = new DenseLayer(4, 3, NeuralNetworkModel.Linear);
        
        layer.Weight = Layer1Weights.Transpose();
        layer.Bias = Layer1Bias;

        var result = layer.ForwardPropagate(TestInput.SubMatrix(0, 1, 0, 4));

        var expected = Matrix<double>.Build.DenseOfArray(new[,] { { 4.8, 1.21, 2.385 } });

        Console.WriteLine(result.ToMatrixString());

        expected.AlmostEqual(result, 5).Should().Be(true);
    }

    [Test]
    public void ForwardTest2()
    {
        var layer = new DenseLayer(4, 3, NeuralNetworkModel.Linear);
        layer.Weight = Layer1Weights.Transpose();
        layer.Bias = Layer1Bias;

        var result = layer.ForwardPropagate(TestInput);
        var expected = Matrix<double>.Build.DenseOfArray(new[,]
        {
            { 4.8, 1.21, 2.385 },
            {8.9, -1.81, 0.2},
            {1.41, 1.051, 0.026}
        });

        Console.WriteLine(result.ToMatrixString());
        expected.AlmostEqual(result, 5).Should().Be(true);
    }

    [Test]
    public void ForwardTestDualLayer()
    {
        var layer1 = new DenseLayer(4, 3, NeuralNetworkModel.Linear);
        layer1.Weight = Layer1Weights.Transpose();
        layer1.Bias = Layer1Bias;

        var layer2 = new DenseLayer(3, 3, NeuralNetworkModel.Linear);
        layer2.Weight = Layer2Weights.Transpose();
        layer2.Bias = Layer2Bias;

        var result1 = layer1.ForwardPropagate(TestInput);
        var result2 = layer2.ForwardPropagate(result1);


        var expected = Matrix<double>.Build.DenseOfArray(new[,]
        {
            { 0.5031,-1.04185,-2.03875},
            {0.2434, -2.7332, -5.7633},
            {-0.99314, 1.41254, -0.35655}
        });

        Console.WriteLine(result2.ToMatrixString());
        expected.AlmostEqual(result2, 4).Should().Be(true);
    }

}