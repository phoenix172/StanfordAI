using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Windows.Media;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworks;

public class DenseLayer
{
    public int InputSize { get; }
    public int Units { get; }
    public Func<Matrix<double>, Matrix<double>> ActivationFunction { get; }
    public Matrix<double> Weight { get; }
    public Vector<double> Bias { get; }


    public DenseLayer(int inputSize, int units, Func<Matrix<double>, Matrix<double>> activationFunction)
    {
        InputSize = inputSize;
        Units = units;
        ActivationFunction = activationFunction;

        Weight = Matrix<double>.Build.Dense(inputSize, units);
        Bias = Vector<double>.Build.Dense(units);
    }

    public Matrix<double> ForwardPropagate(Matrix<double> input)
    {
        var biasMatrix = Matrix<double>.Build.DenseOfColumns(Enumerable.Repeat(Bias, Units));
        return ActivationFunction(input.Multiply(Weight).Add(biasMatrix));
    }

}