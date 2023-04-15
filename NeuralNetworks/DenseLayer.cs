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
    public Matrix<double> Weight { get; private set; }
    public Vector<double> Bias { get; private set; }
    public Matrix<double> Activations { get; private set; }

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
        var biasMatrix = Matrix<double>.Build.DenseOfRows(Enumerable.Repeat(Bias, input.RowCount));
        return Activations = ActivationFunction(input.Multiply(Weight).Add(biasMatrix));
    }

    public Matrix<double> BackPropagate(Matrix<double> outputDerivative, double learningRate)
    {
        var derivative = Activations.PointwiseMultiply(outputDerivative);
        //Weight -=  derivative.ColumnSums().Divide(derivative.RowCount).Multiply(learningRate);
        return derivative;
    }
}