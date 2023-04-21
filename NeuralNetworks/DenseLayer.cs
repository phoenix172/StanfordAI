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
    public Matrix<double> Weight { get; set; }
    public Vector<double> Bias { get; set; }
    public Matrix<double> Input { get; private set; }

    public DenseLayer(int inputSize, int units, Func<Matrix<double>, Matrix<double>> activationFunction)
    {
        InputSize = inputSize;
        Units = units;
        ActivationFunction = activationFunction;

        Weight = Matrix<double>.Build.Random(inputSize, units).Divide(10);
        Bias = Vector<double>.Build.Dense(units);
    }

    public Matrix<double> ForwardPropagate(Matrix<double> input)
    {
        Input = input;
        var biasMatrix = Matrix<double>.Build.DenseOfRows(Enumerable.Repeat(Bias, input.RowCount));
        return ActivationFunction(input.Multiply(Weight).Add(biasMatrix));
    }

    public Matrix<double> BackPropagate(Matrix<double> outputDerivative, double learningRate)
    {
        var derivative = Input.Transpose().Multiply(outputDerivative);
        var weightDerivative = derivative.Divide(derivative.RowCount);
        var biasDerivative = outputDerivative.Multiply(Bias);

        Weight -=  learningRate * weightDerivative;
        //Bias -= learningRate * outputDerivative.ColumnSums();
        return derivative;
    }
}