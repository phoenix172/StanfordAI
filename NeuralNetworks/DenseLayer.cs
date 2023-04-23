using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Windows.Media;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;

namespace NeuralNetworks;

public class DenseLayer
{
    public int InputSize { get; }
    public int Units { get; }
    public Func<Matrix<double>, Matrix<double>> ActivationFunction { get; }
    public Matrix<double> Weight { get; set; }
    public Vector<double> Bias { get; set; }
    public Matrix<double> Input { get; private set; }
    public Matrix<double> Activations { get; private set; }

    public DenseLayer(int inputSize, int units, Func<Matrix<double>, Matrix<double>> activationFunction)
    {
        InputSize = inputSize;
        Units = units;
        ActivationFunction = activationFunction;

        Weight = Matrix<double>.Build.Random(inputSize, units).Divide(10);
        Bias = Vector<double>.Build.Random(units).Divide(10);
        Activations = Matrix<double>.Build.Random(inputSize, units).Divide(10);
        Input = Matrix<double>.Build.Random(inputSize, units).Divide(10);
    }

    public Matrix<double> ForwardPropagate(Matrix<double> input)
    {
        Input = input;
        var biasMatrix = Matrix<double>.Build.DenseOfRows(Enumerable.Repeat(Bias, input.RowCount));
        return Activations = ActivationFunction(input.Multiply(Weight).Add(biasMatrix));
    }

    private Matrix<double> ReluDerivative(Matrix<double> input)
    {
        return input.Map(x => x >= 0 ? (double)1 : double.Epsilon);
    }

    private Matrix<double> ActivationDerivative(Matrix<double> input)
    {
        if (ActivationFunction == NeuralNetworkModel.ReLU)
        {
            return ReluDerivative(input);
        }
        else if (ActivationFunction == NeuralNetworkModel.Linear)
        {
            return Matrix<double>.Build.Dense(input.RowCount, input.ColumnCount, 1);
        }

        throw new NotImplementedException();
    }

    //public Matrix<double> BackPropagate(Matrix<double> outputDerivative, double learningRate)
    //{
    //    var activationDerivative = ActivationDerivative(Input*Weight);
    //    var weightDerivative = Input.Transpose() * outputDerivative.PointwiseMultiply(activationDerivative);
    //    var biasDerivative = activationDerivative.ColumnSums();

    //    Weight -=  learningRate * weightDerivative;
    //    Bias -= learningRate * biasDerivative;

    //    return weightDerivative;
    //}

    public Matrix<double> BackPropagate(Matrix<double> outputDerivative, double learningRate)
    {
        // Calculate the activation derivative
        var activationDerivative = outputDerivative.PointwiseMultiply(ActivationDerivative(Activations));

        // Calculate the weight derivative
        var weightDerivative = Input.Transpose() * activationDerivative;

        // Calculate the bias derivative
        var biasDerivative = activationDerivative.ColumnSums();

        // Update the weights and biases
        Weight -= learningRate * weightDerivative;
        Bias -= learningRate * biasDerivative;

        // Calculate the input derivative for the previous layer
        var inputDerivative = activationDerivative * Weight.Transpose();

        return inputDerivative;
    }

    //derivative of activation function
    //
}