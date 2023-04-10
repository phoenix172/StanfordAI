using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    internal class NeuralNetworkModel
    {
        public DenseLayer[] Layers { get; }

        public NeuralNetworkModel(params DenseLayer[] layers)
        {
            Layers = layers;
        }

        public double ComputeCost(Matrix<double> input, Vector<double> target)
        {
            var predictions = Predict(input);

            var predictionsExp = predictions.PointwiseExp();
            var divisors = predictionsExp.RowSums();

            var top = Vector<double>.Build.DenseOfEnumerable(target.Select((x, i) => predictions[i, (int)target[i] - 1]));
            double cost = top.PointwiseDivide(divisors).PointwiseLog().Sum() / -input.RowCount;

            Debug.WriteLine("Cost: " + cost);

            return cost;
        }

        private Matrix<double> Predict(Matrix<double> input)
        {
            var current = input;
            foreach (var layer in Layers)
            {
                current = layer.ForwardPropagate(current);
            }

            var predictions = current;
            return predictions;
        }

        public Vector<double> BackPropagate(Vector<double> target)
        {

        }
    }
}
