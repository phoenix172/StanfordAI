using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;
using MathNet.Numerics;
using MathNet.Numerics.Financial;
using MultipleLinearRegressionWithGradientDescent;

namespace NeuralNetworks
{
    //Softmax Cross Entropy Cost Function
    //Linear layer must be last
    public class NeuralNetworkModel
    {
        public DenseLayer[] Layers { get; }

        public NormalMatrix NormalizedInput { get; private set; }
        public Matrix<double> TrainingInput { get; private set; }
        public Vector<double> Target { get; private set; }
        public double LearningRate { get; set; } = 1e-5;


        public NeuralNetworkModel(params DenseLayer[] layers)
        {
            Layers = layers;
            if (layers.Last().ActivationFunction != Linear)
                throw new ArgumentException("Last layer must have linear activation function to apply softmax");
        }

        public static Matrix<double> ReLU(Matrix<double> input) => input.PointwiseMaximum(double.Epsilon);
        public static Matrix<double> Linear(Matrix<double> input) => input;


        //public static Matrix<double> Sigmoid(Matrix<double> input) => input;


        public double ComputeCost(Matrix<double> input, Vector<double> target, DenseLayer[]? layers = null)
        {
            layers ??= Layers;
            var predictions = Predict(input, layers);

            var softmax = Softmax(predictions);
            var costs = CrossEntropyLoss(softmax, target);
            var totalCost = costs.Sum();

            //Debug.WriteLine("Cost: " + totalCost);

            return totalCost / input.RowCount;
        }

        public static Vector<double> CrossEntropyLoss(Matrix<double> prediction, Vector<double> target)
        {
            prediction = prediction.Map(x => Math.Clamp(x, 1e-9, 1 - 1e-9), Zeros.Include);
            var oneHot = OneHotMatrix(target, prediction);
            var loss = -prediction.PointwiseLog().PointwiseMultiply(oneHot).RowSums();
            return loss;
        }

        public static Vector<double> OneHotReduce(Matrix<double> input, Vector<double> target)
        {
            var enumerable = target.Select((x, i) => input[i, (int)target[i] - 1]);
            var hotPredictions = Vector<double>.Build.DenseOfEnumerable(enumerable);
            return hotPredictions;
        }

        public Matrix<double> Predict(Matrix<double> input) => Predict(
            Matrix<double>.Build.DenseOfRows(FeatureMapMatrix(input).Select(NormalizedInput.NormalizeRow)), Layers);


        private Matrix<double> Predict(Matrix<double> input, DenseLayer[] layers)
        {
            var current = input;
            foreach (var layer in layers)
            {
                current = layer.ForwardPropagate(current);
            }

            var predictions = current;
            return predictions;
        }

        //One forward propagate and one back-propagate

        //ComputeCost per layer

        //BackPropagate per layer

        private double Epoch(Matrix<double> input, Vector<double> target, int batchSize = 64)
        {
            for (int i = 0; i < input.RowCount / batchSize; i++)
            {
                int startIndex = batchSize * i;
                int count = Math.Min(batchSize, input.RowCount - startIndex);
                var inputSubset = input.SubMatrix(startIndex, count, 0, input.ColumnCount);
                var targetSubset = target.SubVector(startIndex, count);

                BackPropagate(inputSubset, targetSubset);
            }

            return ComputeCost(input, target, Layers);
        }


        public static Matrix<double> Softmax(Matrix<double> input)
        {
            var basedInput = input - Vector<double>.Build.DenseOfEnumerable(
                input.EnumerateRows().Select(x => x.Max())).ColumnExpand(input.ColumnCount);

            var exp = basedInput.PointwiseExp();
            //exp = Matrix<double>.Build.DenseOfArray(new double[,]
            //{
            //    {1,2,3},
            //    {4,5,6}
            //});
            var divisors = Matrix<double>.Build.DenseOfColumns(Enumerable.Repeat(exp.RowSums(), exp.ColumnCount));
            var result = exp.PointwiseDivide(divisors);
            return result;
        }

        private void BackPropagate(Matrix<double> input, Vector<double> target)
        {
            var predictions = Predict(input, Layers);
            var softmax = Softmax(predictions);
            var oneHot = OneHotMatrix(target, predictions);
            Matrix<double>? derivative = softmax - oneHot;
            foreach (var layer in Layers.Reverse())
            {
                derivative = layer.BackPropagate(derivative, LearningRate);
            }
        }

        public static Matrix<double> OneHotMatrix(Vector<double> target, Matrix<double> predictions)
        {
            try
            {
                var oneHot = Matrix<double>.Build.SameAs(predictions);
                for (var index = 0; index < target.Count; index++)
                {
                    oneHot[index, (int)target[index] - 1] = 1;
                }

                return oneHot;
            }
            catch (Exception ex)
            {
                throw new ArgumentException(
                    "Failed to calculate OneHotMatrix for predictions. Labels are probably incorrectly specified. Should have equal count as the features and start from 1",
                    ex);
            }
        }

        public Func<Vector<double>, Vector<double>> FeatureMap { get; set; } = x => x;

        public IEnumerable<double> Fit(Matrix<double> trainingInput, Vector<double> trainingOutput, int epochs = 1000,
            int batchSize = 64)
        {
            NormalizedInput = MapAndNormalizeInput(trainingInput);
            TrainingInput = NormalizedInput.Normal;
            Target = trainingOutput;

            return Enumerable.Range(1, epochs).Select(x => this.Epoch(trainingInput, trainingOutput, batchSize));
        }

        private NormalMatrix MapAndNormalizeInput(Matrix<double> trainingInput)
        {
            return new NormalMatrix(
                Matrix<double>.Build.DenseOfRowVectors(FeatureMapMatrix(trainingInput)));
        }

        private IEnumerable<Vector<double>> FeatureMapMatrix(Matrix<double> input)
        {
            return input.EnumerateRows().Select(FeatureMap);
        }
    }
}