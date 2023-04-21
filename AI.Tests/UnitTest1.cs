using System.Diagnostics;
using FluentAssertions;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MultipleLinearRegressionWithGradientDescent;
using NeuralNetworks;

namespace AI.Tests
{
    public class Tests
    {
        [Test]
        public void Test()
        {
            NeuralNetworkModel model = new NeuralNetworkModel(
                new DenseLayer(4, 2, NeuralNetworkModel.Linear)
            //new DenseLayer(2, 2, NeuralNetworkModel.Linear)
            );

            //1: Numbers have odd sum
            //2: Numbers have even sum
            Random rand = new Random((int)DateTime.Now.ToFileTimeUtc());
            var inputArray = Enumerable.Range(1, 7).Select(x =>
            {
                var numbers =
                    Vector<double>.Build.DenseOfEnumerable(Enumerable.Range(0, 4).Select(_ => (double)rand.Next(0, 10)));
                return numbers;
            }).ToList();

            Console.WriteLine("input");
            Console.WriteLine(inputArray.FormatAsString());

            var input = Matrix<double>.Build.DenseOfRows(inputArray);

            var target = Vector<double>.Build.DenseOfEnumerable(inputArray.Select(x => (double)(x.Sum() % 2 == 0 ? 2 : 1)));

            Console.WriteLine("target");
            var oneHot = NeuralNetworkModel.OneHotMatrix(target, Matrix<double>.Build.Dense(7, 2));
            Console.WriteLine(oneHot.ToRowArrays().FormatAsString());

            double cost = model.ComputeCost(input, target);

            for (int i = 0; i < 100; i++)
            {
                model.Epoch(input, target);
            }

        }

        [Test]
        public void CrossEntropyTest2()
        {
            var input = Matrix<double>.Build.DenseOfRowArrays(new[]
            {
                new double[] { 0.7,0.1,0.2},
            });
            var prediction = Vector<double>.Build.Dense(new double[] { 1});

            var expected = 0.356675;
            
            var output = NeuralNetworkModel.CrossEntropyLoss(input, prediction);

            output[0].Round(6).Should().Be(expected);
        }

        [Test]
        public void SoftmaxTest()
        {
            double[] input = { 3, 4, 1 };
            double[] expected = { 0.25949646034242, 0.70538451269824, 0.03511902695934 };
            expected = expected.Select(x => double.Round(x, 5)).ToArray();

            var result = NeuralNetworkModel.Softmax(Matrix<double>.Build.DenseOfRows(new[] { input }));
            Assert.That(result.Row(0).ToArray().Select(x => double.Round(x, 5)), Is.EquivalentTo(expected));
        }

        [Test]
        public void SoftmaxMatrixTest()
        {
            var input = Matrix<double>.Build.DenseOfArray(new[,]
            {
                { 4.8, 1.21, 2.385 },
                { 8.9, -1.81, 0.2 },
                { 1.41, 1.051, 0.026 }
            });
            var expected = Matrix<double>.Build.DenseOfArray(new[,]
            {
                { 8.952e-1, 2.470e-2, 8.000e-2},
                { 9.998e-1, 2.231e-5, 1.665e-4},
                {5.130e-1, 3.583e-1, 1.285e-1 }
            });

            var result = NeuralNetworkModel.Softmax(input);
            

            Console.WriteLine(result);
            Console.WriteLine("\nExpected:");
            Console.WriteLine(expected);
            expected.AlmostEqual(result, 3).Should().BeTrue();


        }

        [Test]
        public void OneHotReduce()
        {
            var input = Matrix<double>.Build.DenseOfArray(new double[,]
            {
                { 1, 2, 3, 4 },
                { 4, 5, 6, 7 },
                { 7, 8, 9, 10 }
            });
            var hotVector = Vector<double>.Build.DenseOfArray(new double[] { 1, 3, 4 });
            var result = NeuralNetworkModel.OneHotReduce(input, hotVector);
            CollectionAssert.AreEqual(result.ToArray(), new double[] { 1, 6, 10 });
        }
    }
}