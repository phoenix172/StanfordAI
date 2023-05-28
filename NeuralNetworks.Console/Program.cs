using MathNet.Numerics.LinearAlgebra;
using MultipleLinearRegressionWithGradientDescent;

namespace NeuralNetworks.Console
{
    internal class Program
    {
        static Random Rand = new Random((int)DateTime.Now.ToFileTimeUtc());


        static void Main(string[] args)
        {
            NeuralNetworkModel model = new NeuralNetworkModel(
                new DenseLayer(2, 6, NeuralNetworkModel.ReLU),
                new DenseLayer(6, 6, NeuralNetworkModel.ReLU),
                new DenseLayer(6, 2, NeuralNetworkModel.Linear)
            );

            var (input, target) = GenerateEquallySplitDataSet(2000);

            System.Console.WriteLine("target");

            for (int i = 0; i < 500; i++)
            {
                var cost = 0; //model.Epoch(input, target);

                if (i % 100 == 0)
                    System.Console.WriteLine(cost);
            }

            Verify(model);
        }

        private static void Verify(NeuralNetworkModel model)
        {
            var (testInput, testTarget) = GenerateEquallySplitDataSet(1000);

            var prediction = NeuralNetworkModel.Softmax(model.Predict(testInput)).EnumerateRows()
                .Select(x => x.MaximumIndex() + (double)1).ToList();

            int errors = 0;
            for (int i = 0; i < testInput.RowCount; i++)
            {
                System.Console.Write(
                    $"Sum({testInput.Row(i).ToVectorString().ReplaceLineEndings(",").Trim(',')}) = {testInput.Row(i).Sum()} is predicted to be: ");

                if (prediction[i] == testTarget[i])
                {
                    System.Console.ForegroundColor = ConsoleColor.Green;
                }
                else
                {
                    errors++;
                    System.Console.ForegroundColor = ConsoleColor.Red;
                }

                System.Console.WriteLine($"{(prediction[i] == 2 ? "Even" : "Odd")}");

                System.Console.ResetColor();
            }

            System.Console.WriteLine($"Errors: {errors}/{testInput.RowCount}");
        }

        private static (Matrix<double> input, Vector<double> target) GenerateEquallySplitDataSet(int count,
            int maxValue = 1000)
        {
            var (input1, target1) = GenerateTestData(count / 2, SumGreaterThan200, 0, 100);
            var (input2, target2) = GenerateTestData(count / 2, SumGreaterThan200, 200, maxValue);

            var input = Matrix<double>.Build.DenseOfRows(input1.EnumerateRows().Concat(input2.EnumerateRows()));
            var target = Vector<double>.Build.DenseOfEnumerable(target1.Concat(target2));
            return (input, target);
        }

        private static double SumGreaterThan200(Vector<double> x)
        {
            return (double)(x.Sum() > 200 ? 2 : 1);
        }

        private static (Matrix<double> input, Vector<double> target) GenerateTestData(int count,
            Func<Vector<double>, double> targetSelector, int minValue = 0, int maxValue = 1000)
        {
            var inputArray = Enumerable.Range(1, count).Select(x =>
            {
                var numbers =
                    Vector<double>.Build.DenseOfEnumerable(Enumerable.Range(0, 2)
                        .Select(_ => (double)Rand.Next(minValue, maxValue)));
                return numbers;
            }).ToList();

            var input = Matrix<double>.Build.DenseOfRows(inputArray);

            var target = Vector<double>.Build.DenseOfEnumerable(inputArray.Select(targetSelector));
            return (input, target);
        }
    }
}