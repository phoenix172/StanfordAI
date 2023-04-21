using MathNet.Numerics.LinearAlgebra;
using MultipleLinearRegressionWithGradientDescent;

namespace NeuralNetworks.Console
{
    internal class Program
    {
        static void Main(string[] args)
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

            System.Console.WriteLine("input");
            System.Console.WriteLine(inputArray.FormatAsString());

            var input = Matrix<double>.Build.DenseOfRows(inputArray);

            var target = Vector<double>.Build.DenseOfEnumerable(inputArray.Select(x => (double)(x.Sum() % 2 == 0 ? 2 : 1)));

            System.Console.WriteLine("target");
            var oneHot = NeuralNetworkModel.OneHotMatrix(target, Matrix<double>.Build.Dense(7, 2));
            System.Console.WriteLine(oneHot.ToRowArrays().FormatAsString());

            double cost = model.ComputeCost(input, target);

            for (int i = 0; i < 100000000; i++)
            {
                model.Epoch(input, target);
            }
        }
    }
}