using MathNet.Numerics.LinearAlgebra;
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
                    Vector<double>.Build.DenseOfEnumerable(Enumerable.Range(0, 4).Select(_ => (double)rand.Next()));
                return numbers;
            }).ToList();

            var input = Matrix<double>.Build.DenseOfRows(inputArray);

            var target = Vector<double>.Build.DenseOfEnumerable(inputArray.Select(x => (double)(x.Sum() % 2 == 0 ? 2 : 1)));

            double cost = model.ComputeCost(input, target);

            for (int i = 0; i < 100; i++)
            {
                model.Epoch(input, target);
            }

        }

        [Test]
        public void SoftmaxTest()
        {
            double [] input = { 3, 4, 1 };
            double[] expected = { 0.25949646034242, 0.70538451269824, 0.03511902695934 };
            expected = expected.Select(x => double.Round(x, 5)).ToArray();

            var result = NeuralNetworkModel.Softmax(Matrix<double>.Build.DenseOfRows(new[] { input }));
            Assert.That(result.Row(0).ToArray().Select(x=>double.Round(x, 5)), Is.EquivalentTo(expected));
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
            CollectionAssert.AreEqual(result.ToArray(), new double[]{1,6,10});
        }
    }
}