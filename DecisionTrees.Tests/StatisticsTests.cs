using FluentAssertions;

namespace DecisionTrees.Tests
{
    public class StatisticsTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [TestCase(0.5,1)]
        [TestCase(0, 0)]
        [TestCase(1, 0)]
        public void Test_Entropy(double fraction, double expectedEntropy)
        {
            Statistics.Entropy(fraction).Should().Be(expectedEntropy);
        }

        [Test]
        public void Test_MulticlassEntropy()
        {
            var data = Enumerable.Repeat(TestEnum.Value1, 2)
                .Concat(Enumerable.Repeat(TestEnum.Value2, 2))
                .Concat(Enumerable.Repeat(TestEnum.Value3, 2))
                .Concat(Enumerable.Repeat(TestEnum.Value4, 2));

            var entropy = data.Entropy(new DecisionTreeFeature<TestEnum, TestEnum>("Feature", x => x));

            entropy.Should().Be(Math.Log2(4));
        }

        private enum TestEnum
        {
            Value1,
            Value2,
            Value3,
            Value4
        }
    }
}