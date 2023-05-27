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
    }
}