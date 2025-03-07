using AnomalyDetection.Core;
using FluentAssertions;

namespace AnomalyDetection.Tests;

public class UtilitiesTests
{
    [Test]
    public void TestVectorArrange()
    {
        var powers = Utilities.VectorArrange(-20, 1, 3);
        powers.Should().BeEquivalentTo([-20d, -17, -14, -11, -8, -5, -2]);
    }
}