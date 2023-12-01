using FluentAssertions;
using FluentAssertions.Collections;

namespace AnomalyDetection.Tests;

public static class TestExtensions
{
    public static void BeRoundedEquivalentTo<T>(this GenericCollectionAssertions<T> assertion, T[] expected)
    {
        assertion.BeEquivalentTo(
            expected,
            options => options.Using<double>(ctx => ctx.Subject.Should().BeApproximately(ctx.Expectation, 0.01)).WhenTypeIs<double>()
        );
    }
}