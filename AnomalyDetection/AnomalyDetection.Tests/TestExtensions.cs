using FluentAssertions;
using FluentAssertions.Collections;

namespace AnomalyDetection.Tests;

public static class TestExtensions
{
    public static AndConstraint<GenericCollectionAssertions<T>> BeRoundedEquivalentTo<T>(this GenericCollectionAssertions<T> assertion, T[] expected, double precision = 0.01, string because = "")
    {
        return assertion.BeEquivalentTo(
            expected,
            options => options.Using<double>(ctx => ctx.Subject.Should().BeApproximately(ctx.Expectation, precision)).WhenTypeIs<double>(),
            because
        );
    }
}