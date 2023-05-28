using System.Linq;
using System.Numerics;

namespace DecisionTrees;

public static class DecisionTreeBuilderExtensions
{
    public static DecisionTreeBuilder<TValue[], TTarget> ColumnFeatures<TValue, TTarget>(
        this DecisionTreeBuilder<TValue[], TTarget> builder, params string[] names)
        where TValue : INumber<TValue>
    {
        var columnsCount = builder.Items.First().Length;

        for (int i = 0; i < columnsCount; i++)
        {
            builder.FeatureIndex(i, i < names.Length ? names[i] : null);
        }

        return builder;
    }

    public static DecisionTreeBuilder<TValue[], TTarget> FeatureIndex<TValue, TTarget>(
        this DecisionTreeBuilder<TValue[], TTarget> builder, int featureIndex, string? name = null)
        where TValue : INumber<TValue>
    {
        return builder.Feature((_, x) => x[featureIndex] > default(TValue), name);
    }
}