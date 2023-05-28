using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace DecisionTrees;

public static class Statistics
{
    public static double WeightedEntropy()
    {
        return 0;
    }

    public static double Entropy<TNode, TValue>(this IEnumerable<TNode> items, DecisionTreeFeature<TNode, TValue> targetFeature)
    {
        var itemsList = items.ToList();

        var fractions = itemsList.GroupBy(targetFeature.Get).Select(x => (double)x.Count() / itemsList.Count);

        var entropy = fractions.Sum(x =>
        {
            if (x.IsApproximately(0) || x.IsApproximately(1))
                return 0;

            return -x * Math.Log2(x);
        });

        return entropy;
    }

    public static double Entropy(double fraction)
    {
        if (fraction.IsApproximately(0) || fraction.IsApproximately(1))
            return 0;

        return -fraction * Math.Log2(fraction) - (1 - fraction) * Math.Log2(1 - fraction);
    }

    public static bool IsApproximately(this double fraction, double value)
    {
        return Math.Abs(fraction - value)<double.Epsilon;
    }
}