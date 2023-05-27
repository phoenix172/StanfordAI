using System;

namespace DecisionTrees;

public static class Statistics
{
    public static double WeightedEntropy()
    {
        return 0;
    }

    public static double Entropy(double fraction)
    {
        if (fraction.IsApproximately(0) || fraction.IsApproximately(1))
            return 0;

        return -fraction * Math.Log2(fraction) - (1 - fraction) * Math.Log2(1 - fraction);
    }

    private static bool IsApproximately(this double fraction, double value)
    {
        return Math.Abs(fraction - value)<double.Epsilon;
    }
}