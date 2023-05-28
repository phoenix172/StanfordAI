using System;
using System.Collections.Generic;
using System.Linq;

namespace DecisionTrees;

public class SampleWithReplacement : ISampler
{
    private readonly Random _random = new(DateTime.Now.GetHashCode());

    public IEnumerable<int> Sample(int sampleSize)
    {
        var result = Enumerable.Range(0, sampleSize).Select(_ => _random.Next(0, sampleSize - 1));
        return result;
    }
}

public class SampleWithoutReplacement : ISampler
{
    public IEnumerable<int> Sample(int sampleSize)
    {
        return Enumerable.Range(0, sampleSize);
    }
}