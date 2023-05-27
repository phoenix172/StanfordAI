using System;
using System.CodeDom;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace DecisionTrees;

public static class DecisionTreeBuilder
{
    public static DecisionTreeBuilder<T> Create<T>(IEnumerable<T> items) => new(items);
}

public static class DecisionTreeBuilderExtensions
{
    public static DecisionTreeBuilder<int[]> FeatureIndex(this DecisionTreeBuilder<int[]> builder, int featureIndex, string? name = null)
    {
        return builder.Feature((_,x) => x[featureIndex] > 0, name);
    }

    public static DecisionTreeBuilder<int[]> TargetFeature(this DecisionTreeBuilder<int[]> builder, int[] targetValues, string? name = null)
    {
        return builder.TargetFeature((items, x) => targetValues[Array.IndexOf(items, x)] > 0, name);
    }
}

public class DecisionTreeBuilder<T>
{
    private readonly T[] _items;
    private DecisionTreeFeature<T, bool>? _targetFeature;
    private readonly List<DecisionTreeFeature<T, bool>> _splitFeatures = new();

    public DecisionTreeBuilder(IEnumerable<T> items)
    {
        _items = items.ToArray();
    }

    public DecisionTreeBuilder<T> Feature(Func<T[], T, bool> getter, string? name = null)
    {
        name ??= _splitFeatures.Count.ToString();
        Func<T, bool> itemsGetter = x => getter(_items, x);
        _splitFeatures.Add(new DecisionTreeFeature<T, bool>(name, itemsGetter));
        return this;
    }

    public DecisionTreeBuilder<T> TargetFeature(Func<T[], T, bool> getter, string? name = null)
    {
        name ??= "Target";
        Func<T, bool> itemsGetter = x => getter(_items, x);
        _targetFeature = new DecisionTreeFeature<T, bool>(name, itemsGetter);
        return this;
    }

    public DecisionTreeNode<T> Build()
    {
        _targetFeature = _targetFeature ?? throw new ArgumentException(nameof(TargetFeature));

        return new DecisionTreeNode<T>(_items, _targetFeature, _splitFeatures.ToArray());
    }
}

public class DecisionTreeFeature<TNode, TValue>
{
    private readonly Func<TNode, TValue> _getter;

    public DecisionTreeFeature(string name, Func<TNode, TValue> getter)
    {
        _getter = getter;
        Name = name;
    }

    public string Name { get; set; }

    public TValue Get(TNode node)
    {
        return _getter(node);
    }

    public (IEnumerable<TNode> Left, IEnumerable<TNode> Right) Split(IEnumerable<TNode> items)
    {
        var grouped = items.GroupBy(Get).ToArray();
        return (grouped[0], grouped[1]);
    }
}

public class DecisionTreeSplit<T>
{
    public IEnumerable<T> Items { get; }
    public DecisionTreeFeature<T, bool> SplitFeature { get; }
    public DecisionTreeFeature<T, bool> TargetFeature { get; }
    public IEnumerable<T> Left { get; }
    public IEnumerable<T> Right { get; }

    public DecisionTreeSplit(IEnumerable<T> items, DecisionTreeFeature<T, bool> splitFeature, DecisionTreeFeature<T, bool> targetFeature)
    {
        Items = items;
        SplitFeature = splitFeature;
        TargetFeature = targetFeature;
        (Left, Right) = SplitFeature.Split(Items);
    }

    public double WeightedEntropy()
    {
        var weightLeft = (double)Left.Count() / Items.Count();
        var weightRight = (double)Right.Count() / Items.Count();
        var fractionLeft = (double)Left.Count(TargetFeature.Get) / Left.Count();
        var fractionRight = (double)Right.Count(TargetFeature.Get) / Right.Count();

        var weightedEntropy = weightLeft * Statistics.Entropy(fractionLeft) + weightRight * Statistics.Entropy(fractionRight);
        return weightedEntropy;
    }

    public double InformationGain()
    {
        var fractionPreSplit = (double)Items.Count(TargetFeature.Get) / Items.Count();
        var entropyPreSplit = Statistics.Entropy(fractionPreSplit);
        var entropyPostSplit = WeightedEntropy();
        return entropyPreSplit - entropyPostSplit;
    }
}

public class DecisionTreeNode<T>
{
    public DecisionTreeFeature<T, bool> TargetFeature { get; }
    public List<T> Items { get; }
    public List<DecisionTreeNode<T>> Children { get; }
    public bool IsLeaf => Items.Count > 0;
    public DecisionTreeFeature<T, bool>[] Features { get; }

    public DecisionTreeNode(IEnumerable<T> items, DecisionTreeFeature<T, bool> targetFeature, params DecisionTreeFeature<T, bool>[] features)
    {
        Items = items.ToList();
        TargetFeature = targetFeature;
        Features = features;
        Children = new List<DecisionTreeNode<T>>();
    }

    public DecisionTreeSplit<T> Split(string featureName) => Split(GetFeature(featureName));
    public void GenerateChildren(string featureName) => GenerateChildren(GetFeature(featureName));

    public DecisionTreeFeature<T, bool> GetFeature(string featureName)
    {
        var feature = Features.FirstOrDefault(x => x.Name == featureName)
                      ?? throw new ArgumentException($"Feature not found '{featureName}'", nameof(featureName));
        return feature;
    }

    public DecisionTreeSplit<T> Split(DecisionTreeFeature<T, bool> splitFeature)
    {
        return new DecisionTreeSplit<T>(Items, splitFeature, TargetFeature);
    }

    public void GenerateChildren(DecisionTreeFeature<T, bool> splitFeature)
    {
        Children.Clear();

        DecisionTreeSplit<T> split = Split(splitFeature);

        Children.Add(new DecisionTreeNode<T>(split.Left, TargetFeature, Features));
        Children.Add(new DecisionTreeNode<T>(split.Right, TargetFeature, Features));
    }
}