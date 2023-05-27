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
    private DecisionTreeFeature<T>? _targetFeature;
    private readonly List<DecisionTreeFeature<T>> _splitFeatures = new();

    public DecisionTreeBuilder(IEnumerable<T> items)
    {
        _items = items.ToArray();
    }

    public DecisionTreeBuilder<T> Feature(Func<T[], T, bool> getter, string? name = null)
    {
        name ??= _splitFeatures.Count.ToString();
        Func<T, bool> itemsGetter = x => getter(_items, x);
        _splitFeatures.Add(new DecisionTreeFeature<T>(name, itemsGetter));
        return this;
    }

    public DecisionTreeBuilder<T> TargetFeature(Func<T[], T, bool> getter, string? name = null)
    {
        name ??= "Target";
        Func<T, bool> itemsGetter = x => getter(_items, x);
        _targetFeature = new DecisionTreeFeature<T>(name, itemsGetter);
        return this;
    }

    public DecisionTreeNode<T> Build()
    {
        _targetFeature = _targetFeature ?? throw new ArgumentException(nameof(TargetFeature));

        return new DecisionTreeNode<T>(_items, _targetFeature, 0, _splitFeatures.ToArray());
    }
}

public class DecisionTreeFeature<TNode>
{
    private readonly Func<TNode, bool> _getter;

    public DecisionTreeFeature(string name, Func<TNode, bool> getter)
    {
        _getter = getter;
        Name = name;
    }

    public string Name { get; set; }

    public bool Get(TNode node)
    {
        return _getter(node);
    }

    public (IEnumerable<TNode> Left, IEnumerable<TNode> Right) Split(IEnumerable<TNode> items)
    {
        List<TNode> left = new(), right = new();

        foreach (var item in items)
        {
            if(Get(item))
                left.Add(item);
            else
                right.Add(item);
        }
        return (left, right);
    }
}

public class DecisionTreeSplit<T>
{
    public IEnumerable<T> Items { get; }
    public DecisionTreeFeature<T> SplitFeature { get; }
    public DecisionTreeFeature<T> TargetFeature { get; }
    public IEnumerable<T> Left { get; }
    public IEnumerable<T> Right { get; }

    public DecisionTreeSplit(IEnumerable<T> items, DecisionTreeFeature<T> splitFeature, DecisionTreeFeature<T> targetFeature)
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
    public DecisionTreeFeature<T> TargetFeature { get; }
    public int Depth { get; }
    public List<T> Items { get; }
    public List<DecisionTreeNode<T>> Children { get; }
    public bool IsLeaf => Items.Count > 0;
    public DecisionTreeFeature<T>[] Features { get; }

    public DecisionTreeNode(IEnumerable<T> items, DecisionTreeFeature<T> targetFeature, int depth = 0, params DecisionTreeFeature<T>[] features)
    {
        Items = items.ToList();

        if (!features.Any()) throw new ArgumentException(nameof(features));
        if (!Items.Any()) throw new ArgumentException(nameof(items));

        TargetFeature = targetFeature;
        Depth = depth;
        Features = features;
        Children = new List<DecisionTreeNode<T>>();
    }

    public void Generate(Func<DecisionTreeNode<T>, bool> stoppingCriteria)
    {
        if (stoppingCriteria(this)) return;

        GenerateChildren();
        Children.ForEach(x=>x.Generate(stoppingCriteria));
    }

    public DecisionTreeSplit<T> Split(string featureName) => Split(GetFeature(featureName));

    public void GenerateChildren(string featureName) => GenerateChildren(GetFeature(featureName));

    public DecisionTreeFeature<T> GetFeature(string featureName)
    {
        var feature = Features.FirstOrDefault(x => x.Name == featureName)
                      ?? throw new ArgumentException($"Feature not found '{featureName}'", nameof(featureName));
        return feature;
    }

    public DecisionTreeSplit<T> BestSplit()
    {
        return Features.Select(Split).MaxBy(x => x.InformationGain())!;
    }

    public DecisionTreeSplit<T> Split(DecisionTreeFeature<T> splitFeature)
    {
        return new DecisionTreeSplit<T>(Items, splitFeature, TargetFeature);
    }

    public void GenerateChildren(DecisionTreeFeature<T>? splitFeature = null)
    {
        var split = splitFeature == null ? 
            BestSplit() : Split(splitFeature);

        GenerateChildren(split);
    }

    private void GenerateChildren(DecisionTreeSplit<T> split)
    {
        Children.Clear();
        if(split.Left.Any())
            Children.Add(new DecisionTreeNode<T>(split.Left, TargetFeature, Depth+1, Features));
        if(split.Right.Any())
            Children.Add(new DecisionTreeNode<T>(split.Right, TargetFeature, Depth+1, Features));
    }
}