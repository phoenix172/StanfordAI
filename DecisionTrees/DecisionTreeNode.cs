using System;
using System.CodeDom;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Windows.Automation;

namespace DecisionTrees;

public static class DecisionTreeBuilder
{
    public static DecisionTreeBuilder<TNode, TTarget> Create<TNode, TTarget>(IEnumerable<TNode> items) => new(items);
}

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

public class DecisionTreeBuilder<TNode, TTarget>
{
    public TNode[] Items { get; }
    private DecisionTreeFeature<TNode, TTarget>? _targetFeature;
    private readonly List<DecisionTreeFeature<TNode>> _splitFeatures = new();

    public DecisionTreeBuilder(IEnumerable<TNode> items)
    {
        Items = items.ToArray();
    }

    public DecisionTreeBuilder<TNode, TTarget> Feature(Func<TNode[], TNode, bool> getter, string? name = null)
    {
        name ??= _splitFeatures.Count.ToString();
        Func<TNode, bool> itemsGetter = x => getter(Items, x);
        _splitFeatures.Add(new DecisionTreeFeature<TNode>(name, itemsGetter));
        return this;
    }

    public DecisionTreeBuilder<TNode, TTarget> TargetFeature(Func<TNode[], TNode, TTarget> getter, string? name = null)
    {
        name ??= "Target";
        Func<TNode, TTarget> itemsGetter = x => getter(Items, x);
        _targetFeature = new DecisionTreeFeature<TNode, TTarget>(name, itemsGetter);
        return this;
    }

    public DecisionTreeBuilder<TNode, TTarget> TargetFeature(TTarget[] targetValues, string? name = null)
    {
        return TargetFeature((items, x) => targetValues[Array.IndexOf(items, x)], name);
    }

    public DecisionTreeNode<TNode, TTarget> Build()
    {
        _targetFeature = _targetFeature ?? throw new ArgumentException(nameof(TargetFeature));

        return new DecisionTreeNode<TNode, TTarget>(Items, _targetFeature, 0, _splitFeatures.ToArray());
    }
}

public class DecisionTreeFeature<TNode> : DecisionTreeFeature<TNode, bool>
{
    public DecisionTreeFeature(string name, Func<TNode, bool> getter) : base(name, getter)
    {
    }

    public (IEnumerable<TNode> Left, IEnumerable<TNode> Right) Split(IEnumerable<TNode> items)
    {
        List<TNode> left = new(), right = new();

        foreach (var item in items)
        {
            if (Get(item))
                left.Add(item);
            else
                right.Add(item);
        }

        return (left, right);
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
}

public class DecisionTreeSplit<TNode, TTarget>
{
    public IEnumerable<TNode> Items { get; }
    public DecisionTreeFeature<TNode> SplitFeature { get; }
    public DecisionTreeFeature<TNode, TTarget> TargetFeature { get; }
    public IEnumerable<TNode> Left { get; }
    public IEnumerable<TNode> Right { get; }

    public DecisionTreeSplit(IEnumerable<TNode> items, DecisionTreeFeature<TNode> splitFeature,
        DecisionTreeFeature<TNode, TTarget> targetFeature)
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

        var weightedEntropy = weightLeft * Left.Entropy(TargetFeature) + weightRight * Right.Entropy(TargetFeature);
        return weightedEntropy;
    }

    public double InformationGain()
    {
        var entropyPreSplit = Items.Entropy(TargetFeature);
        var entropyPostSplit = WeightedEntropy();
        return entropyPreSplit - entropyPostSplit;
    }
}

public class DecisionTreeNode<TNode> : DecisionTreeNode<TNode, bool>
{
    public DecisionTreeNode(IEnumerable<TNode> items, DecisionTreeFeature<TNode, bool> targetFeature, int depth = 0,
        params DecisionTreeFeature<TNode>[] features)
        : base(items, targetFeature, depth, features)
    {
    }
}

public class DecisionTreeNode<TNode, TTarget>
{
    public DecisionTreeFeature<TNode, TTarget> TargetFeature { get; }
    public int Depth { get; }
    public List<TNode> Items { get; }
    public List<DecisionTreeNode<TNode, TTarget>> Children { get; }
    public DecisionTreeSplit<TNode, TTarget> SplitOn { get; set; }

    public TTarget? LeafTargetValue { get; set; }
    public bool IsLeaf => Children.Count == 0;
    public DecisionTreeFeature<TNode>[] Features { get; }

    public DecisionTreeNode(IEnumerable<TNode> items, DecisionTreeFeature<TNode, TTarget> targetFeature, int depth = 0,
        params DecisionTreeFeature<TNode>[] features)
    {
        Items = items.ToList();

        if (!features.Any()) throw new ArgumentException(nameof(features));
        if (!Items.Any()) throw new ArgumentException(nameof(items));

        TargetFeature = targetFeature;
        Depth = depth;
        Features = features;
        Children = new List<DecisionTreeNode<TNode, TTarget>>();
    }

    public DecisionTreeSplit<TNode, TTarget> Split(string featureName) => Split(GetFeature(featureName));

    public void GenerateChildren(string featureName) => GenerateChildren(GetFeature(featureName));

    public DecisionTreeFeature<TNode> GetFeature(string featureName)
    {
        var feature = Features.FirstOrDefault(x => x.Name == featureName)
                      ?? throw new ArgumentException($"Feature not found '{featureName}'", nameof(featureName));
        return feature;
    }

    public DecisionTreeSplit<TNode, TTarget> BestSplit()
    {
        return Features.Select(Split).MaxBy(x => x.InformationGain())!;
    }

    public DecisionTreeSplit<TNode, TTarget> Split(DecisionTreeFeature<TNode> splitFeature)
    {
        return new DecisionTreeSplit<TNode, TTarget>(Items, splitFeature, TargetFeature);
    }

    public void Generate(Func<DecisionTreeNode<TNode, TTarget>, bool> stoppingCriteria)
    {
        if (!stoppingCriteria(this))
        {
            GenerateChildren();

            Children.ForEach(x => x.Generate(stoppingCriteria));
        }

        if (IsLeaf)
        {
            LabelLeaf();
        }
    }

    public void GenerateChildren(DecisionTreeFeature<TNode>? splitFeature = null)
    {
        var split = splitFeature == null ? BestSplit() : Split(splitFeature);

        GenerateChildren(split);
    }

    public TTarget? Predict(TNode item)
    {
        if (IsLeaf)
            return LeafTargetValue;

        if (SplitOn.SplitFeature.Get(item))
            return Children[0].Predict(item);

        return Children[1].Predict(item);
    }

    private void GenerateChildren(DecisionTreeSplit<TNode, TTarget> split)
    {
        Children.Clear();
        if (split.Left.Any() && split.Right.Any())
        {
            Children.Add(new DecisionTreeNode<TNode, TTarget>(split.Left, TargetFeature, Depth + 1, Features));
            Children.Add(new DecisionTreeNode<TNode, TTarget>(split.Right, TargetFeature, Depth + 1, Features));
        }

        if (Children.Any())
            SplitOn = split;
    }

    private void LabelLeaf()
    {
        var groupBy = Items.GroupBy(TargetFeature.Get);
        var maxBy = groupBy.MaxBy(x => x.Count());
        if (maxBy is { } max)
            LeafTargetValue = max.Key;
        else
            LeafTargetValue = default;
    }
}