using System;
using System.Collections.Generic;
using System.Linq;

namespace DecisionTrees;

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

public static class DecisionTreeBuilder
{
    public static DecisionTreeBuilder<TNode, TTarget> Create<TNode, TTarget>(IEnumerable<TNode> items) => new(items);
}