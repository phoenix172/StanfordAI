using System;
using System.Collections.Generic;

namespace DecisionTrees;

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