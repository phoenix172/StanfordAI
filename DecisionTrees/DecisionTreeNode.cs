using System;
using System.CodeDom;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Automation;

namespace DecisionTrees;

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