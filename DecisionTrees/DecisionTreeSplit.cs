using System.Collections.Generic;
using System.Linq;

namespace DecisionTrees;

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