using DecisionTrees.Tests.TestData;

namespace DecisionTrees.Tests;

public static class TreePrinter
{
    public static void Print<T>(this DecisionTreeNode<int[], T>? node, TestDecisionTree<int[], T> tree,
        string indent = "")
    {
        if (node == null)
            return;

        var indices = tree.TrainingInput.IndicesOf(node.Items);
        Console.WriteLine($"{indent}Node: {(node.IsLeaf ? node.LeafTargetValue : node.SplitOn.SplitFeature.Name)}" +
                          $" ({string.Join(",", indices.Select(x => $"{x} - {node.TargetFeature.Get(tree.TrainingInput[x])}"))})");

        foreach (var child in node.Children)
        {
            Print(child, tree, indent + "  ");
        }
    }
}