using DecisionTrees.Tests.TestData;
using FluentAssertions;

namespace DecisionTrees.Tests;

public class MulticlassDecisionTreeTests
{
    private readonly CarsTestDecisionTree _testTree = new();


    [Test]
    public void Generate_UsesBestSplit_CorrectNodes()
    {
        var tree = _testTree.BuildTree();
        tree.Generate(node => node.Depth >= 3);

        // Assumption: The tree splits on "HasLuxuryBrand" for the root node.
        //             Then, it splits on "IsCompact" for the children nodes.
        //             Finally, it splits on "HasTurbo" for the grandchildren nodes.

        tree.Print(_testTree);

        _testTree.TrainingInput.IndicesOf(tree.Children[0].Children[0].Children[0].Items)
            .Should().BeEquivalentTo(new[] { 5 }); // HasLuxuryBrand: 0, IsCompact: 0, HasTurbo: 0

        _testTree.TrainingInput.IndicesOf(tree.Children[0].Children[0].Children[1].Items)
            .Should().BeEquivalentTo(new[] { 7 }); // HasLuxuryBrand: 0, IsCompact: 0, HasTurbo: 1

        _testTree.TrainingInput.IndicesOf(tree.Children[0].Children[1].Children[0].Items)
            .Should().BeEquivalentTo(new[] { 4 }); // HasLuxuryBrand: 0, IsCompact: 1, HasTurbo: 0

        _testTree.TrainingInput.IndicesOf(tree.Children[0].Children[1].Children[1].Items)
            .Should().BeEquivalentTo(new[] { 2 }); // HasLuxuryBrand: 0, IsCompact: 1, HasTurbo: 1

        _testTree.TrainingInput.IndicesOf(tree.Children[1].Children[0].Items)
            .Should().BeEquivalentTo(new[] { 0, 3, 9 }); // HasLuxuryBrand: 1, IsCompact: 0, HasTurbo: 0

        _testTree.TrainingInput.IndicesOf(tree.Children[1].Children[1].Children[0].Items)
            .Should().BeEquivalentTo(new[] { 1, 8 }); // HasLuxuryBrand: 1, IsCompact: 1, HasTurbo: 0

        _testTree.TrainingInput.IndicesOf(tree.Children[1].Children[1].Children[1].Items)
            .Should().BeEquivalentTo(new[] { 6 }); // HasLuxuryBrand: 1, IsCompact: 1, HasTurbo: 1
    }


    [Test]
    public void Predict_ReturnsCorrectResult()
    {
        var tree = _testTree.BuildTree();
        tree.Generate(x => x.Depth >= 3);

        CarsTestDecisionTree.Drivetrain predicted = tree.Predict(_testTree.TrainingInput[0]);

        predicted.Should().Be(_testTree.TrainingOutput[0]);
    }
}

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