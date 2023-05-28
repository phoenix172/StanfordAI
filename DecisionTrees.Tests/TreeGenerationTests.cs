using DecisionTrees.Tests.TestData;
using FluentAssertions;

namespace DecisionTrees.Tests;

public class TreeGenerationTests
{
    private readonly MushroomsTestDecisionTree _testTree = new();

    [Test]
    public void Generate_UsesBestSplit_CorrectNodes()
    {
        var tree = _testTree.BuildTree();
        tree.Generate(x=>x.Depth >= 3);

        _testTree.TrainingInput.IndicesOf(tree.Children[0].Children[0].Items)
            .Should().BeEquivalentTo(new[] { 0, 1, 4, 7 });

        _testTree.TrainingInput.IndicesOf(tree.Children[0].Children[1].Items)
            .Should().BeEquivalentTo(new[] { 5 });

        _testTree.TrainingInput.IndicesOf(tree.Children[1].Children[0].Items)
            .Should().BeEquivalentTo(new[] { 8 });

        _testTree.TrainingInput.IndicesOf(tree.Children[1].Children[1].Items)
            .Should().BeEquivalentTo(new[] { 2,3,6,9 });
    }

    [Test]
    public void Predict_ReturnsCorrectResult()
    {
        var tree = _testTree.BuildTree();
        tree.Generate(x => x.Depth >= 3);
        tree.Print(_testTree);


        tree.Predict(_testTree.TrainingInput[0]);
    }
}