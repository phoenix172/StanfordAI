using System.Runtime.CompilerServices;
using FluentAssertions;

using static DecisionTrees.Tests.AnimalsTestDecisionTree;

namespace DecisionTrees.Tests;

[TestFixture]
public class DecisionTreeTests
{
    [Test]
    public void Split_ByFeature_LeftAndRightAreCorrect()
    {
        var tree = BuildTree();

        var split = tree.Split(tree.Features[0]);

        var leftIndices = TrainingInput.IndicesOf(split.Left);
        var rightIndices = TrainingInput.IndicesOf(split.Right);

        leftIndices.Should().BeEquivalentTo(new[] { 0, 3, 4, 5, 7 });
        rightIndices.Should().BeEquivalentTo(new[] { 1, 2, 6, 8, 9 });
    }

    [Test]
    public void Split_ByFeature_WeightedEntropyIsCorrect()
    {
        var tree = BuildTree();

        var split = tree.Split(tree.Features[0]);

        split.WeightedEntropy().Should().Be(0.7219280948873623d);
    }

    [TestCase(Features.EarShape, 0.28)]
    [TestCase(Features.FaceShape, 0.03)]
    [TestCase(Features.Whiskers, 0.12)]
    public void Split_ByFeature_InformationGainIsCorrect(Features feature, double expectedInformationGain)
    {
        var tree = BuildTree();

        var split = tree.Split(feature.ToString());

        var roundedResult = double.Round(split.InformationGain(), 2);
        roundedResult.Should().Be(expectedInformationGain);
    }

    [Test]
    public void Split_ByFeature_InformationGainIsPrecise()
    {
        var tree = BuildTree();

        var split = tree.Split(Features.EarShape.ToString());

        split.InformationGain().Should().Be(0.2780719051126377);
    }

    [Test]
    public void GenerateChildren_ByFeature_ChildrenContainCorrectItems()
    {
        var tree = BuildTree();

        tree.GenerateChildren(tree.Features[0]);

        tree.Children[0].GenerateChildren(tree.Features[1]);
        tree.Children[1].GenerateChildren(tree.Features[2]);

        TrainingInput.IndicesOf(tree.Children[0].Children[0].Items)
            .Should().BeEquivalentTo(new[] { 0, 4, 5, 7 });

        TrainingInput.IndicesOf(tree.Children[0].Children[1].Items)
            .Should().BeEquivalentTo(new[] { 3 });

        TrainingInput.IndicesOf(tree.Children[1].Children[0].Items)
            .Should().BeEquivalentTo(new[] { 1 });

        TrainingInput.IndicesOf(tree.Children[1].Children[1].Items)
            .Should().BeEquivalentTo(new[] { 2,6,8,9 });
    }
}