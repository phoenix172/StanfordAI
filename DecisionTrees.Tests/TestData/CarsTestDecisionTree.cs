using NUnit.Framework.Interfaces;

namespace DecisionTrees.Tests.TestData;

public class CarsTestDecisionTree : TestDecisionTree<int[], CarsTestDecisionTree.Drivetrain>
{
    public enum Features
    {
        IsCompact,
        HasTurbo,
        HasLuxuryBrand
    }

    public enum Drivetrain
    {
        Fwd,
        Rwd,
        Awd
    }

    public override int[][] TrainingInput { get; } =
        new[,]
        {
            { 1, 0, 0 },
            { 0, 1, 0 },
            { 0, 0, 1 },
            { 1, 1, 0 },
            { 0, 1, 1 },
            { 1, 1, 1 },
            { 0, 0, 0 },
            { 1, 0, 1 },
            { 0, 1, 0 },
            { 1, 1, 0 }
        }.ToJagged();

    public override Drivetrain[] TrainingOutput { get; } =
    {
        Drivetrain.Fwd, Drivetrain.Rwd, Drivetrain.Awd, Drivetrain.Fwd, Drivetrain.Rwd, Drivetrain.Awd, Drivetrain.Fwd,
        Drivetrain.Rwd, Drivetrain.Awd, Drivetrain.Fwd
    };

    public override DecisionTreeNode<int[], Drivetrain> BuildTree()
    {
        var tree = DecisionTreeBuilder.Create<int[], Drivetrain>(TrainingInput)
            .FeatureIndex(0, Features.IsCompact.ToString())
            .FeatureIndex(1, Features.HasTurbo.ToString())
            .FeatureIndex(2, Features.HasLuxuryBrand.ToString())
            .TargetFeature(TrainingOutput)
            .Build();
        return tree;
    }
}