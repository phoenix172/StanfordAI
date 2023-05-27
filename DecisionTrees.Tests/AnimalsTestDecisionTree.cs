namespace DecisionTrees.Tests;

public class AnimalsTestDecisionTree
{
    public enum Features
    {
        EarShape,
        FaceShape,
        Whiskers
    }

    public static readonly int[][] TrainingInput =
        new [,]
        {
            { 1, 1, 1 },
            { 0, 0, 1 },
            { 0, 1, 0 },
            { 1, 0, 1 },
            { 1, 1, 1 },
            { 1, 1, 0 },
            { 0, 0, 0 },
            { 1, 1, 0 },
            { 0, 1, 0 },
            { 0, 1, 0 }
        }.ToJagged();

    public static readonly int[] TrainingOutput = { 1, 1, 0, 0, 1, 1, 0, 1, 0, 0 };

    public static DecisionTreeNode<int[]> BuildTree()
    {
        DecisionTreeNode<int[]> tree = DecisionTreeBuilder.Create(TrainingInput)
            .FeatureIndex(0, Features.EarShape.ToString())
            .FeatureIndex(1, Features.FaceShape.ToString())
            .FeatureIndex(2, Features.Whiskers.ToString())
            .TargetFeature(TrainingOutput)
            .Build();
        return tree;
    }
}