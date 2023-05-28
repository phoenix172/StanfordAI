namespace DecisionTrees.Tests.TestData;

public class AnimalsTestDecisionTree : TestDecisionTree
{
    public enum Features
    {
        EarShape,
        FaceShape,
        Whiskers
    }

    public override int[][] TrainingInput { get; } =
    new[,]
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

    public override int[] TrainingOutput { get; } = { 1, 1, 0, 0, 1, 1, 0, 1, 0, 0 };

    public override DecisionTreeNode<int[], int> BuildTree()
    {
        var tree = DecisionTreeBuilder.Create<int[], int>(TrainingInput)
            .FeatureIndex(0, Features.EarShape.ToString())
            .FeatureIndex(1, Features.FaceShape.ToString())
            .FeatureIndex(2, Features.Whiskers.ToString())
            .TargetFeature(TrainingOutput)
            .Build();
        return tree;
    }
}

public abstract class TestDecisionTree : TestDecisionTree<int[], int>
{

}

public abstract class TestDecisionTree<TNode, TTarget>
{
    public abstract TNode[] TrainingInput { get; }
    public abstract TTarget[] TrainingOutput { get; }
    public abstract DecisionTreeNode<TNode, TTarget> BuildTree();
}