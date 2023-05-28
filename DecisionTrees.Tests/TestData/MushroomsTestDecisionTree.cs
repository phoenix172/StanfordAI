namespace DecisionTrees.Tests.TestData;

public class MushroomsTestDecisionTree : TestDecisionTree
{
    public override int[][] TrainingInput { get; } = new[,]
    {
        { 1, 1, 1 }, { 1, 0, 1 }, { 1, 0, 0 }, { 1, 0, 0 }, { 1, 1, 1 }, { 0, 1, 1 }, { 0, 0, 0 }, { 1, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 }
    }.ToJagged();

    public override int[] TrainingOutput { get; } = { 1, 1, 0, 0, 1, 0, 0, 1, 1, 0 };

    public override DecisionTreeNode<int[], int> BuildTree()
    {
        return DecisionTreeBuilder.Create<int[], int>(TrainingInput)
            .TargetFeature(TrainingOutput)
            .ColumnFeatures("Brown Cap", "Tapering Stalk", "Solitary")
            .Build();
    }
}