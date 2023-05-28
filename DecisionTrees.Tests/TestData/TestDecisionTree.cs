namespace DecisionTrees.Tests.TestData;

public abstract class TestDecisionTree<TNode, TTarget>
{
    public abstract TNode[] TrainingInput { get; }
    public abstract TTarget[] TrainingOutput { get; }
    public abstract DecisionTreeNode<TNode, TTarget> BuildTree();
}

public abstract class TestDecisionTree : TestDecisionTree<int[], int>
{
}