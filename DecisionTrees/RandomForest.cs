using System;
using System.Collections.Generic;
using System.Linq;

namespace DecisionTrees;

public class RandomForestBuilder<TNode, TTarget> : IRandomForestBuilderTreeConfiguration<TNode, TTarget>,
    IConfiguredRandomForestBuilder<TNode, TTarget>
{
    private IList<TTarget> _targets;
    private IList<TNode> _input;
    private Action<DecisionTreeBuilder<TNode, TTarget>> _configureTree;
    private ISampler _sampler;
    private Func<DecisionTreeNode<TNode, TTarget>, bool> _stoppingCriteria;

    public IRandomForestBuilderTreeConfiguration<TNode, TTarget> FromData(IEnumerable<TNode> input,
        IEnumerable<TTarget> targets)
    {
        _input = input.ToList();
        _targets = targets.ToList();
        return this;
    }

    public RandomForestBuilder<TNode, TTarget> WithSampler(ISampler sampler)
    {
        _sampler = sampler;
        return this;
    }

    public RandomForestBuilder<TNode, TTarget> WithStoppingCriteria(
        Func<DecisionTreeNode<TNode, TTarget>, bool> stoppingCriteria)
    {
        _stoppingCriteria = stoppingCriteria;
        return this;
    }

    IConfiguredRandomForestBuilder<TNode, TTarget> IRandomForestBuilderTreeConfiguration<TNode, TTarget>
        .ConfigureTree(Action<DecisionTreeBuilder<TNode, TTarget>> configure)
    {
        _configureTree = configure;
        return this;
    }

    RandomForest<TNode, TTarget> IConfiguredRandomForestBuilder<TNode, TTarget>.Build(int forestSize)
    {
        var forest = new RandomForest<TNode, TTarget>(_input, _targets, _configureTree, _sampler, _stoppingCriteria,
            forestSize);
        forest.Grow();
        return forest;
    }
}

public interface IConfiguredRandomForestBuilder<TNode, TTarget>
{
    RandomForest<TNode, TTarget> Build(int forestSize);
}

public interface IRandomForestBuilderTreeConfiguration<TNode, TTarget>
{
    IConfiguredRandomForestBuilder<TNode, TTarget> ConfigureTree(Action<DecisionTreeBuilder<TNode, TTarget>> configure);
}

public class RandomForest<TNode, TTarget>
{
    private readonly Action<DecisionTreeBuilder<TNode, TTarget>> _configureTree;
    private readonly ISampler _sampler;
    private readonly Func<DecisionTreeNode<TNode, TTarget>, bool> _stoppingCriteria;
    private readonly int _forestSize;
    private readonly List<DecisionTreeNode<TNode, TTarget>> _trees = new();

    public IList<TNode> Input { get; }
    public IList<TTarget> Targets { get; }

    public IReadOnlyCollection<DecisionTreeNode<TNode, TTarget>> Trees => _trees;

    public RandomForest(IEnumerable<TNode> input,
        IEnumerable<TTarget> targets,
        Action<DecisionTreeBuilder<TNode, TTarget>> configureTree,
        ISampler sampler,
        Func<DecisionTreeNode<TNode, TTarget>, bool> stoppingCriteria,
        int forestSize)
    {
        _configureTree = configureTree;
        _sampler = sampler;
        _stoppingCriteria = stoppingCriteria;
        _forestSize = forestSize;
        Input = input.ToList();
        Targets = targets.ToList();
    }

    public TTarget? Predict(TNode input)
    {
        if (!Trees.Any()) return default;

        var predictions = _trees.Select(x => x.Predict(input));
        var groupBy = predictions.GroupBy(x => x);
        if (groupBy.MaxBy(x => x.Count()) is { } max)
            return max.Key;
        return default;
    }

    public void Grow()
    {
        for (int i = 0; i < _forestSize; i++)
            _trees.Add(GenerateTree());
    }

    private DecisionTreeNode<TNode, TTarget> GenerateTree()
    {
        int sampleSize = Input.Count;
        var sampleMask = _sampler.Sample(sampleSize);
        var input = Input.AtIndices(sampleMask);
        var target = Targets.AtIndices(sampleMask).ToArray();

        var builder = DecisionTreeBuilder
            .Create<TNode, TTarget>(input)
            .TargetFeature(target);

        _configureTree(builder);
        var tree = builder.Build();
        tree.Generate(_stoppingCriteria);

        return tree;
    }
}

public interface ISampler
{
    IEnumerable<int> Sample(int sampleSize);
}