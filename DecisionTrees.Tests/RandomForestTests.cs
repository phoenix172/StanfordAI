using DecisionTrees.Tests.TestData;
using FluentAssertions;
using NUnit.Framework.Internal;

namespace DecisionTrees.Tests;

public class RandomForestTests
{
    private readonly AnimalsTestDecisionTree _testTree = new();

    [Test]
    public void BuildRandomForest()
    {
        double trainTestSplit = 0.7;

        var (trainingInput, validationInput) = _testTree.TrainingInput.Split(trainTestSplit);
        var (trainingOutput, validationOutput) = _testTree.TrainingOutput.Split(trainTestSplit);

        var forest = new RandomForestBuilder<int[], int>()
            .WithSampler(new SampleWithoutReplacement())
            .WithStoppingCriteria(x => x.Depth >= 3)
            .FromData(trainingInput, trainingOutput)
            .ConfigureTree(x => x.ColumnFeatures(Enum.GetNames(typeof(AnimalsTestDecisionTree.Features))))
            .Build(10);

        int correctAnswers = validationInput.Select(forest.Predict).Zip(validationOutput)
            .Count(x => x.First == x.Second);
        int totalAnswers = validationInput.Count();
        double accuracy = ((double)correctAnswers / totalAnswers);

        accuracy.Should().Be(1);

        Console.WriteLine($"Correct answers: {correctAnswers}/{totalAnswers}\n" +
                          $"Accuracy: {accuracy}");
    }
}