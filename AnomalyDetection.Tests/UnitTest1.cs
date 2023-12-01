using AnomalyDetection.Client.Business;
using FluentAssertions;

namespace AnomalyDetection.Tests
{
    public class AnomalyDetectorTests
    {
        private const int Part1DataRowsCount = 307;
        private const int Part1DataColumnsCount = 2;

        readonly AnomalyDetector _detector;

        public AnomalyDetectorTests()
        {
            _detector = new(new NumPyMatrixLoader());
            _detector.LoadFrom("Data/Part1");
        }

        [Test]
        public void LoadFrom_TrainingData_Loaded()
        {
            var actual = _detector.TrainingData.ToRowArrays();
            double[][] expected = [[13.04681517, 14.74115241],
                [13.40852019, 13.7632696],
                [14.19591481, 15.85318113],
                [14.91470077, 16.17425987],
                [13.57669961, 14.04284944]];

            actual.Take(5).Should().BeRoundedEquivalentTo(expected);
            actual.Should().HaveCount(Part1DataRowsCount);
            actual.Should().AllSatisfy(x => x.Should().HaveCount(Part1DataColumnsCount));
        }

        [Test]
        public void LoadFrom_ValidationInput_Loaded()
        {
            var actual = _detector.ValidationInput.ToRowArrays();
            double[][] expected =
            [
                [15.79025979, 14.9210243],
                [13.63961877, 15.32995521],
                [14.86589943, 16.47386514],
                [13.58467605, 13.98930611],
                [13.46404167, 15.63533011]
            ];
                

            actual.Take(5).Should().BeRoundedEquivalentTo(expected);
            actual.Should().HaveCount(Part1DataRowsCount);
            actual.Should().AllSatisfy(x => x.Should().HaveCount(Part1DataColumnsCount));
        }

        [Test]
        public void LoadFrom_ValidationTarget_Loaded()
        {
            var actual = _detector.ValidationTarget;

            actual.Take(5).Should().BeRoundedEquivalentTo([0, 0, 0, 0, 0]);
            actual.Should().HaveCount(Part1DataRowsCount);
        }
    }
}