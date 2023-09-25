using Codeuctivity.ImageSharpCompare;
using KMeansImageCompression.Data;
using KMeansImageCompression.Model;
using System.Windows.Media;

namespace KMeansImageCompression.Tests
{
    public class CompressedImageTests
    {
        private const string ResultsFolder = "CompressionTestResults";
        [OneTimeSetUp]
        public void SetUp()
        {
            if (Directory.Exists(ResultsFolder))
                Directory.Delete(ResultsFolder, true);
            Directory.CreateDirectory(ResultsFolder);
        }

        [TestCase("TestData/1.jpeg")]
        [TestCase("TestData/2.jpeg")]
        [TestCase("TestData/3.jpeg")]
        [TestCase("TestData/4.jpeg")]
        public async Task Compress_TestImage_CompressedImage_MatchesExpectedResult(string testImage)
        {
            CompressedImage compression = new CompressedImage(testImage, new()
            {
                IterationsCount = 24,
                TargetColorsCount = 16
            });
            var compressed = await compression.CompressImageAsync();

            var resultFileName = GetResultFileName();
            compressed.SaveToFile(resultFileName);
            ImageSharpCompare.ImagesAreEqual(resultFileName, testImage);
        }

        private static string GetResultFileName()
        {
            var resultFileName = Path.Combine(ResultsFolder, Guid.NewGuid() + ".png");
            return resultFileName;
        }
    }
}