using Codeuctivity.ImageSharpCompare;
using KMeansImageCompression.Data;
using KMeansImageCompression.Model;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace KMeansImageCompression.Tests
{
    public class CompressedImageTests
    {
        private const string ResultsFolder = "CompressionTestResults";
        private const string TestDataFolder = "TestData";
        
        [OneTimeSetUp]
        public void SetUp()
        {
            if (Directory.Exists(ResultsFolder))
                Directory.Delete(ResultsFolder, true);
            Directory.CreateDirectory(ResultsFolder);
        }

        //[TestCase("TestData/1.jpeg")]
        //[TestCase("TestData/2.jpeg")]
        //[TestCase("TestData/3.jpeg")]
        //[TestCase("TestData/4.jpeg")]
        [TestCaseSource(nameof(GetTestImages))]
        public async Task Compress_TestImage_CompressedImage_MatchesExpectedResult(string testImage)
        {
            var options = new ImageCompressionOptions()
            {
                IterationsCount = 24,
                TargetColorsCount = 16
            };

            CompressedImage compression = new CompressedImage(testImage, options);
            var compressed = await compression.CompressImageAsync();

            AssertCloseToOriginal(testImage, compressed);
        }

        public static IEnumerable<string> GetTestImages()
        {
            return Directory.GetFiles(TestDataFolder);
        }

        private static void AssertCloseToOriginal(string testImage, BitmapSource compressed)
        {
            var resultFileName = GetResultFileName();
            compressed.SaveToFile(resultFileName);
            var result = ImageSharpCompare.CalcDiff(resultFileName, testImage);
            Assert.LessOrEqual(result.MeanError, 50);
        }

        private static string GetResultFileName()
        {
            var resultFileName = Path.Combine(ResultsFolder, Guid.NewGuid() + Constants.OutputImageFileExtension);
            return resultFileName;
        }
    }
}