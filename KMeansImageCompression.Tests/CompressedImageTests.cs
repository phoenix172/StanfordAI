using Codeuctivity.ImageSharpCompare;
using KMeansImageCompression.Data;
using KMeansImageCompression.Model;

namespace KMeansImageCompression.Tests
{
    public class CompressedImageTests
    {
        [Test]
        public void Compress_TestImage_CompressedImage_MatchesExpectedResult()
        {
            CompressedImage compression = new CompressedImage("TestData/1.jpeg");
            var compressed = compression.CompressImage(24);
            compressed.SaveToFile("result.png");
            ImageSharpCompare.ImagesAreEqual("result.png", "TestData/1.jpeg");
        }
    }
}