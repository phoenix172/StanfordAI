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
            CompressedImage compression = new CompressedImage("TestData/1/Original.jpeg");
            var compressed = compression.CompressImage();
            compressed.SaveToFile("result.png");
            ImageSharpCompare.ImagesAreEqual("result.png", "TestData/1/Compressed.png");
        }
    }
}