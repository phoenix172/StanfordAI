using System;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace KMeansImageCompression.Data;

public static class TestData
{
    public static BitmapImage OriginalImage =
            new(new Uri(Environment.CurrentDirectory + "/Data/OriginalImage.jpeg"));

    public static BitmapSource Get24BitOriginalImage()
    {
        FormatConvertedBitmap converted = new FormatConvertedBitmap();
        converted.BeginInit();
        converted.Source = OriginalImage;
        converted.DestinationFormat = PixelFormats.Rgb24;
        converted.EndInit();
        return converted;
    }
}