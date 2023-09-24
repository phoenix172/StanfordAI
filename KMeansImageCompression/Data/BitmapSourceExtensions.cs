using System.Runtime.InteropServices;
using System.Windows.Media.Imaging;
using System.Windows.Media;

namespace KMeansImageCompression.Data;

[StructLayout(LayoutKind.Sequential)]
public struct PixelColor
{
    public byte Blue;
    public byte Green;
    public byte Red;
    public byte Alpha;
}

public static class BitmapSourceExtensions
{
    public static PixelColor[,] GetPixels(this BitmapSource source, PixelFormat? targetFormat = null)
    {
        targetFormat ??= PixelFormats.Rgb24;

        if (source.Format != targetFormat)
            source = new FormatConvertedBitmap(source, targetFormat.Value, null, 0);

        int width = source.PixelWidth;
        int height = source.PixelHeight;
        PixelColor[,] result = new PixelColor[width, height];

        source.CopyPixels(result, width * 4, 0);
        return result;
    }
}