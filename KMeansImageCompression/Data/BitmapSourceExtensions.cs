using System.Runtime.InteropServices;
using System.Windows.Media.Imaging;
using System.Windows.Media;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Windows;

namespace KMeansImageCompression.Data;

[StructLayout(LayoutKind.Sequential)]
public struct PixelColor
{
    public byte Blue;
    public byte Green;
    public byte Red;
}

public static class BitmapSourceExtensions
{
    //public static PixelColor[,] GetPixels(this BitmapSource source, PixelFormat? targetFormat = null)
    //{
    //    targetFormat ??= PixelFormats.Rgb24;

    //    if (source.Format != targetFormat)
    //        source = new FormatConvertedBitmap(source, targetFormat.Value, null, 0);

    //    int width = source.PixelWidth;
    //    int height = source.PixelHeight;
    //    PixelColor[,] result = new PixelColor[width, height];

    //    source.CopyPixels(result, width * 4, 0);

    //    return result;
    //}

    public static PixelColor[] GetPixels(this BitmapSource source)
    {
        byte[] pixels = new byte[
            source.PixelHeight * source.PixelWidth *
            source.Format.BitsPerPixel / 8];

        source.CopyPixels(pixels, source.PixelWidth * source.Format.BitsPerPixel / 8, 0);

        var pixelsArray = pixels.Chunk(3).Select(x => new PixelColor()
        {
            Red = x[0],
            Green = x[1],
            Blue = x[2]
        }).ToArray();

        return pixelsArray;
    }

    public static BitmapSource ToBitmapImage(this Matrix<double> pixels, BitmapSource originalImage)
    {
        // Define parameters used to create the BitmapSource.
        var newBitmap = new WriteableBitmap(originalImage.PixelWidth, originalImage.PixelHeight, originalImage.DpiX, originalImage.DpiY, PixelFormats.Rgb24, null);

        var bytesCount = originalImage.PixelHeight * originalImage.PixelWidth *
            originalImage.Format.BitsPerPixel / 8;

        byte[] pixelBytes = pixels.EnumerateRows().SelectMany(x => x.Select(y=>(byte)y).ToArray()).ToArray();
        Debug.Assert(bytesCount == pixelBytes.Length);

        newBitmap.WritePixels(
            new Int32Rect(0, 0, newBitmap.PixelWidth, newBitmap.PixelHeight),
            pixelBytes,
            newBitmap.PixelWidth * newBitmap.Format.BitsPerPixel / 8,
            0);

        return newBitmap;
    }
}