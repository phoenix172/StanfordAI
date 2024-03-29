﻿using System.Runtime.InteropServices;
using System.Windows.Media.Imaging;
using System.Windows.Media;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Windows;
using System.IO;
using KMeansImageCompression.Model;

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

    public static BitmapSource ToBitmapImage(this Matrix<double> pixels, BitmapSource originalImage, BitmapPalette? palette = null)
    {
        var newBitmap = new WriteableBitmap(originalImage.PixelWidth, originalImage.PixelHeight, originalImage.DpiX, originalImage.DpiY, PixelFormats.Rgb24, palette);

        byte[] pixelBytes = pixels.EnumerateRows().SelectMany(x => x.Select(y => (byte)y).ToArray()).ToArray();

        newBitmap.WritePixels(
            new Int32Rect(0, 0, newBitmap.PixelWidth, newBitmap.PixelHeight),
            pixelBytes,
            newBitmap.PixelWidth * newBitmap.Format.BitsPerPixel / 8,
            0);

        return newBitmap;
    }

    public static void SaveToFile(this BitmapSource image, string filePath, BitmapEncoder? encoder = null)
    {
        string destinationPath = 
        Path.Combine(
            Path.GetDirectoryName(filePath),
            Path.GetFileNameWithoutExtension(filePath))+
            Constants.OutputImageFileExtension;

        using var fileStream = new FileStream(destinationPath, FileMode.Create);
        encoder ??= new PngBitmapEncoder();

        encoder.Frames.Add(BitmapFrame.Create(image));
        encoder.Save(fileStream);
    }

    public static BitmapSource To24BitFormat(this BitmapSource originalImage)
    {
        FormatConvertedBitmap converted = new FormatConvertedBitmap();
        converted.BeginInit();
        converted.Source = originalImage;
        converted.DestinationFormat = PixelFormats.Rgb24;
        converted.EndInit();
        return converted;
    }
}