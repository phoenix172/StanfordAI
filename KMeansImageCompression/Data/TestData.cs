using System;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using KMeansImageCompression.Model;

namespace KMeansImageCompression.Data;

public static class TestData
{
    public static BitmapImage OriginalImage = LoadBitmapImage("/Data/1.png");
    
    private static BitmapImage LoadBitmapImage(string path)
        => new(new Uri(Environment.CurrentDirectory + path));
}