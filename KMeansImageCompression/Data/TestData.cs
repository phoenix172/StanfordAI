﻿using System;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace KMeansImageCompression.Data;

public static class TestData
{
    public static BitmapImage OriginalImage =
            new(new Uri(Environment.CurrentDirectory + "/OriginalImage.jpeg"));
}