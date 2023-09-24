using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using KMeansImageCompression.Data;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;
using NumSharp;

namespace KMeansImageCompression.Model;

public class ImageCompressorService
{

    public ImageCompressorService()
    {

    }

    public void CompressImage(BitmapImage originalImage)
    {
        var pixels = originalImage.GetPixels();
    }

    public Vector<int> FindClosestCentroids(PixelColor[,] pixels, PixelColor[] centroids)
    {
        Matrix<double> pixelMatrix = pixels.Cast<PixelColor>().ToArray().ToMatrix();
        Matrix<double> centroidMatrix = centroids.ToMatrix();
        //Matrix<Centroid(K) by Pixel(n=s*s)>
        Matrix<double> centroidDistanceMatrix = Matrix<double>.Build.DenseOfRowVectors
        (
        centroidMatrix
                .EnumerateRows()
                .Select(centroid => Extensions.DistanceToCentroid(pixelMatrix, centroid))
        );

        //Result Vector<int> : index of closest centroid
        var closestCentroids = Vector<int>.Build.DenseOfEnumerable
        (
            centroidDistanceMatrix
                .EnumerateColumns()
                .Select(pixelDistancesColumn => pixelDistancesColumn.MinimumIndex())
        );

        return closestCentroids;
    }

    public Matrix<double> ComputeCentroids(Matrix<double> pixels, Vector<double> closestCentroids)
    {
        var kMeansEnumerable = pixels.EnumerateRowsIndexed()
            .GroupBy(pixel => closestCentroids[pixel.Item1], x => x.Item2)
            .Select(Matrix<double>.Build.DenseOfRowVectors)
            .Select(cluster => cluster.ColumnSums().Divide(cluster.RowCount));

        var kMeans = Matrix<double>.Build.DenseOfRowVectors(kMeansEnumerable);

        return kMeans;
    }

    public void Load()
    {

    }
}