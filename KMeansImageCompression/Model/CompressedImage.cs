﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using KMeansImageCompression.Data;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;
using NumSharp;

namespace KMeansImageCompression.Model;

public class CompressedImage
{
    private readonly Matrix<double> _pixelMatrix;
    private readonly Matrix<double> _centroids;
    private readonly BitmapSource _imageSource;

    public CompressedImage(string path)
        : this(new Uri(Path.Combine(Directory.GetCurrentDirectory(), path)))
    {
    }

    public CompressedImage(Uri uri)
        : this(new BitmapImage(uri))
    {
    }

    public CompressedImage(BitmapSource bitmapSource)
    {
        _imageSource = bitmapSource;

        var pixels = _imageSource.To24BitFormat().GetPixels();
        _pixelMatrix = pixels.ToMatrix();
        _centroids = CreateCentroidMatrix(_pixelMatrix, ColorsCount);
    }

    public int ColorsCount { get; init; } = 16;

    public BitmapSource CompressImage(int iterations = 3)
    {
        IterationResult result = KMeansIteration(_centroids);

        for (int i = 0; i < iterations; i++)
        {
            result = KMeansIteration(result);
        }

        var compressedImage = result.ToMatrix().ToBitmapImage(_imageSource);

        return compressedImage;
    }

    private record IterationResult(Matrix<double> Centroids, uint[] ClosestCentroidIndices)
    {
        public Matrix<double> ToMatrix()
        {
            var matrix = Matrix<double>.Build.DenseOfRowVectors
            (
                ClosestCentroidIndices.Select(x => Centroids.Row((int)x))
            );
            return matrix;
        }
    }

    private IterationResult KMeansIteration(IterationResult result)
        => KMeansIteration(result.Centroids);

    private IterationResult KMeansIteration(Matrix<double> currentCentroids)
    {
        uint[] closestCentroidIndices = FindClosestCentroids(_pixelMatrix, currentCentroids); //new color index for each pixel
        var kMeans = ComputeCentroids(_pixelMatrix, closestCentroidIndices); //new colors
        return new(kMeans, closestCentroidIndices);
    }

    public Matrix<double> CreateCentroidMatrix(Matrix<double> pixels, int centroidsCount)
    {
        Random random = new Random(DateTime.Now.ToFileTimeUtc().GetHashCode());
        var randomPixelIndices =
            Enumerable.Range(0, centroidsCount).Select(x => (int)random.NextInt64(0, pixels.RowCount - 1));

        var randomPixels = Matrix<double>.Build.DenseOfRowVectors(randomPixelIndices.Select(pixels.Row));

        return randomPixels;
    }

    public uint[] FindClosestCentroids(Matrix<double> pixelMatrix, Matrix<double> centroidMatrix)
    {
        Matrix<double> centroidDistanceMatrix = Extensions.CalculateEuclideanDistances(pixelMatrix, centroidMatrix);

        var closestCentroids = centroidDistanceMatrix
            .EnumerateRows()
            .Select(pixelDistancesColumn => (uint)pixelDistancesColumn.MinimumIndex())
            .ToArray();

        return closestCentroids;
    }

    public Matrix<double> ComputeCentroids(Matrix<double> pixels, uint[] closestCentroidIndices)
    {
        int k = ColorsCount;
        int d = pixels.ColumnCount;
        var sums = new double[k, d];
        var counts = new int[k];

        for (int i = 0; i < pixels.RowCount; i++)
        {
            var clusterIndex = closestCentroidIndices[i];
            for (int j = 0; j < d; j++)
            {
                sums[clusterIndex, j] += pixels[i, j];
            }
            counts[clusterIndex]++;
        }
        var centroids = Matrix<double>.Build.Dense(k, d, (i, j) => counts[i] > 0 ? sums[i, j] / counts[i] : 0);

        return centroids;
    }
}