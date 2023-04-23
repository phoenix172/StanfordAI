using System;
using System.CodeDom;
using System.Collections;
using System.Diagnostics;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using ScottPlot;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Data;
using System.Drawing;
using System.Runtime.CompilerServices;
using OxyPlot;
using ScottPlot.Plottable;

namespace MultipleLinearRegressionWithGradientDescent;

public static class Extensions
{
    public static string FormatAsString(this IEnumerable input, string delimiter = ",")
    {
        var enumerator = input.GetEnumerator();
        enumerator.MoveNext();
        var first = enumerator.Current;

        var values = first switch
        {
            IEnumerable => input.Cast<IEnumerable>().Select(x=>x.FormatAsString(delimiter)),
            _ => input.Cast<object>().Select(x => x.ToString()),
        };

        return "[" + string.Join(first is IEnumerable ? "\n" : delimiter, values) + "]";
    }


    public static ScatterPlotList<double> SamplePlot(this RegressionModel model, WpfPlot plot, int samples =1000)
    {
        var scatterList = plot.Plot.AddScatterList(Color.Red, markerSize: 2f, lineStyle: LineStyle.Solid,
            lineWidth: 2f, markerShape: MarkerShape.none);
        model.SamplePlot(scatterList, samples);
        App.Current.Dispatcher.Invoke(() => plot.RenderRequest());
        return scatterList;
    }

    public static void SamplePlot(this RegressionModel model, ScatterPlotList<double> plotList, int samples = 1000, bool clear = true)
    {
        if (model.Weight == null)
            return;

        if (model.OriginalTrainingInput.ColumnCount > 1)
            throw new ArgumentException("Only allowed for models with 1 feature");

        int min = (int)Math.Round(model.OriginalTrainingInput.Column(0).Min());
        int max = (int)Math.Round(model.OriginalTrainingInput.Column(0).Max())+100;

        var xs = Extensions.RangeStep(-max, max, samples);
        var ys = xs.Select(x =>
        {
            var inputVector = Vector<double>.Build.DenseOfArray(new[] { x });
            var y = model.Predict(inputVector);

            if (double.IsNaN(y) || double.IsInfinity(y)) return 0;
            return y;
        }).ToArray();

        if(clear)plotList.Clear();

        plotList.AddRange(xs, ys);
    }

    public static async Task<double[]> FitAndPlot(this RegressionModel model, WpfPlot costPlot,
        Matrix<double> trainingInput, Vector<double> trainingOutput, int chunkSize = 1000, WpfPlot? dataPlot = null)
    {
        var costList = costPlot.Plot.AddScatterList(markerSize: 1.5f, lineStyle: LineStyle.Solid,
            lineWidth: 0.5f, markerShape: MarkerShape.none);
        ScatterPlotList<double>? functionPlot = null;

        if (dataPlot != null)
            functionPlot = model.SamplePlot(dataPlot);

        double[] PlotChunk(double[] x, int i)
        {
            var costs = x.ToList();
            Debug.WriteLine($"iteration {i}: cost {x.Last()}");

            var xs = Enumerable.Range(chunkSize * i, costs.Count).Select(y => (double)y).ToArray();
            var ys = costs.ToArray();

            var finite = xs.Zip(ys).Where(x => double.IsFinite(x.Second) && !double.IsNaN(x.Second));
            costList.AddRange(finite.Select(x => x.First).ToArray(), finite.Select(x => x.Second).ToArray());

            if (dataPlot != null) 
                model.SamplePlot(functionPlot);

            App.Current.Dispatcher.Invoke(() =>
            {
                costPlot.RenderRequest();
                dataPlot?.RenderRequest();
            });


            return x;
        }

        var cost = await Task.Run(() =>
        {
            var costs = model.Fit(trainingInput, trainingOutput).Chunk(chunkSize)
                .Select(PlotChunk).ToList();
            return costs.Last();
        });
        return cost;
    }


    public static T1[,] Grid<T, T1>(this IEnumerable<T> input, Func<T,T,T1> valueFunc)
    {
        var data = input.ToList();
        var result = new T1[data.Count,data.Count];
        for (int i = 0; i < data.Count; i++)
        {
            for (int j = 0; j < data.Count; j++)
            {
                result[i,j] = valueFunc(data[i], data[j]);
            }
        }

        return result;
    }

    public static void PlotBoundary(this LogisticalRegressionModel model, WpfPlot plot)
    {
        var rowMajorOriginal = model.OriginalTrainingInput.ToRowMajorArray();
        var min = rowMajorOriginal.Min();
        var max = rowMajorOriginal.Max();

        min -= Math.Abs(min) * 0.25;
        max += Math.Abs(max) * 0.25;

        var range = RangeStep(min, max ,100);

        double minZ = double.MaxValue;
        double maxZ = double.MinValue;
        var valuesGrid = range
            .Grid((x, y) =>
            {
                var z = model.Predict(Vector<double>.Build.Dense(new[] { x, y }));
                minZ = Math.Min(z, minZ);
                maxZ = Math.Max(z, maxZ);
                return z;
            });

        Conrec.Contour(valuesGrid, range.ToArray(), range.ToArray(), new[] { 0.5 }, model.PlotNormalLine(plot));
        plot.Refresh();
    }



    public static double[] RangeStep(double start, double end, int steps)
    {
        double stepSize = Math.Abs(start - end) / steps;
        return Enumerable.Range(0, steps).Select(x => start + x * stepSize).ToArray();
    }

    public static Conrec.RendererDelegate PlotNormalLine(this RegressionModel model, WpfPlot plot) =>
        (x1, y1, x2, y2, z) =>
        {
            var a = model.NormalizedInput.NormalizeRow(model.FeatureMap(Vector<double>.Build.Dense(new[] { x1, y1 })));
            var b = model.NormalizedInput.NormalizeRow(model.FeatureMap(Vector<double>.Build.Dense(new[] { x2, y2 })));
            plot.Plot.AddLine(a[0], a[1], b[0], b[1], Color.Blue);
        };

}