using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using ScottPlot;
using Color = System.Drawing.Color;

namespace MultipleLinearRegressionWithGradientDescent
{
    public record DataPoint(double X, double Y);

    /// <summary>
    /// Interaction logic for GenericLinearRegression.xaml
    /// </summary>
    public partial class GenericLinearRegression : Window
    {
        private readonly Matrix<double> _inputX;
        private readonly Vector<double> _inputY;
        public ObservableCollection<DataPoint> Records { get; set; }

        public GenericLinearRegression()
        {
            InitializeComponent();

            using var streamReader = new StreamReader("Data/primary.csv");
            using var csvReader = new CsvHelper.CsvReader(streamReader, CultureInfo.InvariantCulture, false);
            Records = new ObservableCollection<DataPoint>(csvReader.GetRecords<DataPoint>().OrderBy(x=>x.X));

            //int dataPoints = 100;
            //double a = 3, b = 2, c = 1; // Coefficients for the quadratic function y = ax^2 + bx + c
            //Random random = new Random();

            //Matrix<double> testInput = Matrix<double>.Build.Dense(dataPoints, 1, (i, j) => i);
            //Vector<double> testOutput = Vector<double>.Build.Dense(dataPoints, i => a * Math.Pow(i, 2) + b * i + c + random.NextDouble() * 0.1); // Add some random noise to the data

            //_inputX = testInput;
            //_inputY = testOutput;

            _inputX = Matrix<double>.Build.DenseOfRowVectors(Records.Select(x => Vector<double>.Build.DenseOfArray(new[] { x.X })).ToArray());
            _inputY = Vector<double>.Build.DenseOfEnumerable(Records.Select(x => x.Y));
        }

        private void GenericLinearRegression_OnLoaded(object sender, RoutedEventArgs e)
        {
            DataPlot.Plot.AddSignalXY(_inputX.Column(0).ToArray(), _inputY.ToArray());
            DataPlot.Refresh();
        }

        private async void Button_Click(object sender, RoutedEventArgs e)
        {
            RegressionModel model = new RegressionModel();
            model.FeatureMap = RegressionModel.MapFeatureDegree(6);
            //model.BatchSize = 32;

            //model.FeatureMap = RegressionModel.MapFeatureDegree(1);
            model.LearningRate = 1e-1;
            model.TrainingThreshold = 1e-10;
            //model.RegularizationTerm = 500;
            double[] cost = await model.FitAndPlot(CostPlot, _inputX, _inputY, 10000, DataPlot);

            Debug.WriteLine(model.Weight);
            Debug.WriteLine(model.Bias);

            model.SamplePlot(DataPlot);


            var funcStr = string.Join("+",
                new[] { "x", "x^2", "x^3" }.Zip(model.Weight).Select(x => x.First + "*" + x.Second)) + "+" + model.Bias;
            Debug.WriteLine(funcStr);
            }

    }
}
