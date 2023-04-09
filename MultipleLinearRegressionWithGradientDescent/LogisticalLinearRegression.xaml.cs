using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography.Xml;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using OxyPlot;
using ScottPlot;
using Color = System.Drawing.Color;
using Window = System.Windows.Window;

namespace MultipleLinearRegressionWithGradientDescent
{
    public record DataItem(double Left, double Right, double Target);

    /// <summary>
    /// Interaction logic for LogisticalLinearRegression.xaml
    /// </summary>
    public partial class LogisticalLinearRegression : Window
    {


        private readonly LogisticalRegressionModel _model;
        public List<DataItem> Records { get; private set; }
        private readonly Matrix<double> _inputX;
        private readonly Vector<double> _inputY;
        private readonly string dataFile;

        public LogisticalLinearRegression(int datasetNumber)
        {
            _model = new LogisticalRegressionModel();
            if (datasetNumber == 1)
            {
                dataFile = "ex2data1.txt";
                _model.TrainingThreshold = 1e-7;
                _model.LearningRate = 1;
            }
            else
            {
                dataFile = "ex2data2.txt";
                _model.TrainingThreshold = 1e-7;
                _model.LearningRate = 1e-3;
            }
            _model.FeatureMap = RegressionModel.MapFeatureDegree(12);

            var valuesArray = File.ReadAllLines(dataFile).Select(x => x.Split(',').Select(double.Parse).ToArray()).ToArray();
            _inputX = Matrix<double>.Build.DenseOfRows(valuesArray.GetLength(0), valuesArray.First().Length - 1,
                valuesArray.Select(a => a[..^1]));
            _inputY = Vector<double>.Build.Dense(valuesArray.Select(x => x.Last()).ToArray());

            Records = valuesArray
                .Select(x => new DataItem(x[0], x[1], x[2])).ToList();


            InitializeComponent();

            PlotNormalData();
        }

        private void PlotNormalData()
        {
            var normalData = new NormalMatrix(_inputX).Normal;
            normalData.EnumerateRows().ToList()
                .Zip(_inputY)
                .ToList()
                .ForEach(x => DataPlot.Plot.AddMarker(x.First[0], x.First[1],
                    x.Second >= 0.5 ? MarkerShape.filledCircle : MarkerShape.cross, 10,
                    x.Second >= 0.5 ? Color.Blue : Color.Red));

            DataPlot.Refresh();
        }

        private async void Descent_Click(object sender, RoutedEventArgs e)
        {
            await _model.FitAndPlot(CostPlot, _inputX, _inputY);
            _model.PlotBoundary(DataPlot);
        }


        private void Control_OnMouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            var dataItem = (data.SelectedItem as DataItem);
            double output = _model.Predict(Vector<double>.Build.Dense(new[] { dataItem.Left, dataItem.Right }));
            MessageBox.Show(output + " " + (output >= 0.5));
        }
    }
}
