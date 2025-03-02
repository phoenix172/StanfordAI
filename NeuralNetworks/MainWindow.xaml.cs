using System;
using System.Collections.Generic;
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
using MathNet.Numerics.LinearAlgebra;
using MultipleLinearRegressionWithGradientDescent;
using OxyPlot;
using OxyPlot.Annotations;
using OxyPlot.Series;
using Color = System.Drawing.Color;

namespace NeuralNetworks
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private readonly Matrix<double> _input;
        private readonly Vector<double> _target;

        public MainWindow()
        {
            InitializeComponent();

            var inputData = File.ReadAllLines("spiral.csv").Skip(1)
                .Select(x => x.Split(',').Select(double.Parse).ToArray()).ToArray();

            var denseOfRowArrays = Matrix<double>.Build.DenseOfRowArrays(
                inputData.Select(x => x[..2]));

            var normalMatrix = new NormalMatrix(denseOfRowArrays);
            _input = normalMatrix.Normal;

            _target = Vector<double>.Build.DenseOfEnumerable(inputData.Select(x => x[2] + 1));
        }


        private void MainWindow_OnLoaded(object sender, RoutedEventArgs e)
        {
            DataPlot.Model = new PlotModel();
            var positiveList = new ScatterSeries()
            {
                MarkerFill = OxyColor.FromUInt32((uint)Color.Green.ToArgb()),
                MarkerType = MarkerType.Circle
            };

            var negativeList = new ScatterSeries()
            {
                MarkerFill = OxyColor.FromUInt32((uint)Color.Red.ToArgb()),
                MarkerType = MarkerType.Circle
            };
            DataPlot.Model.Series.Add(positiveList);
            DataPlot.Model.Series.Add(negativeList);

            for (int i = 0; i < _input.RowCount; i++)
            {
                if (_target[i] == 1)
                {
                    DataPlot.Model.Annotations.Add(new PointAnnotation()
                    {
                        X = _input[i, 0],
                        Y = _input[i, 1],
                        Fill = OxyColors.Red,
                        Layer = AnnotationLayer.AboveSeries
                    });
                    //negativeList.Points.Add(new ScatterPoint(_input[i, 0], _input[i, 1]));
                }
                else
                {
                    DataPlot.Model.Annotations.Add(new PointAnnotation()
                    {
                        X = _input[i, 0],
                        Y = _input[i, 1],
                        Fill = OxyColors.Green,
                        Layer = AnnotationLayer.AboveSeries
                    });
                    //positiveList.Points.Add(new ScatterPoint(_input[i, 0], _input[i, 1]));
                }
            }

            DataPlot.Model.InvalidatePlot(true);
        }

        private async void ButtonBase_OnClick(object sender, RoutedEventArgs e)
        {
            NeuralNetworkModel model = new NeuralNetworkModel(
                new DenseLayer(4, 8, NeuralNetworkModel.ReLU),
                new DenseLayer(8, 8, NeuralNetworkModel.ReLU),
                new DenseLayer(8, 8, NeuralNetworkModel.ReLU),
                new DenseLayer(8, 4, NeuralNetworkModel.ReLU),
                new DenseLayer(4, 2, NeuralNetworkModel.Linear))
            {
                LearningRate = 1e-4
            };

            model.FeatureMap = x =>
                Vector<double>.Build.DenseOfEnumerable(x.Append(Math.Sin(x[0])).Append(Math.Sin(x[1])));

            CostPlot.Model = new PlotModel();
            await model.FitAndPlot(CostPlot.Model, DataPlot.Model, _input, _target, 10, 10000);

            //model.PlotBoundary(DataPlot);
        }
    }
}