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
using ScottPlot;
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

            
            _input = new NormalMatrix(Matrix<double>.Build.DenseOfRowArrays(inputData.Select(x => x[..2]))).Normal;
            _target = Vector<double>.Build.DenseOfEnumerable(inputData.Select(x => x[2]+1));
        }


        private void MainWindow_OnLoaded(object sender, RoutedEventArgs e)
        {
            var negativeList = DataPlot.Plot.AddScatterList(Color.Red, lineStyle: LineStyle.None);
            var positiveList = DataPlot.Plot.AddScatterList(Color.Green, lineStyle: LineStyle.None);

            for (int i = 0; i < _input.RowCount; i++)
            {
                if (_target[i] == 1)
                    negativeList.Add(_input[i,0], _input[i,1]);
                else
                    positiveList.Add(_input[i, 0], _input[i, 1]);
            }

            DataPlot.Refresh();
        }

        private async void ButtonBase_OnClick(object sender, RoutedEventArgs e)
        {
            NeuralNetworkModel model = new NeuralNetworkModel(
                new DenseLayer(2, 8, NeuralNetworkModel.ReLU),
                new DenseLayer(8, 100, NeuralNetworkModel.ReLU),
                new DenseLayer(100, 200, NeuralNetworkModel.ReLU),
                new DenseLayer(200, 8, NeuralNetworkModel.ReLU),
                new DenseLayer(8, 2, NeuralNetworkModel.Linear))
            {
                LearningRate = 1e-4
            };

            model.Epoch(_input, _target);

            await model.FitAndPlot(CostPlot, _input, _target, 100, DataPlot, 10000);

            //model.PlotBoundary(DataPlot);
        }
    }
}
