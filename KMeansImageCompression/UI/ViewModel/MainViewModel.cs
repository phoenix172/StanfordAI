using System;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using KMeansImageCompression.Data;
using KMeansImageCompression.Model;
using Microsoft.Win32;

namespace KMeansImageCompression;

[ObservableObject]
public partial class MainViewModel
{
    [NotifyCanExecuteChangedFor(nameof(SaveCompressedCommand))]
    [ObservableProperty] private BitmapSource? _compressedImage;
    [ObservableProperty] private BitmapSource _originalImage;
    [ObservableProperty] private int _iterations = 24;
    [ObservableProperty] private int _colors = 16;


    public MainViewModel()
        : this(TestData.OriginalImage.To24BitFormat())
    {
    }

    public MainViewModel(BitmapSource image)
    {
        OriginalImage = image;
    }

    [RelayCommand]
    public void OpenImage()
    {
        OpenFileDialog fileDialog = new OpenFileDialog();
        if (fileDialog.ShowDialog() == true)
        {
            OriginalImage = new BitmapImage(new Uri(fileDialog.FileName));
        }
    }

    public bool CanSaveCompressed => CompressedImage != null;

    [RelayCommand(CanExecute = nameof(CanSaveCompressed))]
    public void SaveCompressed()
    {
        if (CompressedImage == null) return;

        SaveFileDialog fileDialog = new SaveFileDialog()
        {
            Filter = $"Image | *{Constants.OutputImageFileExtension}"
        };
        if (fileDialog.ShowDialog() == true)
        {
            CompressedImage.SaveToFile(fileDialog.FileName);
        }
    }

    [RelayCommand(AllowConcurrentExecutions = false)]
    public async Task CompressImage()
    {
        var a = new CompressedImage(OriginalImage, new()
        {
            IterationsCount = Iterations,
            TargetColorsCount = Colors
        });

        try
        {
            CompressedImage = await a.CompressImageAsync();
        }
        catch(Exception ex)
        {
            MessageBox.Show("Failed to compress image: "+ex.Message);
        }
    }
}