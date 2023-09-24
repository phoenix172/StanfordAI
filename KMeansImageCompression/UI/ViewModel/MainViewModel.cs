using System.Windows.Media;
using System.Windows.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using KMeansImageCompression.Data;
using KMeansImageCompression.Model;

namespace KMeansImageCompression;

[ObservableObject]
public partial class MainViewModel
{
    [ObservableProperty]private BitmapSource _compressedImage;

    public MainViewModel()
        : this(TestData.Get24BitOriginalImage(), TestData.Get24BitOriginalImage())
    {
    }

    public MainViewModel(BitmapSource originalImage, BitmapSource compressedImage)
    {
        OriginalImage = originalImage;
        _compressedImage = compressedImage;
    }

    public BitmapSource OriginalImage { get; }

    [RelayCommand]
    public void CompressImage()
    {
        var a = new ImageCompressorService();
        CompressedImage = a.CompressImage(OriginalImage);
    }

    
}