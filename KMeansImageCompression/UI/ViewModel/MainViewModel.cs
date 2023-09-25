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
    [ObservableProperty] private BitmapSource _compressedImage;

    public MainViewModel()
        : this(TestData.OriginalImage.To24BitFormat())
    {
    }

    public MainViewModel(BitmapSource image)
    {
        OriginalImage = image;
        _compressedImage = image;
    }

    public BitmapSource OriginalImage { get; }

    [RelayCommand]
    public void CompressImage()
    {
        var a = new CompressedImage(OriginalImage);
        CompressedImage = a.CompressImage(24);

        CompressedImage.SaveToFile("compressed.png");
    }
}