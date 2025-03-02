using AnomalyDetection.Core.IO;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
namespace AnomalyDetection.Core;

public static class ServiceCollectionExtensions
{
    public static IServiceCollection RegisterServices(this IServiceCollection services)
    {
        services
            .AddScoped<IFileReader, FileReader>(serviceProvider =>
            {
                string contentRootPath = serviceProvider.GetRequiredService<IHostingEnvironment>().WebRootPath;
                return new FileReader(contentRootPath);
            })
            .AddScoped<IAnomalyDetector, AnomalyDetector>()
            .AddScoped<IMatrixLoader, CsvMatrixLoader>();
        //.AddScoped<IMatrixLoader, NumPyMatrixLoader>();
        return services;
    }
}