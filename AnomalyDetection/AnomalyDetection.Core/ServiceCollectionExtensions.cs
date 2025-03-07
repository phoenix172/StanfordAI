using AnomalyDetection.Core.IO;
using Microsoft.Extensions.DependencyInjection;

namespace AnomalyDetection.Core;

public static class ServiceCollectionExtensions
{
    public static IServiceCollection RegisterAnomalyDetection(this IServiceCollection services)
    {
        services
            .AddScoped<IFileReader, FileReader>()
            .AddScoped<IAnomalyDetector, AnomalyDetector>()
            .AddScoped<IMatrixLoader, CsvMatrixLoader>();
        
        return services;
    }
}