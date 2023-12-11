using AnomalyDetection.Client.ServiceContracts;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;

namespace AnomalyDetection.Client.Business;

public static class ServiceCollectionExtensions
{
    public static void RegisterServices(this IServiceCollection services)
    {
        services.AddScoped(sp =>
        {
            //sp.GetRequiredService<IHostEnvironment>()
            var hostEnvironment = sp.GetRequiredService<IWebAssemblyHostEnvironment>();
            return new HttpClient { BaseAddress = new Uri(hostEnvironment.BaseAddress) };
        });
        services.AddScoped<FileReader>();
        services.AddScoped<IAnomalyDetector, AnomalyDetector>();
        services.AddScoped<IMatrixLoader, CsvMatrixLoader>();
        //        builder.Services.AddScoped<IMatrixLoader, NumPyMatrixLoader>();
    }
}