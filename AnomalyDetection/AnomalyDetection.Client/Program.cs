using AnomalyDetection.Core;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using Microsoft.AspNetCore.Hosting;
using MudBlazor.Services;

namespace AnomalyDetection.Client;

internal class Program
{
    public static async Task Main(string[] args)
    {
        var builder = WebAssemblyHostBuilder.CreateDefault(args);
        
        
        builder.Services
            .AddScoped(_ => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });
        // builder.Services.AddScoped<DataConfiguration>(svc =>
        //     new DataConfiguration(svc.GetRequiredService<IWebHostEnvironment>().WebRootPath));
        builder.Services.RegisterAnomalyDetection();
        
        builder.Services.AddMudServices();
        
        await builder.Build().RunAsync();
    }
}